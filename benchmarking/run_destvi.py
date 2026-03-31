#!/usr/bin/env python3
"""
DestVI Spatial Deconvolution Pipeline with Cell State Tracking
================================================================================
OUTPUTS:
1. Cell Type Proportions (β): Abundance of each cell type per spot
2. Cell-Type-Specific States (γ): Latent representation of cell state variation

Key outputs:
- proportions.csv: Cell type proportions per spot
- cell_states/gamma_states_<cell_type>.csv: Cell-type-specific latent states (FILTERED)
- cell_states/spatial_state_<cell_type>.png: Spatial visualization of states
- cell_states/continuum_<cell_type>.png: Cell-type-specific UMAP (using only that cell type's data)
- cell_states/continuum_global_<cell_type>.png: Global UMAP colored by cell type
================================================================================
"""

import argparse
import scvi
import scanpy as sc
import pandas as pd
import anndata
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import math
import warnings
import torch
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ==============================================================================
# GLOBAL MATPLOTLIB FONT SETTINGS
# ==============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 16,
    'font.weight': 'bold',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.linewidth': 1.5,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'legend.fontsize': 14,
    'legend.title_fontsize': 15,
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    scvi.settings.seed = seed


# ==============================================================================
# CELL STATE EXTRACTION - CORE IMPROVEMENT
# ==============================================================================

class DestVICellStateExtractor:
    """
    Extract cell-type-specific latent states from DestVI model.
    
    DestVI models:
    - β (proportions): Abundance of each cell type in each spot
    - γ (gamma/states): Cell-type-specific latent representation capturing
                        continuous variation in cell state
    
    The gamma values represent WHERE along a cell type's expression program
    continuum each spot falls (e.g., activation state, differentiation stage).
    """
    
    def __init__(self, st_model, adata_st):
        """
        Initialize extractor.
        
        Args:
            st_model: Trained DestVI model
            adata_st: Spatial AnnData object used for training
        """
        self.st_model = st_model
        self.adata_st = adata_st
        self.cell_types = list(st_model.cell_type_mapping)
        self.n_spots = adata_st.n_obs
        self.spot_names = adata_st.obs_names.tolist()
        
    def get_proportions(self) -> pd.DataFrame:
        """
        Extract cell type proportions (β) from DestVI.
        
        Returns:
            DataFrame with shape (n_spots, n_cell_types)
        """
        props = self.st_model.get_proportions()
        # Handle different return types
        if hasattr(props, 'values'):
            props = props.values
        elif hasattr(props, 'numpy'):
            props = props.numpy()
        elif hasattr(props, 'cpu'):
            props = props.cpu().numpy()
        props = np.array(props)
        
        return pd.DataFrame(
            props,
            index=self.spot_names,
            columns=self.cell_types
        )
    
    def get_gamma_states(self, method='scale_pca', n_components=10) -> dict:
        """
        Extract cell-type-specific latent states (γ) from DestVI.
        
        DestVI learns gamma through its decoder - we can extract it by:
        1. 'scale_pca': Get cell-type-specific expression scale, then PCA
        2. 'direct': Attempt to access internal gamma (if available in API)
        
        Args:
            method: Extraction method ('scale_pca' or 'direct')
            n_components: Number of latent dimensions to extract
            
        Returns:
            Dictionary mapping cell_type -> DataFrame of gamma values
        """
        gamma_dict = {}
        
        for ct in self.cell_types:
            print(f"    Extracting gamma for: {ct}")
            if method == 'scale_pca':
                gamma_df = self._extract_gamma_via_scale(ct, n_components)
            elif method == 'direct':
                gamma_df = self._extract_gamma_direct(ct, n_components)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            gamma_dict[ct] = gamma_df
            
        return gamma_dict
    
    def _extract_gamma_via_scale(self, cell_type: str, n_components: int) -> pd.DataFrame:
        """
        Extract gamma by analyzing cell-type-specific expression scale.
        
        The scale represents the expected gene expression for this cell type
        at each spot. Variation in scale reflects variation in cell state (gamma).
        """
        # Get cell-type-specific expected expression
        # Handle different scvi-tools API versions
        try:
            # Try newer API with return_numpy argument
            scale = self.st_model.get_scale_for_ct(cell_type, return_numpy=True)
        except TypeError:
            # Older API without return_numpy argument
            scale = self.st_model.get_scale_for_ct(cell_type)
        
        # Convert to numpy array regardless of return type
        if hasattr(scale, 'values'):
            # It's a pandas DataFrame
            scale = scale.values
        elif hasattr(scale, 'numpy'):
            # It's a tensor with numpy() method
            scale = scale.numpy()
        elif hasattr(scale, 'cpu'):
            # It's a CUDA tensor
            scale = scale.cpu().numpy()
        
        # Ensure it's a numpy array
        scale = np.array(scale, dtype=np.float32)
        
        # Log-transform for stability
        scale_log = np.log1p(scale * 1e4)
        
        # Handle non-finite values
        scale_log = np.nan_to_num(scale_log, nan=0, posinf=0, neginf=0)
        
        # Standardize
        scaler = StandardScaler()
        scale_scaled = scaler.fit_transform(scale_log)
        scale_scaled = np.nan_to_num(scale_scaled, nan=0, posinf=0, neginf=0)
        
        # PCA to extract latent dimensions
        n_comp = min(n_components, scale_scaled.shape[1] - 1, scale_scaled.shape[0] - 1)
        n_comp = max(1, n_comp)
        
        pca = PCA(n_components=n_comp, random_state=42)
        gamma = pca.fit_transform(scale_scaled)
        
        # Create DataFrame
        col_names = [f"gamma_{i}" for i in range(gamma.shape[1])]
        return pd.DataFrame(gamma, index=self.spot_names, columns=col_names)
    
    def _extract_gamma_direct(self, cell_type: str, n_components: int) -> pd.DataFrame:
        """
        Attempt to directly access gamma from model internals.
        Falls back to scale_pca if not available.
        """
        try:
            # Try to access the model's internal gamma representation
            # This depends on scvi-tools version and API
            
            # Get the cell type index
            ct_idx = self.cell_types.index(cell_type)
            
            # Access the spatial module
            module = self.st_model.module
            
            # Try to get gamma through the model's inference
            # This is version-dependent
            with torch.no_grad():
                # Get data loader
                dl = self.st_model._make_data_loader(self.adata_st)
                
                gamma_list = []
                for batch in dl:
                    # Move to device
                    batch = {k: v.to(module.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Get inference outputs
                    inference_outputs = module.inference(**batch)
                    
                    # Try to access gamma
                    if 'gamma' in inference_outputs:
                        gamma_batch = inference_outputs['gamma']
                        if isinstance(gamma_batch, dict):
                            gamma_batch = gamma_batch.get(ct_idx, gamma_batch.get(cell_type))
                        gamma_list.append(gamma_batch.cpu().numpy())
                
                if gamma_list:
                    gamma = np.vstack(gamma_list)
                    col_names = [f"gamma_{i}" for i in range(gamma.shape[1])]
                    return pd.DataFrame(gamma, index=self.spot_names, columns=col_names)
                    
        except Exception as e:
            print(f"    Direct gamma extraction failed for {cell_type}: {e}")
            print(f"    Falling back to scale_pca method")
        
        # Fallback to scale-based extraction
        return self._extract_gamma_via_scale(cell_type, n_components)
    
    def get_imputed_expression(self, cell_type: str) -> pd.DataFrame:
        """
        Get cell-type-specific imputed gene expression.
        
        This represents the expected expression profile for this cell type
        at each spot, given the inferred cell state (gamma).
        
        Args:
            cell_type: Name of cell type
            
        Returns:
            DataFrame with shape (n_spots, n_genes)
        """
        try:
            scale = self.st_model.get_scale_for_ct(cell_type, return_numpy=True)
        except TypeError:
            scale = self.st_model.get_scale_for_ct(cell_type)
        
        # Convert to numpy
        if hasattr(scale, 'values'):
            scale = scale.values
        elif hasattr(scale, 'numpy'):
            scale = scale.numpy()
        elif hasattr(scale, 'cpu'):
            scale = scale.cpu().numpy()
        scale = np.array(scale)
        
        return pd.DataFrame(
            scale,
            index=self.spot_names,
            columns=self.adata_st.var_names
        )
    
# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

class VisualizationUtils:
    """Visualization utilities for spatial deconvolution results."""
    
    @staticmethod
    def _calculate_hex_radius(coords: np.ndarray, scale_factor: float = 0.6) -> float:
        """Auto-calculate hexagon radius from spot spacing."""
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=2)
        nn_distances = distances[:, 1]
        median_spacing = np.median(nn_distances)
        return median_spacing * scale_factor
    
    @staticmethod
    def plot_cooccurrence(df_props: pd.DataFrame, output_path: str):
        """Generates a correlation heatmap of cell type proportions."""
        plt.figure(figsize=(10, 8))
        corr = df_props.corr(method='pearson')
        
        sns.heatmap(
            corr, 
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            vmin=-1,
            vmax=1,
            fmt='.2f', 
            square=True,
            linewidths=.5,
            annot_kws={'size': 10, 'weight': 'bold'}
        )
        plt.title("Cell Type Co-occurrence (Correlation)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    
    @staticmethod
    def plot_spatial_maps(
        df_props: pd.DataFrame, 
        coords: np.ndarray, 
        output_dir: str,
        coords_full: np.ndarray = None,
        matched_mask: np.ndarray = None,
        hex_orientation: float = 0.0
    ):
        """Generates spatial maps with HEXAGONAL spot markers."""
        cell_types = df_props.columns
        n_types = len(cell_types)
        
        # Convert orientation from degrees to radians
        orientation_rad = np.radians(hex_orientation)
        
        coords_for_limits = coords_full if coords_full is not None else coords
        hex_radius = VisualizationUtils._calculate_hex_radius(coords_for_limits, scale_factor=0.55)
        
        x_min = coords_for_limits[:, 0].min() - hex_radius * 2
        x_max = coords_for_limits[:, 0].max() + hex_radius * 2
        y_min = coords_for_limits[:, 1].min() - hex_radius * 2
        y_max = coords_for_limits[:, 1].max() + hex_radius * 2
        
        def add_background_spots(ax, coords_full, matched_mask, hex_radius, orientation):
            if coords_full is None or matched_mask is None:
                return
            unmatched_mask = ~matched_mask
            if not unmatched_mask.any():
                return
            unmatched_coords = coords_full[unmatched_mask]
            patches = []
            for x, y in unmatched_coords:
                hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=orientation)
                patches.append(hexagon)
            collection = PatchCollection(patches, facecolors='lightgrey', edgecolors='none', alpha=0.3)
            ax.add_collection(collection)
        
        # Grid Plot (Intensity per type)
        print(f"  Generating spatial intensity maps for {n_types} cell types...")
        print(f"  Using hexagon orientation: {hex_orientation}°")
        cols = 4
        rows = math.ceil(n_types / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        axes = axes.flatten() if n_types > 1 else [axes]
        
        for i, ct in enumerate(cell_types):
            ax = axes[i]
            values = df_props[ct].values
            
            add_background_spots(ax, coords_full, matched_mask, hex_radius, orientation_rad)
            
            vmin, vmax = values.min(), values.max()
            if vmax - vmin < 1e-8: vmax = vmin + 1e-8
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.plasma

            patches = []
            colors = []
            for j, (x, y) in enumerate(coords):
                hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=orientation_rad)
                patches.append(hexagon)
                colors.append(cmap(norm(values[j])))
            
            collection = PatchCollection(patches, facecolors=colors, edgecolors='none')
            ax.add_collection(collection)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')
            ax.set_title(ct, fontsize=14, fontweight='bold', pad=10)
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.85, aspect=15)
            cbar.locator = ticker.MaxNLocator(nbins=5)
            cbar.formatter = ticker.FormatStrFormatter('%.2f')
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=10)
            
        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spatial_intensity_maps.png"), dpi=300)
        plt.close()

        # Dominant Cell Type Map
        print("  Generating dominant cell type map...")
        dominant_idx = np.argmax(df_props.values, axis=1)
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap_cat = plt.colormaps.get_cmap('tab20').resampled(n_types)
        
        add_background_spots(ax, coords_full, matched_mask, hex_radius, orientation_rad)
        
        patches = []
        colors = []
        for j, (x, y) in enumerate(coords):
            hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=orientation_rad)
            patches.append(hexagon)
            colors.append(cmap_cat(dominant_idx[j] / max(n_types - 1, 1)))
        
        collection = PatchCollection(patches, facecolors=colors, edgecolors='none')
        ax.add_collection(collection)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        
        handles = [plt.Line2D([0], [0], marker='H', color='w', markerfacecolor=cmap_cat(i / max(n_types - 1, 1)),
                              label=cell_types[i], markersize=12, markeredgecolor='none') for i in range(n_types)]
        
        if coords_full is not None and matched_mask is not None and (~matched_mask).any():
            handles.append(plt.Line2D([0], [0], marker='H', color='w', markerfacecolor='lightgrey', alpha=0.5,
                                      label='No count data', markersize=12, markeredgecolor='none'))
        
        ax.legend(handles=handles, title="Cell Type", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)
        plt.title(f"Dominant Cell Type per Spot (hex orientation: {hex_orientation}°)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spatial_dominant_type.png"), dpi=300, bbox_inches='tight')
        plt.close()


# ==============================================================================
# CELL STATE VISUALIZATION
# ==============================================================================

class CellStateVisualization:
    """
    Comprehensive cell state visualization for DestVI results.
    
    Generates two types of continuum plots:
    1. Cell-type-specific UMAP (continuum_*.png): UMAP computed using ONLY that cell type's gamma
    2. Global UMAP (continuum_global_*.png): UMAP computed using ALL cell types, colored by one type
    """
    
    def __init__(self, extractor: DestVICellStateExtractor, prop_df: pd.DataFrame, 
                 adata_st, output_dir: str, presence_threshold: float = 0.05,
                 hex_orientation: float = 0.0):
        """
        Initialize cell state visualization.
        
        Args:
            extractor: DestVICellStateExtractor instance
            prop_df: DataFrame of cell type proportions
            adata_st: Spatial AnnData object
            output_dir: Output directory path
            presence_threshold: Minimum proportion to consider cell type "present" (default 0.05 = 5%)
            hex_orientation: Hexagon rotation angle in degrees (default 0.0)
        """
        self.extractor = extractor
        self.prop_df = prop_df
        self.adata_st = adata_st
        self.output_dir = output_dir
        self.cell_types = extractor.cell_types
        
        self.state_dir = os.path.join(output_dir, "cell_states")
        os.makedirs(self.state_dir, exist_ok=True)
        
        self.spatial_coords = adata_st.obsm.get('spatial', None)
        self.spatial_coords_full = adata_st.uns.get('spatial_full', None)
        self.matched_mask = adata_st.uns.get('matched_mask', None)
        
        # Use the passed threshold instead of hardcoding
        self.presence_threshold = presence_threshold
        
        # Store hex orientation in radians
        self.hex_orientation = hex_orientation
        self.hex_orientation_rad = np.radians(hex_orientation)
        
        print(f"  Cell state visualization initialized:")
        print(f"    - presence_threshold={self.presence_threshold:.1%}")
        print(f"    - hex_orientation={self.hex_orientation}°")
        
    def _safe_filename(self, cell_type: str) -> str:
        return re.sub(r'[^\w\-_]', '_', cell_type)
    
    def _calculate_hex_radius(self, coords: np.ndarray, scale_factor: float = 0.55) -> float:
        if coords is None or len(coords) < 2:
            return 1.0
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=2)
        nn_distances = distances[:, 1]
        median_spacing = np.median(nn_distances)
        return median_spacing * scale_factor
    
    def save_all_outputs(self, gamma_dict: dict):
        """
        Save all cell state outputs to files.
        
        CSV files contain ONLY filtered spots (proportion >= threshold) without proportion column.
        This is consistent with the Starfysh script output format.
        """
        print("\n" + "="*70)
        print("SAVING CELL STATE OUTPUTS")
        print("="*70)
        
        # 1. Save gamma states for each cell type (FILTERED - only present spots, no proportion column)
        print("\nSaving gamma state CSVs (filtered spots only)...")
        for ct, gamma_df in gamma_dict.items():
            safe_name = self._safe_filename(ct)
            
            # Get proportion mask for this cell type
            props = self.prop_df[ct].values
            present_mask = props >= self.presence_threshold
            n_present = present_mask.sum()
            
            if n_present == 0:
                print(f"  Skipping {ct}: no spots above threshold ({self.presence_threshold:.1%})")
                continue
            
            # Filter to only present spots
            gamma_filtered = gamma_df.loc[present_mask].copy()
            
            # Save WITHOUT proportion column (consistent with Starfysh)
            output_path = os.path.join(self.state_dir, f"gamma_states_{safe_name}.csv")
            gamma_filtered.to_csv(output_path)
            print(f"  Saved: gamma_states_{safe_name}.csv ({n_present} spots)")
        
        # 2. Save summary statistics
        print("\nSaving summary statistics...")
        summary_data = []
        for ct, gamma_df in gamma_dict.items():
            props = self.prop_df[ct].values
            present_mask = props >= self.presence_threshold
            
            summary_data.append({
                'cell_type': ct,
                'n_spots_present': present_mask.sum(),
                'n_spots_total': len(props),
                'presence_threshold': self.presence_threshold,
                'mean_proportion': props[present_mask].mean() if present_mask.any() else 0,
                'gamma_variance': gamma_df.loc[present_mask].var().sum() if present_mask.any() else 0,
                'gamma_range': (gamma_df.loc[present_mask].max() - gamma_df.loc[present_mask].min()).sum() if present_mask.any() else 0,
                'n_gamma_dims': gamma_df.shape[1]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.state_dir, "cell_state_summary.csv"), index=False)
        print(f"  Saved: cell_state_summary.csv")
    
    def compute_global_umap_embedding(self, gamma_dict: dict) -> np.ndarray:
        """
        Compute unified UMAP embedding using ALL cell types' gamma values.
        This is used for global continuum plots.
        """
        print("\nComputing GLOBAL UMAP embedding (all cell types combined)...")
        
        try:
            import umap
            use_umap = True
        except ImportError:
            print("  UMAP not available, using PCA")
            use_umap = False
        
        # Combine all cell types' gamma values
        all_gamma = [gamma_dict[ct].values for ct in self.cell_types]
        combined = np.hstack(all_gamma)
        
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)
        combined_scaled = np.nan_to_num(combined_scaled, nan=0, posinf=0, neginf=0)
        
        if use_umap:
            n_neighbors = min(15, len(combined_scaled) - 1)
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
            embedding = reducer.fit_transform(combined_scaled)
        else:
            pca = PCA(n_components=2, random_state=42)
            embedding = pca.fit_transform(combined_scaled)
        
        print(f"  Global UMAP computed: {embedding.shape}")
        return embedding
    
    def compute_celltype_specific_umap(self, gamma_df: pd.DataFrame, present_mask: np.ndarray) -> np.ndarray:
        """
        Compute UMAP embedding using ONLY this cell type's gamma values for present spots.
        This creates a cell-type-specific embedding.
        
        Args:
            gamma_df: DataFrame of gamma values for this cell type (all spots)
            present_mask: Boolean mask for spots where this cell type is present
            
        Returns:
            UMAP embedding for present spots only
        """
        try:
            import umap
            use_umap = True
        except ImportError:
            use_umap = False
        
        # Get gamma values for present spots only
        gamma_present = gamma_df.values[present_mask]
        
        # Standardize
        scaler = StandardScaler()
        gamma_scaled = scaler.fit_transform(gamma_present)
        gamma_scaled = np.nan_to_num(gamma_scaled, nan=0, posinf=0, neginf=0)
        
        if use_umap:
            n_neighbors = min(15, len(gamma_scaled) - 1)
            n_neighbors = max(2, n_neighbors)  # Ensure at least 2 neighbors
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
            embedding = reducer.fit_transform(gamma_scaled)
        else:
            n_components = min(2, gamma_scaled.shape[1], gamma_scaled.shape[0] - 1)
            n_components = max(1, n_components)
            pca = PCA(n_components=n_components, random_state=42)
            embedding = pca.fit_transform(gamma_scaled)
            if embedding.shape[1] == 1:
                embedding = np.column_stack([embedding, np.zeros(len(embedding))])
        
        return embedding
    
    def plot_cell_type_continuum(self, ct: str, gamma_df: pd.DataFrame):
        """
        Plot CELL-TYPE-SPECIFIC continuum using ONLY this cell type's gamma data.
        This is the primary continuum plot (consistent with Starfysh).
        
        Output: continuum_<cell_type>.png
        """
        safe_name = self._safe_filename(ct)
        
        props = self.prop_df[ct].values
        present_mask = props >= self.presence_threshold
        n_present = present_mask.sum()
        
        if n_present < 10:
            print(f"  Skipping cell-specific continuum plot for {ct}: only {n_present} spots")
            return
        
        # Compute cell-type-specific UMAP using ONLY this cell type's gamma
        print(f"    Computing cell-type-specific UMAP for {ct}...")
        embedding = self.compute_celltype_specific_umap(gamma_df, present_mask)
        
        # Get filtered data
        gamma_filtered = gamma_df.values[present_mask]
        props_filtered = props[present_mask]
        
        # Get primary state dimension
        if gamma_filtered.shape[1] > 1:
            pca = PCA(n_components=1, random_state=42)
            state_values = pca.fit_transform(gamma_filtered).flatten()
        else:
            state_values = gamma_filtered.flatten()
        
        # Calculate consistent point size
        point_size = max(10, min(50, 3000 / n_present))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: State variation
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            embedding[:, 0], embedding[:, 1],
            c=state_values, cmap='coolwarm', s=point_size, alpha=0.8, edgecolors='none'
        )
        ax1.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
        ax1.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
        ax1.set_title(f"{ct}\nCell State (γ) - Cell-Type-Specific UMAP", fontsize=14, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=12)
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label("Latent State", fontsize=12, fontweight='bold')
        cbar1.ax.tick_params(labelsize=10)
        
        # Panel 2: Proportion
        ax2 = axes[1]
        scatter2 = ax2.scatter(
            embedding[:, 0], embedding[:, 1],
            c=props_filtered, cmap='viridis', s=point_size, alpha=0.8, edgecolors='none'
        )
        ax2.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
        ax2.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
        ax2.set_title(f"{ct}\nProportion (β) - Cell-Type-Specific UMAP", fontsize=14, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=12)
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label("Proportion", fontsize=12, fontweight='bold')
        cbar2.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.state_dir, f"continuum_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: continuum_{safe_name}.png (n={n_present}, cell-type-specific UMAP)")
    
    def plot_cell_type_continuum_full(self, ct: str, gamma_df: pd.DataFrame):
        """
        Plot CELL-TYPE-SPECIFIC continuum showing ALL spots.
        Present spots are colored by state/proportion, absent spots are grey.
        
        Output: continuum_full_<cell_type>.png
        """
        safe_name = self._safe_filename(ct)
        
        props = self.prop_df[ct].values
        present_mask = props >= self.presence_threshold
        n_present = present_mask.sum()
        n_absent = (~present_mask).sum()
        n_total = len(props)
        
        if n_present < 10:
            print(f"  Skipping full continuum plot for {ct}: only {n_present} spots present")
            return
        
        # Compute cell-type-specific UMAP using ONLY present spots' gamma
        print(f"    Computing cell-type-specific UMAP (full view) for {ct}...")
        
        # First compute UMAP for present spots
        embedding_present = self.compute_celltype_specific_umap(gamma_df, present_mask)
        
        # For absent spots, we need to project them into the same space
        # Use a simple approach: fit on present, transform on all
        try:
            import umap
            use_umap = True
        except ImportError:
            use_umap = False
        
        gamma_all = gamma_df.values
        gamma_present = gamma_all[present_mask]
        
        scaler = StandardScaler()
        scaler.fit(gamma_present)
        gamma_all_scaled = scaler.transform(gamma_all)
        gamma_all_scaled = np.nan_to_num(gamma_all_scaled, nan=0, posinf=0, neginf=0)
        gamma_present_scaled = gamma_all_scaled[present_mask]
        
        if use_umap:
            n_neighbors = min(15, n_present - 1)
            n_neighbors = max(2, n_neighbors)
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
            reducer.fit(gamma_present_scaled)
            embedding_all = reducer.transform(gamma_all_scaled)
        else:
            n_components = min(2, gamma_present_scaled.shape[1], gamma_present_scaled.shape[0] - 1)
            n_components = max(1, n_components)
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(gamma_present_scaled)
            embedding_all = pca.transform(gamma_all_scaled)
            if embedding_all.shape[1] == 1:
                embedding_all = np.column_stack([embedding_all, np.zeros(len(embedding_all))])
        
        # Get state values for present spots
        gamma_filtered = gamma_all[present_mask]
        props_filtered = props[present_mask]
        
        if gamma_filtered.shape[1] > 1:
            pca_state = PCA(n_components=1, random_state=42)
            state_values_present = pca_state.fit_transform(gamma_filtered).flatten()
        else:
            state_values_present = gamma_filtered.flatten()
        
        # Calculate consistent point size
        point_size = max(10, min(50, 3000 / n_total))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: State variation (all spots)
        ax1 = axes[0]
        
        # Plot absent spots first (grey background)
        if n_absent > 0:
            ax1.scatter(
                embedding_all[~present_mask, 0], embedding_all[~present_mask, 1],
                c='lightgrey', s=point_size, alpha=0.4, edgecolors='none', label='Absent'
            )
        
        # Plot present spots (colored by state)
        scatter1 = ax1.scatter(
            embedding_all[present_mask, 0], embedding_all[present_mask, 1],
            c=state_values_present, cmap='coolwarm', s=point_size, alpha=0.8, edgecolors='none'
        )
        ax1.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
        ax1.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
        ax1.set_title(f"{ct}\nCell State (γ) - All Spots", fontsize=14, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=12)
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label("Latent State", fontsize=12, fontweight='bold')
        cbar1.ax.tick_params(labelsize=10)
        
        # Panel 2: Proportion (all spots)
        ax2 = axes[1]
        
        # Plot absent spots first (grey background)
        if n_absent > 0:
            ax2.scatter(
                embedding_all[~present_mask, 0], embedding_all[~present_mask, 1],
                c='lightgrey', s=point_size, alpha=0.4, edgecolors='none', label='Absent'
            )
        
        # Plot present spots (colored by proportion)
        scatter2 = ax2.scatter(
            embedding_all[present_mask, 0], embedding_all[present_mask, 1],
            c=props_filtered, cmap='viridis', s=point_size, alpha=0.8, edgecolors='none'
        )
        ax2.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
        ax2.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
        ax2.set_title(f"{ct}\nProportion (β) - All Spots", fontsize=14, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=12)
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label("Proportion", fontsize=12, fontweight='bold')
        cbar2.ax.tick_params(labelsize=10)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                      markersize=8, label=f'Present (n={n_present})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey',
                      markersize=8, alpha=0.5, label=f'Absent (n={n_absent})')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
                  fontsize=10, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(os.path.join(self.state_dir, f"continuum_full_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: continuum_full_{safe_name}.png (present={n_present}, absent={n_absent})")
    
    def plot_global_continuum(self, ct: str, gamma_df: pd.DataFrame, global_embedding: np.ndarray):
        """
        Plot GLOBAL continuum using ALL cell types' combined gamma for UMAP.
        This shows one cell type's state/proportion on the global embedding.
        
        Output: continuum_global_<cell_type>.png
        """
        safe_name = self._safe_filename(ct)
        
        props = self.prop_df[ct].values
        present_mask = props >= self.presence_threshold
        n_present = present_mask.sum()
        
        if n_present < 10:
            print(f"  Skipping global continuum plot for {ct}: only {n_present} spots")
            return
        
        # Use global embedding, filter to present spots
        embedding_filtered = global_embedding[present_mask]
        gamma_filtered = gamma_df.values[present_mask]
        props_filtered = props[present_mask]
        
        # Get primary state dimension
        if gamma_filtered.shape[1] > 1:
            pca = PCA(n_components=1, random_state=42)
            state_values = pca.fit_transform(gamma_filtered).flatten()
        else:
            state_values = gamma_filtered.flatten()
        
        # Calculate consistent point size
        point_size = max(10, min(50, 3000 / n_present))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: State variation
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            embedding_filtered[:, 0], embedding_filtered[:, 1],
            c=state_values, cmap='coolwarm', s=point_size, alpha=0.8, edgecolors='none'
        )
        ax1.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
        ax1.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
        ax1.set_title(f"{ct}\nCell State (γ) - Global UMAP", fontsize=14, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=12)
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label("Latent State", fontsize=12, fontweight='bold')
        cbar1.ax.tick_params(labelsize=10)
        
        # Panel 2: Proportion
        ax2 = axes[1]
        scatter2 = ax2.scatter(
            embedding_filtered[:, 0], embedding_filtered[:, 1],
            c=props_filtered, cmap='viridis', s=point_size, alpha=0.8, edgecolors='none'
        )
        ax2.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
        ax2.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
        ax2.set_title(f"{ct}\nProportion (β) - Global UMAP", fontsize=14, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=12)
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label("Proportion", fontsize=12, fontweight='bold')
        cbar2.ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.state_dir, f"continuum_global_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: continuum_global_{safe_name}.png (n={n_present}, global UMAP)")
    
    def plot_global_continuum_full(self, ct: str, gamma_df: pd.DataFrame, global_embedding: np.ndarray):
        """
        Plot GLOBAL continuum showing ALL spots on the global UMAP.
        Present spots colored by state/proportion, absent spots are grey.
        
        Output: continuum_full_global_<cell_type>.png
        """
        safe_name = self._safe_filename(ct)
        
        props = self.prop_df[ct].values
        present_mask = props >= self.presence_threshold
        n_present = present_mask.sum()
        n_absent = (~present_mask).sum()
        n_total = len(props)
        
        if n_present < 10:
            print(f"  Skipping full global continuum plot for {ct}: only {n_present} spots")
            return
        
        # Get gamma and state values for present spots
        gamma_filtered = gamma_df.values[present_mask]
        props_filtered = props[present_mask]
        
        if gamma_filtered.shape[1] > 1:
            pca = PCA(n_components=1, random_state=42)
            state_values_present = pca.fit_transform(gamma_filtered).flatten()
        else:
            state_values_present = gamma_filtered.flatten()
        
        # Calculate consistent point size
        point_size = max(10, min(50, 3000 / n_total))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: State variation (all spots)
        ax1 = axes[0]
        
        # Plot absent spots first (grey background)
        if n_absent > 0:
            ax1.scatter(
                global_embedding[~present_mask, 0], global_embedding[~present_mask, 1],
                c='lightgrey', s=point_size, alpha=0.4, edgecolors='none'
            )
        
        # Plot present spots (colored by state)
        scatter1 = ax1.scatter(
            global_embedding[present_mask, 0], global_embedding[present_mask, 1],
            c=state_values_present, cmap='coolwarm', s=point_size, alpha=0.8, edgecolors='none'
        )
        ax1.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
        ax1.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
        ax1.set_title(f"{ct}\nCell State (γ) - Global UMAP (All Spots)", fontsize=14, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=12)
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label("Latent State", fontsize=12, fontweight='bold')
        cbar1.ax.tick_params(labelsize=10)
        
        # Panel 2: Proportion (all spots)
        ax2 = axes[1]
        
        # Plot absent spots first (grey background)
        if n_absent > 0:
            ax2.scatter(
                global_embedding[~present_mask, 0], global_embedding[~present_mask, 1],
                c='lightgrey', s=point_size, alpha=0.4, edgecolors='none'
            )
        
        # Plot present spots (colored by proportion)
        scatter2 = ax2.scatter(
            global_embedding[present_mask, 0], global_embedding[present_mask, 1],
            c=props_filtered, cmap='viridis', s=point_size, alpha=0.8, edgecolors='none'
        )
        ax2.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
        ax2.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
        ax2.set_title(f"{ct}\nProportion (β) - Global UMAP (All Spots)", fontsize=14, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=12)
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label("Proportion", fontsize=12, fontweight='bold')
        cbar2.ax.tick_params(labelsize=10)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                      markersize=8, label=f'Present (n={n_present})'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey',
                      markersize=8, alpha=0.5, label=f'Absent (n={n_absent})')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
                  fontsize=10, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(os.path.join(self.state_dir, f"continuum_full_global_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: continuum_full_global_{safe_name}.png (present={n_present}, absent={n_absent})")
    
    def plot_spatial_state_map(self, ct: str, gamma_df: pd.DataFrame):
        """Plot spatial map of cell states with clear legend."""
        if self.spatial_coords is None:
            return
        
        safe_name = self._safe_filename(ct)
        props = self.prop_df[ct].values
        present_mask = props >= self.presence_threshold
        
        n_present = present_mask.sum()
        n_absent = (~present_mask).sum()
        
        if n_present < 5:
            print(f"  Skipping spatial state map for {ct}: only {n_present} spots above threshold")
            return
        
        # Get state values
        gamma_values = gamma_df.values
        if gamma_values.shape[1] > 1:
            pca = PCA(n_components=1, random_state=42)
            state_values = pca.fit_transform(gamma_values).flatten()
        else:
            state_values = gamma_values.flatten()
        
        coords_for_limits = self.spatial_coords_full if self.spatial_coords_full is not None else self.spatial_coords
        hex_radius = self._calculate_hex_radius(coords_for_limits, scale_factor=0.55)
        
        x_min = coords_for_limits[:, 0].min() - hex_radius * 2
        x_max = coords_for_limits[:, 0].max() + hex_radius * 2
        y_min = coords_for_limits[:, 1].min() - hex_radius * 2
        y_max = coords_for_limits[:, 1].max() + hex_radius * 2
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Track legend handles
        legend_handles = []
        
        # Helper for background (spots without ANY count data)
        def add_background(ax):
            if self.spatial_coords_full is None or self.matched_mask is None:
                return 0
            unmatched_mask = ~self.matched_mask
            n_unmatched = unmatched_mask.sum()
            if n_unmatched == 0:
                return 0
            patches = [RegularPolygon((x, y), numVertices=6, radius=hex_radius, 
                                      orientation=self.hex_orientation_rad)
                    for x, y in self.spatial_coords_full[unmatched_mask]]
            ax.add_collection(PatchCollection(patches, facecolors='#E0E0E0', edgecolors='none', alpha=0.4))
            return n_unmatched
        
        # ==================== Panel 1: State map ====================
        ax1 = axes[0]
        n_no_data = add_background(ax1)
        
        # Absent spots (proportion below threshold) - use very light color, not white
        absent_mask = ~present_mask
        if absent_mask.any():
            patches = [RegularPolygon((self.spatial_coords[j, 0], self.spatial_coords[j, 1]),
                                    numVertices=6, radius=hex_radius, 
                                    orientation=self.hex_orientation_rad)
                    for j in np.where(absent_mask)[0]]
            # Use a distinct color for "below threshold" - light blue-grey
            ax1.add_collection(PatchCollection(patches, facecolors='#F5F5F5', edgecolors='#CCCCCC',
                                            linewidths=0.3, alpha=0.7))
        
        # Present spots (proportion >= threshold) - colored by state
        if present_mask.any():
            present_state = state_values[present_mask]
            vmin, vmax = present_state.min(), present_state.max()
            if vmax - vmin < 1e-8: 
                vmax = vmin + 1e-8
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.coolwarm
            
            patches = []
            colors = []
            for j in np.where(present_mask)[0]:
                patches.append(RegularPolygon((self.spatial_coords[j, 0], self.spatial_coords[j, 1]),
                                            numVertices=6, radius=hex_radius, 
                                            orientation=self.hex_orientation_rad))
                colors.append(cmap(norm(state_values[j])))
            
            ax1.add_collection(PatchCollection(patches, facecolors=colors, edgecolors='none'))
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04, shrink=0.7)
            cbar.set_label("Cell State (γ)", fontsize=10)
        
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        ax1.axis('off')
        ax1.set_title(f"{ct}\nSpatial Cell State", fontsize=14, fontweight='bold')
        
        # ==================== Panel 2: Proportion map ====================
        ax2 = axes[1]
        add_background(ax2)
        
        # Absent spots
        if absent_mask.any():
            patches = [RegularPolygon((self.spatial_coords[j, 0], self.spatial_coords[j, 1]),
                                    numVertices=6, radius=hex_radius, 
                                    orientation=self.hex_orientation_rad)
                    for j in np.where(absent_mask)[0]]
            ax2.add_collection(PatchCollection(patches, facecolors='#F5F5F5', edgecolors='#CCCCCC',
                                            linewidths=0.3, alpha=0.7))
        
        # Present spots - colored by proportion
        if present_mask.any():
            present_props = props[present_mask]
            vmin, vmax = 0, present_props.max()  # Start from 0 for proportion
            if vmax < 1e-8: 
                vmax = 1e-8
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.plasma
            
            patches = []
            colors = []
            for j in np.where(present_mask)[0]:
                patches.append(RegularPolygon((self.spatial_coords[j, 0], self.spatial_coords[j, 1]),
                                            numVertices=6, radius=hex_radius, 
                                            orientation=self.hex_orientation_rad))
                colors.append(cmap(norm(props[j])))
            
            ax2.add_collection(PatchCollection(patches, facecolors=colors, edgecolors='none'))
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04, shrink=0.7)
            cbar.set_label("Proportion (β)", fontsize=10)
        
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        ax2.axis('off')
        ax2.set_title(f"{ct}\nSpatial Proportion", fontsize=14, fontweight='bold')
        
        # ==================== Add unified legend below plots ====================
        # Create legend handles
        legend_elements = []
        
        # Present spots
        legend_elements.append(plt.Line2D([0], [0], marker='H', color='w', 
                                        markerfacecolor='steelblue', markersize=12,
                                        markeredgecolor='none',
                                        label=f'Present (β ≥ {self.presence_threshold:.0%}): n={n_present}'))
        
        # Absent spots (below threshold)
        if n_absent > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='H', color='w',
                                            markerfacecolor='#F5F5F5', markersize=12,
                                            markeredgecolor='#CCCCCC',
                                            label=f'Below threshold (β < {self.presence_threshold:.0%}): n={n_absent}'))
        
        # No data spots
        if n_no_data > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='H', color='w',
                                            markerfacecolor='#E0E0E0', markersize=12,
                                            markeredgecolor='none', alpha=0.4,
                                            label=f'No count data: n={n_no_data}'))
        
        # Add legend at bottom
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
                frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)  # Make room for legend
        plt.savefig(os.path.join(self.state_dir, f"spatial_state_{safe_name}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: spatial_state_{safe_name}.png (present={n_present}, below_threshold={n_absent}, no_data={n_no_data})")
    
    def plot_state_summary(self, gamma_dict: dict):
        """Create summary plot of state distributions."""
        print("\nGenerating state summary plot...")
        
        n_types = len(self.cell_types)
        cols = min(4, n_types)
        rows = math.ceil(n_types / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = [axes] if n_types == 1 else axes.flatten()
        
        for i, ct in enumerate(self.cell_types):
            ax = axes[i]
            
            gamma_df = gamma_dict[ct]
            props = self.prop_df[ct].values
            present_mask = props >= self.presence_threshold
            
            if present_mask.sum() < 5:
                ax.text(0.5, 0.5, f"{ct}\n(insufficient data)", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            gamma_values = gamma_df.values[present_mask]
            if gamma_values.shape[1] > 1:
                pca = PCA(n_components=1, random_state=42)
                state_values = pca.fit_transform(gamma_values).flatten()
            else:
                state_values = gamma_values.flatten()
            
            ax.hist(state_values, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
            ax.set_xlabel("Cell State (γ)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{ct}\n(n={present_mask.sum()})", fontsize=11, fontweight='bold')
            
            mean_state = np.mean(state_values)
            ax.axvline(mean_state, color='red', linestyle='--', linewidth=1.5, label=f'μ={mean_state:.2f}')
            ax.legend(fontsize=8)
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle("Cell State Distribution Summary", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.state_dir, "state_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: state_summary.png")
    
    def run_full_analysis(self, gamma_dict: dict):
        """Run complete cell state analysis."""
        print("\n" + "="*70)
        print("CELL STATE VISUALIZATION AND TRACKING")
        print("="*70)
        
        # 1. Save all outputs (filtered CSVs)
        self.save_all_outputs(gamma_dict)
        
        # 2. Compute global embedding (for global continuum plots)
        global_embedding = self.compute_global_umap_embedding(gamma_dict)
        
        # 3. Generate per-cell-type visualizations
        print("\nGenerating cell type-specific visualizations...")
        for ct in self.cell_types:
            print(f"\n  Processing: {ct}")
            gamma_df = gamma_dict[ct]
            
            # Cell-type-specific UMAP (primary continuum plot)
            self.plot_cell_type_continuum(ct, gamma_df)
            
            # Cell-type-specific UMAP with all spots (full view)
            self.plot_cell_type_continuum_full(ct, gamma_df)
            
            # Global UMAP plots
            self.plot_global_continuum(ct, gamma_df, global_embedding)
            self.plot_global_continuum_full(ct, gamma_df, global_embedding)
            
            # Spatial state map
            self.plot_spatial_state_map(ct, gamma_df)
        
        # 4. Summary plot
        self.plot_state_summary(gamma_dict)
        
        print("\n" + "="*70)
        print("CELL STATE ANALYSIS COMPLETE")
        print("="*70)
        print(f"Output directory: {self.state_dir}")
        print("\nOutput files per cell type:")
        print("  - gamma_states_*.csv: Filtered gamma values (spots with proportion >= threshold)")
        print("  - continuum_*.png: Cell-type-specific UMAP (filtered spots)")
        print("  - continuum_full_*.png: Cell-type-specific UMAP (all spots, absent=grey)")
        print("  - continuum_global_*.png: Global UMAP (filtered spots)")
        print("  - continuum_full_global_*.png: Global UMAP (all spots, absent=grey)")
        print("  - spatial_state_*.png: Spatial cell state map")


# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================

def load_coordinates(st_coords_path, st_barcodes):
    """Robust coordinate loading."""
    print(f"\nLoading ST coordinates: {st_coords_path}")
    
    with open(st_coords_path, 'r') as f:
        first_line = f.readline().strip().split(',')
    
    is_headerless = False
    if len(first_line) >= 2:
        try:
            float(first_line[1])
            is_headerless = True
        except ValueError:
            is_headerless = False

    if is_headerless:
        coords_df = pd.read_csv(st_coords_path, header=None, index_col=0)
        n_cols = coords_df.shape[1]
        if n_cols == 5:
            coords_df.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
        elif n_cols == 4:
            coords_df.columns = ['array_row', 'array_col', 'pxl_row', 'pxl_col']
        elif n_cols == 2:
            coords_df.columns = ['x', 'y']
    else:
        coords_df = pd.read_csv(st_coords_path, index_col=0)
    
    if 'pxl_col' in coords_df.columns and 'pxl_row' in coords_df.columns:
        coord_cols = ['pxl_col', 'pxl_row']
    elif 'x' in coords_df.columns and 'y' in coords_df.columns:
        coord_cols = ['x', 'y']
    elif 'array_col' in coords_df.columns and 'array_row' in coords_df.columns:
        coord_cols = ['array_col', 'array_row']
    else:
        numeric_cols = coords_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            coord_cols = list(numeric_cols[-2:])
        else:
            raise ValueError(f"Could not determine spatial columns from: {list(coords_df.columns)}")

    spatial_coords_full = coords_df[coord_cols].values.astype(np.float32)
    
    coords_index = coords_df.index.astype(str).str.strip().str.strip('"').str.strip("'")
    target_barcodes = np.array([str(x).strip().strip('"').strip("'") for x in st_barcodes])
    
    coords_dict = dict(zip(coords_index, spatial_coords_full))
    matched_coords = []
    matched_mask = []
    
    for barcode in target_barcodes:
        if barcode in coords_dict:
            matched_coords.append(coords_dict[barcode])
            matched_mask.append(True)
        else:
            matched_coords.append([0, 0])
            matched_mask.append(False)
            
    matched_coords = np.array(matched_coords)
    matched_mask = np.array(matched_mask)
    
    print(f"  Matched spots: {matched_mask.sum()} / {len(st_barcodes)}")
    
    if matched_mask.sum() == 0:
        raise ValueError("Coordinate matching failed completely.")
    
    st_barcodes_set = set(target_barcodes)
    viz_mask = np.array([b in st_barcodes_set for b in coords_index])
    
    return matched_coords, spatial_coords_full, matched_mask, viz_mask


def load_sc_data(counts_path, labels_path):
    print(f"Loading single-cell: {counts_path}")
    labels_df = pd.read_csv(labels_path, index_col=0)
    label_col = next((c for c in labels_df.columns if c.lower() in ['celltype', 'cell_type', 'annotation']), labels_df.columns[0])
    
    counts_df = pd.read_csv(counts_path, index_col=0)
    
    overlap_index = len(counts_df.index.intersection(labels_df.index))
    overlap_cols = len(counts_df.columns.intersection(labels_df.index))
    
    if overlap_cols > overlap_index:
        counts_df = counts_df.T
    
    common_cells = counts_df.index.intersection(labels_df.index)
    if len(common_cells) < 100:
        raise ValueError(f"Too few common cells ({len(common_cells)}).")
    
    counts_df = counts_df.loc[common_cells]
    labels_df = labels_df.loc[common_cells]
    
    adata = anndata.AnnData(X=counts_df.values.astype(np.float32))
    adata.obs_names = counts_df.index.astype(str)
    adata.var_names = counts_df.columns.astype(str)
    adata.obs['cell_type'] = labels_df[label_col].values.astype(str)
    adata.layers['counts'] = adata.X.copy()
    
    print(f"  Loaded SC: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


def load_st_data(counts_path):
    print(f"Loading spatial: {counts_path}")
    counts_df = pd.read_csv(counts_path, index_col=0)
    
    adata = anndata.AnnData(X=counts_df.values.astype(np.float32))
    adata.obs_names = counts_df.index.astype(str)
    adata.var_names = counts_df.columns.astype(str)
    adata.layers['counts'] = adata.X.copy()
    print(f"  Loaded ST: {adata.n_obs} spots x {adata.n_vars} genes")
    return adata


def preprocess_data(adata_sc, adata_st, n_hvg=2000, min_genes=0, min_cells=0):
    print(f"\nPreprocessing with target {n_hvg} HVGs...")
    
    if min_genes > 0:
        sc.pp.filter_cells(adata_sc, min_genes=min_genes)
    if min_cells > 0:
        sc.pp.filter_genes(adata_sc, min_cells=min_cells)
    
    adata_sc.X = adata_sc.layers['counts'].copy()
    
    try:
        sc.pp.highly_variable_genes(adata_sc, n_top_genes=n_hvg, flavor="seurat_v3", subset=False)
    except Exception:
        adata_temp = adata_sc.copy()
        sc.pp.normalize_total(adata_temp, target_sum=1e4)
        sc.pp.log1p(adata_temp)
        sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_hvg, flavor="seurat", subset=False)
        adata_sc.var['highly_variable'] = adata_temp.var['highly_variable']

    hvg_genes = adata_sc.var_names[adata_sc.var['highly_variable']]
    common_genes = hvg_genes.intersection(adata_st.var_names)
    
    if len(common_genes) < 50:
        common_genes = adata_sc.var_names.intersection(adata_st.var_names)
    
    if len(common_genes) == 0:
        raise ValueError("No common genes found!")
        
    print(f"  Common genes for training: {len(common_genes)}")
    
    adata_sc = adata_sc[:, common_genes].copy()
    adata_st = adata_st[:, common_genes].copy()
    
    return adata_sc, adata_st


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="DestVI Spatial Deconvolution with Cell State Tracking")
    parser.add_argument("--sc_counts", required=True, help="Path to single-cell counts CSV")
    parser.add_argument("--sc_labels", required=True, help="Path to single-cell labels CSV")
    parser.add_argument("--st_counts", required=True, help="Path to spatial counts CSV")
    parser.add_argument("--st_coords", default=None, help="Path to spatial coordinates CSV")
    parser.add_argument("--output_csv", required=True, help="Output path for proportions CSV")
    parser.add_argument("--output_plot", required=True, help="Output path for main heatmap plot")
    parser.add_argument("--output_corr_plot", default=None, help="Output path for correlation plot")
    parser.add_argument("--n_hvg", type=int, default=2000, help="Number of highly variable genes")
    parser.add_argument("--min_genes", type=int, default=0, help="Min genes per cell (QC)")
    parser.add_argument("--min_cells", type=int, default=0, help="Min cells per gene (QC)")
    parser.add_argument("--max_epochs_sc", type=int, default=300, help="Max epochs for SC model")
    parser.add_argument("--max_epochs_st", type=int, default=3000, help="Max epochs for ST model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", default='0', help="GPU ID or 'cpu'")
    parser.add_argument("--skip_cell_states", action='store_true', help="Skip cell state analysis")
    parser.add_argument("--presence_threshold", type=float, default=0.05, help="Threshold for cell type presence (default: 0.05 = 5%%)")
    parser.add_argument("--n_gamma_components", type=int, default=10, help="Number of gamma latent dimensions")
    parser.add_argument("--hex_orientation", type=float, default=0.0, 
                        help="Hexagon rotation angle in degrees (default: 0.0). "
                             "Use 30 for flat-top hexagons, 0 for pointy-top hexagons.")
    
    args = parser.parse_args()

    set_seeds(args.seed)
    
    if args.gpu == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.output_corr_plot is None:
        args.output_corr_plot = args.output_plot.replace('heatmap.png', 'correlation.png')
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    output_dir = os.path.dirname(args.output_plot)

    print("\n" + "="*70)
    print("DESTVI SPATIAL DECONVOLUTION PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Hexagon orientation: {args.hex_orientation}°")
    print(f"  - Presence threshold: {args.presence_threshold:.1%}")
    print(f"  - Random seed: {args.seed}")

    # 1. Load Data
    adata_sc = load_sc_data(args.sc_counts, args.sc_labels)
    adata_st = load_st_data(args.st_counts)

    # 2. Load Coordinates
    spatial_coords = None
    if args.st_coords:
        try:
            spatial_coords, spatial_coords_full, matched_mask, viz_mask = load_coordinates(
                args.st_coords, list(adata_st.obs_names)
            )
            
            if matched_mask.sum() < len(matched_mask):
                print(f"  Filtering ST to {matched_mask.sum()} matched spots...")
                adata_st = adata_st[matched_mask].copy()
                spatial_coords = spatial_coords[matched_mask]
            
            adata_st.obsm['spatial'] = spatial_coords
            adata_st.uns['spatial_full'] = spatial_coords_full
            adata_st.uns['matched_mask'] = viz_mask
            
        except Exception as e:
            print(f"Warning: Failed to load coordinates: {e}")
            spatial_coords = None

    # 3. Preprocess
    adata_sc, adata_st = preprocess_data(
        adata_sc, adata_st, 
        n_hvg=args.n_hvg,
        min_genes=args.min_genes,
        min_cells=args.min_cells
    )

    # 4. Train CondSCVI
    print("\nTraining SC Model (CondSCVI)...")
    scvi.model.CondSCVI.setup_anndata(adata_sc, layer='counts', labels_key='cell_type')
    sc_model = scvi.model.CondSCVI(adata_sc, n_latent=20)
    sc_model.train(max_epochs=args.max_epochs_sc, batch_size=128)

    # 5. Train DestVI
    print("\nTraining ST Model (DestVI)...")
    scvi.model.DestVI.setup_anndata(adata_st, layer='counts')
    st_model = scvi.model.DestVI.from_rna_model(adata_st, sc_model)
    st_model.train(max_epochs=args.max_epochs_st, batch_size=128)

    # 6. Extract Results using the new extractor
    print("\n" + "="*70)
    print("EXTRACTING DESTVI OUTPUTS")
    print("="*70)
    
    extractor = DestVICellStateExtractor(st_model, adata_st)
    
    # Get proportions (β)
    print("\nExtracting cell type proportions (β)...")
    prop_df = extractor.get_proportions()
    prop_df.to_csv(args.output_csv)
    print(f"  Saved: {args.output_csv}")
    
    # Get gamma states
    print("\nExtracting cell-type-specific states (γ)...")
    gamma_dict = extractor.get_gamma_states(method='scale_pca', n_components=args.n_gamma_components)
    
    # Print extraction summary
    print("\n" + "-"*50)
    print("EXTRACTION SUMMARY:")
    print("-"*50)
    print(f"  Proportions shape: {prop_df.shape}")
    print(f"  Cell types: {len(extractor.cell_types)}")
    for ct in extractor.cell_types:
        gamma_shape = gamma_dict[ct].shape
        gamma_var = gamma_dict[ct].var().sum()
        print(f"    {ct}: gamma shape={gamma_shape}, total_variance={gamma_var:.4f}")

    # 7. Standard Visualizations
    print("\nGenerating standard visualizations...")
    
    if spatial_coords is not None:
        VisualizationUtils.plot_spatial_maps(
            prop_df, spatial_coords, output_dir, 
            coords_full=adata_st.uns.get('spatial_full'), 
            matched_mask=adata_st.uns.get('matched_mask'),
            hex_orientation=args.hex_orientation
        )
    
    VisualizationUtils.plot_cooccurrence(prop_df, os.path.join(output_dir, "cooccurrence_heatmap.png"))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(prop_df, cmap='viridis', yticklabels=False)
    plt.title("Cell Type Proportions per Spot")
    plt.savefig(args.output_plot, dpi=300)
    plt.close()

    # 8. Cell State Visualization
    if not args.skip_cell_states:
        try:
            print(f"\nInitializing cell state analysis:")
            print(f"  - presence_threshold={args.presence_threshold:.1%}")
            print(f"  - hex_orientation={args.hex_orientation}°")
            cell_state_viz = CellStateVisualization(
                extractor=extractor,
                prop_df=prop_df,
                adata_st=adata_st,
                output_dir=output_dir,
                presence_threshold=args.presence_threshold,
                hex_orientation=args.hex_orientation
            )
            cell_state_viz.run_full_analysis(gamma_dict)
        except Exception as e:
            print(f"\nWarning: Cell state analysis error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping cell state visualization (--skip_cell_states flag)")

    # Final summary
    print("\n" + "="*70)
    print("DESTVI PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📊 MAIN OUTPUTS:")
    print(f"  └─ Proportions (β): {args.output_csv}")
    print(f"  └─ Main heatmap: {args.output_plot}")
    print(f"  └─ Spatial maps: {os.path.join(output_dir, 'spatial_intensity_maps.png')}")
    print(f"  └─ Dominant type: {os.path.join(output_dir, 'spatial_dominant_type.png')}")
    print(f"  └─ Co-occurrence: {os.path.join(output_dir, 'cooccurrence_heatmap.png')}")
    
    if not args.skip_cell_states:
        state_dir = os.path.join(output_dir, "cell_states")
        print(f"\n🧬 CELL STATE OUTPUTS (in {state_dir}):")
        print(f"  └─ gamma_states_*.csv: Filtered gamma values (proportion >= {args.presence_threshold:.0%})")
        print(f"  └─ cell_state_summary.csv: Summary statistics")
        print(f"  └─ continuum_*.png: Cell-type-specific UMAP (filtered)")
        print(f"  └─ continuum_full_*.png: Cell-type-specific UMAP (all spots)")
        print(f"  └─ continuum_global_*.png: Global UMAP (filtered)")
        print(f"  └─ continuum_full_global_*.png: Global UMAP (all spots)")
        print(f"  └─ spatial_state_*.png: Spatial cell state maps")
        print(f"  └─ state_summary.png: Distribution summary")
    
    print(f"\n⚙️  VISUALIZATION SETTINGS:")
    print(f"  └─ Hexagon orientation: {args.hex_orientation}°")
    print(f"  └─ Presence threshold: {args.presence_threshold:.1%}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()