#!/usr/bin/env python3
"""
Starfysh Spatial Deconvolution Pipeline with Cell State Tracking
================================================================================
OUTPUTS:
1. Cell Type Proportions: Abundance of each cell type per spot
2. Cell-Type-Specific States: Archetypal representation of cell state variation

Key outputs:
- proportions.csv: Cell type proportions per spot
- cell_states/archetype_weights_<cell_type>.csv: Cell-type-specific archetype weights
- cell_states/spatial_state_<cell_type>.png: Spatial visualization of states
- cell_states/continuum_<cell_type>.png: Cell-type-specific UMAP (only that cell type's data)
- cell_states/continuum_global_<cell_type>.png: Global UMAP (all cell types combined)
================================================================================
"""

import argparse
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
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls

warnings.filterwarnings('ignore')

# Starfysh imports
try:
    import starfysh
    from starfysh import AA, utils, post_analysis
    STARFYSH_AVAILABLE = True
    print("Starfysh package loaded successfully!")
except ImportError:
    STARFYSH_AVAILABLE = False
    print("INFO: Starfysh package not installed. Using alternative NMF-based implementation.")

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
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# CELL STATE EXTRACTION - STARFYSH SPECIFIC
# ==============================================================================

class StarfyshCellStateExtractor:
    """
    Extract cell-type-specific latent states from Starfysh model.
    
    Starfysh models:
    - Cell type proportions: Abundance of each cell type in each spot
    - Archetype weights: Cell-type-specific archetypal representation capturing
                         continuous variation in cell state along defined axes
    
    The archetype weights represent WHERE along a cell type's expression program
    continuum each spot falls (e.g., activation state, differentiation stage).
    """
    
    def __init__(self, model, adata_st, inference_outputs, cell_types):
        """
        Initialize extractor.
        
        Args:
            model: Trained Starfysh model
            adata_st: Spatial AnnData object used for training
            inference_outputs: Dictionary containing Starfysh inference results
            cell_types: List of cell type names
        """
        self.model = model
        self.adata_st = adata_st
        self.inference_outputs = inference_outputs
        self.cell_types = cell_types
        self.n_spots = adata_st.n_obs
        self.spot_names = adata_st.obs_names.tolist()
        
    def get_proportions(self) -> pd.DataFrame:
        """
        Extract cell type proportions from Starfysh.
        
        Returns:
            DataFrame with shape (n_spots, n_cell_types)
        """
        # Starfysh stores proportions in inference outputs
        if 'prop' in self.inference_outputs:
            props = self.inference_outputs['prop']
        elif 'cell_type_proportions' in self.inference_outputs:
            props = self.inference_outputs['cell_type_proportions']
        elif hasattr(self.model, 'get_proportions'):
            props = self.model.get_proportions()
        else:
            # Try to extract from adata
            if 'starfysh_prop' in self.adata_st.obsm:
                props = self.adata_st.obsm['starfysh_prop']
            else:
                raise ValueError("Could not find proportions in Starfysh outputs")
        
        # Handle different return types
        if hasattr(props, 'values'):
            props = props.values
        elif hasattr(props, 'numpy'):
            props = props.numpy()
        elif hasattr(props, 'cpu'):
            props = props.cpu().numpy()
        props = np.array(props)
        
        # Ensure proportions sum to 1
        props = props / props.sum(axis=1, keepdims=True)
        
        return pd.DataFrame(
            props,
            index=self.spot_names,
            columns=self.cell_types
        )
    
    def get_archetype_weights(self, n_components=10) -> dict:
        """
        Extract cell-type-specific archetype weights from Starfysh.
        
        Starfysh uses archetypal analysis to capture cell state variation.
        Each cell type has archetype weights that represent the continuous
        state variation.
        
        Args:
            n_components: Number of latent dimensions to extract
            
        Returns:
            Dictionary mapping cell_type -> DataFrame of archetype weights
        """
        archetype_dict = {}
        
        for ct in self.cell_types:
            print(f"    Extracting archetype weights for: {ct}")
            archetype_df = self._extract_archetype_for_celltype(ct, n_components)
            archetype_dict[ct] = archetype_df
            
        return archetype_dict
    
    def _extract_archetype_for_celltype(self, cell_type: str, n_components: int) -> pd.DataFrame:
        """
        Extract archetype weights for a specific cell type.
        
        Starfysh models cell states through archetype analysis - we extract
        the weights that represent where each spot falls along the archetype axes.
        """
        # Check if archetype weights are directly available
        if 'archetype_weights' in self.inference_outputs:
            if isinstance(self.inference_outputs['archetype_weights'], dict):
                if cell_type in self.inference_outputs['archetype_weights']:
                    weights = self.inference_outputs['archetype_weights'][cell_type]
                    return self._process_weights(weights, n_components)
        
        # Try to get cell-type-specific expression and derive states
        if 'qc_m' in self.inference_outputs:
            # qc_m contains cell-type-specific latent representations
            qc_m = self.inference_outputs['qc_m']
            ct_idx = self.cell_types.index(cell_type)
            
            if isinstance(qc_m, dict) and cell_type in qc_m:
                latent = qc_m[cell_type]
            elif len(qc_m.shape) == 3:
                latent = qc_m[:, ct_idx, :]
            else:
                latent = qc_m
            
            return self._process_weights(latent, n_components)
        
        # Try to get from model's latent space
        if hasattr(self.model, 'get_latent_representation'):
            try:
                latent = self.model.get_latent_representation(cell_type=cell_type)
                return self._process_weights(latent, n_components)
            except:
                pass
        
        # Fallback: use cell-type-specific expression variation
        return self._extract_via_expression(cell_type, n_components)
    
    def _process_weights(self, weights, n_components: int) -> pd.DataFrame:
        """Process raw weights into a standardized DataFrame."""
        if hasattr(weights, 'numpy'):
            weights = weights.numpy()
        elif hasattr(weights, 'cpu'):
            weights = weights.cpu().numpy()
        weights = np.array(weights, dtype=np.float32)
        
        # Handle non-finite values
        weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
        
        # Standardize
        scaler = StandardScaler()
        weights_scaled = scaler.fit_transform(weights)
        weights_scaled = np.nan_to_num(weights_scaled, nan=0, posinf=0, neginf=0)
        
        # Reduce dimensions if needed
        if weights_scaled.shape[1] > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            weights_scaled = pca.fit_transform(weights_scaled)
        
        col_names = [f"archetype_{i}" for i in range(weights_scaled.shape[1])]
        return pd.DataFrame(weights_scaled, index=self.spot_names, columns=col_names)
    
    def _extract_via_expression(self, cell_type: str, n_components: int) -> pd.DataFrame:
        """
        Extract state via cell-type-specific expression variation.
        
        This is a fallback method that uses the reconstructed expression
        for this cell type to infer state variation.
        """
        # Get cell-type-specific reconstructed expression
        if 'reconstructed' in self.inference_outputs:
            recon = self.inference_outputs['reconstructed']
            ct_idx = self.cell_types.index(cell_type)
            
            if isinstance(recon, dict) and cell_type in recon:
                expr = recon[cell_type]
            elif len(recon.shape) == 3:
                expr = recon[:, ct_idx, :]
            else:
                expr = recon
        else:
            # Use original expression weighted by proportion
            props = self.get_proportions()[cell_type].values
            expr = self.adata_st.X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            expr = expr * props[:, np.newaxis]
        
        if hasattr(expr, 'numpy'):
            expr = expr.numpy()
        elif hasattr(expr, 'cpu'):
            expr = expr.cpu().numpy()
        expr = np.array(expr, dtype=np.float32)
        
        # Log-transform for stability
        expr_log = np.log1p(expr * 1e4)
        expr_log = np.nan_to_num(expr_log, nan=0, posinf=0, neginf=0)
        
        # Standardize
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_log)
        expr_scaled = np.nan_to_num(expr_scaled, nan=0, posinf=0, neginf=0)
        
        # PCA to extract latent dimensions
        n_comp = min(n_components, expr_scaled.shape[1] - 1, expr_scaled.shape[0] - 1)
        n_comp = max(1, n_comp)
        
        pca = PCA(n_components=n_comp, random_state=42)
        archetype = pca.fit_transform(expr_scaled)
        
        col_names = [f"archetype_{i}" for i in range(archetype.shape[1])]
        return pd.DataFrame(archetype, index=self.spot_names, columns=col_names)
    
    def get_imputed_expression(self, cell_type: str) -> pd.DataFrame:
        """
        Get cell-type-specific imputed gene expression.
        
        This represents the expected expression profile for this cell type
        at each spot, given the inferred cell state.
        
        Args:
            cell_type: Name of cell type
            
        Returns:
            DataFrame with shape (n_spots, n_genes)
        """
        if 'reconstructed' in self.inference_outputs:
            recon = self.inference_outputs['reconstructed']
            ct_idx = self.cell_types.index(cell_type)
            
            if isinstance(recon, dict) and cell_type in recon:
                expr = recon[cell_type]
            elif len(recon.shape) == 3:
                expr = recon[:, ct_idx, :]
            else:
                expr = recon
            
            if hasattr(expr, 'numpy'):
                expr = expr.numpy()
            elif hasattr(expr, 'cpu'):
                expr = expr.cpu().numpy()
            expr = np.array(expr)
            
            return pd.DataFrame(
                expr,
                index=self.spot_names,
                columns=self.adata_st.var_names
            )
        else:
            raise ValueError("Reconstructed expression not available in inference outputs")


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
        cell_types = df_props.columns.tolist()
        n_types = len(cell_types)
        
        # CRITICAL: Sort cell types alphabetically for consistent color mapping
        cell_types_sorted = sorted(cell_types)
        cell_type_to_color_idx = {ct: i for i, ct in enumerate(cell_types_sorted)}
        
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

        # Dominant Cell Type Map - USE CONSISTENT COLOR MAPPING
        print("  Generating dominant cell type map...")
        dominant_idx = np.argmax(df_props.values, axis=1)
        dominant_cell_types = [cell_types[idx] for idx in dominant_idx]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use sorted cell types for colormap - ensures consistency across methods
        n_sorted = len(cell_types_sorted)
        cmap_cat = plt.colormaps.get_cmap('tab20').resampled(n_sorted)
        
        add_background_spots(ax, coords_full, matched_mask, hex_radius, orientation_rad)
        
        patches = []
        colors = []
        for j, (x, y) in enumerate(coords):
            hexagon = RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=orientation_rad)
            patches.append(hexagon)
            # Use the sorted index for color assignment
            ct = dominant_cell_types[j]
            color_idx = cell_type_to_color_idx[ct]
            colors.append(cmap_cat(color_idx / max(n_sorted - 1, 1)))
        
        collection = PatchCollection(patches, facecolors=colors, edgecolors='none')
        ax.add_collection(collection)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        
        # Legend with sorted cell types for consistency
        handles = [plt.Line2D([0], [0], marker='H', color='w', 
                              markerfacecolor=cmap_cat(cell_type_to_color_idx[ct] / max(n_sorted - 1, 1)),
                              label=ct, markersize=12, markeredgecolor='none') 
                   for ct in cell_types_sorted]
        
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
    Comprehensive cell state visualization for Starfysh results.
    
    Generates TWO types of UMAP plots:
    1. Cell-type-specific UMAP (continuum_*.png): 
       - UMAP computed using ONLY that cell type's archetype data
       - Shows structure specific to that cell type's state variation
       
    2. Global UMAP (continuum_global_*.png):
       - UMAP computed using ALL cell types' archetype data combined
       - Shows each cell type's presence in a shared embedding space
    """
    
    def __init__(self, extractor: StarfyshCellStateExtractor, prop_df: pd.DataFrame, 
                 adata_st, output_dir: str, presence_threshold: float = 0.05,
                 hex_orientation: float = 0.0):
        """
        Initialize cell state visualization.
        
        Args:
            extractor: StarfyshCellStateExtractor instance
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
        
        self.presence_threshold = presence_threshold
        self.hex_orientation = hex_orientation
        self.hex_orientation_rad = np.radians(hex_orientation)
        
        # Consistent plot parameters
        self.point_size = 30
        self.title_fontsize = 14
        self.label_fontsize = 12
        self.tick_fontsize = 10
        self.cbar_label_fontsize = 10
        
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
    
    def save_all_outputs(self, archetype_dict: dict):
        """
        Save all cell state outputs to files.
        """
        print("\n" + "="*70)
        print("SAVING CELL STATE OUTPUTS")
        print("="*70)
        
        # 1. Save archetype weights for each cell type (FILTERED - only present spots)
        print("\nSaving archetype weight CSVs (filtered to present spots)...")
        for ct, archetype_df in archetype_dict.items():
            safe_name = self._safe_filename(ct)
            
            # Filter to only spots where this cell type is present
            props = self.prop_df[ct].values
            present_mask = props >= self.presence_threshold
            n_present = present_mask.sum()
            
            if n_present > 0:
                # Get filtered archetype weights (only barcode index and archetype columns)
                archetype_filtered = archetype_df.loc[present_mask].copy()
                
                output_path = os.path.join(self.state_dir, f"archetype_weights_{safe_name}.csv")
                archetype_filtered.to_csv(output_path)
                print(f"  Saved: archetype_weights_{safe_name}.csv ({n_present} spots with proportion >= {self.presence_threshold:.1%})")
            else:
                print(f"  Skipped: archetype_weights_{safe_name}.csv (no spots above threshold)")
        
        # 2. Save summary statistics
        print("\nSaving summary statistics...")
        summary_data = []
        for ct, archetype_df in archetype_dict.items():
            props = self.prop_df[ct].values
            present_mask = props >= self.presence_threshold
            
            summary_data.append({
                'cell_type': ct,
                'n_spots_present': present_mask.sum(),
                'n_spots_total': len(props),
                'presence_threshold': self.presence_threshold,
                'mean_proportion': props[present_mask].mean() if present_mask.any() else 0,
                'max_proportion': props[present_mask].max() if present_mask.any() else 0,
                'archetype_variance': archetype_df.loc[present_mask].var().sum() if present_mask.any() else 0,
                'n_archetype_dims': archetype_df.shape[1]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.state_dir, "cell_state_summary.csv"), index=False)
        print(f"  Saved: cell_state_summary.csv")
    
    def compute_global_umap_embedding(self, archetype_dict: dict) -> np.ndarray:
        """
        Compute GLOBAL UMAP embedding using ALL cell types combined.
        
        This creates a shared embedding space where all spots are projected
        based on their combined archetype representation across all cell types.
        
        Returns:
            np.ndarray: 2D UMAP coordinates for all spots
        """
        print("\n  Computing GLOBAL UMAP embedding (all cell types combined)...")
        
        try:
            import umap
            use_umap = True
        except ImportError:
            print("    UMAP not available, using PCA")
            use_umap = False
        
        # Combine all cell types' archetypes
        all_archetype = [archetype_dict[ct].values for ct in self.cell_types]
        combined = np.hstack(all_archetype)
        
        print(f"    Combined archetype matrix shape: {combined.shape}")
        
        # Standardize
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)
        combined_scaled = np.nan_to_num(combined_scaled, nan=0, posinf=0, neginf=0)
        
        if use_umap:
            n_neighbors = min(15, len(combined_scaled) - 1)
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
            embedding = reducer.fit_transform(combined_scaled)
            print(f"    UMAP embedding computed: {embedding.shape}")
        else:
            pca = PCA(n_components=2, random_state=42)
            embedding = pca.fit_transform(combined_scaled)
            print(f"    PCA embedding computed: {embedding.shape}")
        
        return embedding
    
    def compute_celltype_specific_umap(self, archetype_df: pd.DataFrame, ct: str) -> np.ndarray:
        """
        Compute CELL-TYPE-SPECIFIC UMAP using ONLY that cell type's archetype data.
        
        This creates an embedding that captures the structure specific to
        this cell type's state variation, independent of other cell types.
        
        Args:
            archetype_df: DataFrame of archetype weights for this cell type
            ct: Cell type name (for logging)
            
        Returns:
            np.ndarray: 2D UMAP coordinates for all spots
        """
        print(f"    Computing cell-type-specific UMAP for {ct}...")
        
        try:
            import umap
            use_umap = True
        except ImportError:
            use_umap = False
        
        # Use only this cell type's archetype data
        archetype_values = archetype_df.values
        
        print(f"      Archetype matrix shape: {archetype_values.shape}")
        
        # Standardize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(archetype_values)
        scaled = np.nan_to_num(scaled, nan=0, posinf=0, neginf=0)
        
        if use_umap:
            n_neighbors = min(15, len(scaled) - 1)
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
            embedding = reducer.fit_transform(scaled)
        else:
            pca = PCA(n_components=2, random_state=42)
            embedding = pca.fit_transform(scaled)
        
        return embedding
    
    def plot_cell_type_continuum(self, ct: str, archetype_df: pd.DataFrame, 
                                  embedding: np.ndarray, plot_type: str = "specific"):
        """
        Plot cell state continuum for a cell type.
        
        Args:
            ct: Cell type name
            archetype_df: Archetype weights DataFrame
            embedding: UMAP embedding coordinates
            plot_type: "specific" for cell-type-specific UMAP, "global" for global UMAP
        """
        safe_name = self._safe_filename(ct)
        
        props = self.prop_df[ct].values
        present_mask = props >= self.presence_threshold
        n_present = present_mask.sum()
        n_total = len(props)
        
        if n_present < 10:
            print(f"    Skipping {plot_type} continuum plot for {ct}: only {n_present} spots")
            return
        
        # Filter to present spots
        embedding_filtered = embedding[present_mask]
        archetype_filtered = archetype_df.values[present_mask]
        props_filtered = props[present_mask]
        
        # Get primary state dimension
        if archetype_filtered.shape[1] > 1:
            pca = PCA(n_components=1, random_state=42)
            state_values = pca.fit_transform(archetype_filtered).flatten()
        else:
            state_values = archetype_filtered.flatten()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine title suffix and filename
        if plot_type == "specific":
            title_suffix = "(Cell-Type-Specific UMAP)"
            filename = f"continuum_{safe_name}.png"
        else:
            title_suffix = "(Global UMAP)"
            filename = f"continuum_global_{safe_name}.png"
        
        # Panel 1: State variation
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            embedding_filtered[:, 0], embedding_filtered[:, 1],
            c=state_values, cmap='coolwarm', s=self.point_size, alpha=0.8, edgecolors='none'
        )
        ax1.set_xlabel("UMAP 1", fontsize=self.label_fontsize, fontweight='bold')
        ax1.set_ylabel("UMAP 2", fontsize=self.label_fontsize, fontweight='bold')
        ax1.set_title(f"{ct}\nCell State (Archetype)\n{title_suffix}", 
                     fontsize=self.title_fontsize, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=self.tick_fontsize)
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label("Archetype Score", fontsize=self.cbar_label_fontsize, fontweight='bold')
        cbar1.ax.tick_params(labelsize=self.tick_fontsize)
        
        # Panel 2: Proportion
        ax2 = axes[1]
        scatter2 = ax2.scatter(
            embedding_filtered[:, 0], embedding_filtered[:, 1],
            c=props_filtered, cmap='viridis', s=self.point_size, alpha=0.8, edgecolors='none'
        )
        ax2.set_xlabel("UMAP 1", fontsize=self.label_fontsize, fontweight='bold')
        ax2.set_ylabel("UMAP 2", fontsize=self.label_fontsize, fontweight='bold')
        ax2.set_title(f"{ct}\nProportion\n{title_suffix}", 
                     fontsize=self.title_fontsize, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=self.tick_fontsize)
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label("Proportion", fontsize=self.cbar_label_fontsize, fontweight='bold')
        cbar2.ax.tick_params(labelsize=self.tick_fontsize)
        
        # Add spot count annotation
        fig.text(0.5, 0.02, f"Showing {n_present} spots with proportion ≥ {self.presence_threshold:.0%} (total: {n_total})", 
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(os.path.join(self.state_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {filename}")
    
    def plot_cell_type_continuum_full(self, ct: str, archetype_df: pd.DataFrame,
                                       embedding: np.ndarray, plot_type: str = "specific"):
        """
        Plot cell state continuum showing ALL spots (present spots colored, absent in grey).
        
        Args:
            ct: Cell type name
            archetype_df: Archetype weights DataFrame
            embedding: UMAP embedding coordinates (for ALL spots)
            plot_type: "specific" for cell-type-specific UMAP, "global" for global UMAP
        """
        safe_name = self._safe_filename(ct)
        
        props = self.prop_df[ct].values
        present_mask = props >= self.presence_threshold
        absent_mask = ~present_mask
        n_present = present_mask.sum()
        n_absent = absent_mask.sum()
        n_total = len(props)
        
        if n_present < 5:
            print(f"    Skipping full continuum plot for {ct}: only {n_present} present spots")
            return
        
        # Get primary state dimension for ALL spots
        archetype_values = archetype_df.values
        if archetype_values.shape[1] > 1:
            pca = PCA(n_components=1, random_state=42)
            state_values = pca.fit_transform(archetype_values).flatten()
        else:
            state_values = archetype_values.flatten()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Determine title suffix and filename
        if plot_type == "specific":
            title_suffix = "(Cell-Type-Specific UMAP)"
            filename = f"continuum_full_{safe_name}.png"
        else:
            title_suffix = "(Global UMAP)"
            filename = f"continuum_full_global_{safe_name}.png"
        
        # Panel 1: State variation
        ax1 = axes[0]
        
        # Plot absent spots first (grey background)
        if absent_mask.any():
            ax1.scatter(
                embedding[absent_mask, 0], embedding[absent_mask, 1],
                c='lightgrey', s=self.point_size * 0.6, alpha=0.4, 
                edgecolors='none', label=f'Below threshold (n={n_absent})'
            )
        
        # Plot present spots (colored by state)
        if present_mask.any():
            scatter1 = ax1.scatter(
                embedding[present_mask, 0], embedding[present_mask, 1],
                c=state_values[present_mask], cmap='coolwarm', 
                s=self.point_size, alpha=0.8, edgecolors='none'
            )
            cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
            cbar1.set_label("Archetype Score", fontsize=self.cbar_label_fontsize, fontweight='bold')
            cbar1.ax.tick_params(labelsize=self.tick_fontsize)
        
        ax1.set_xlabel("UMAP 1", fontsize=self.label_fontsize, fontweight='bold')
        ax1.set_ylabel("UMAP 2", fontsize=self.label_fontsize, fontweight='bold')
        ax1.set_title(f"{ct}\nCell State (Archetype)\n{title_suffix}", 
                     fontsize=self.title_fontsize, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=self.tick_fontsize)
        
        # Panel 2: Proportion
        ax2 = axes[1]
        
        # Plot absent spots first (grey background)
        if absent_mask.any():
            ax2.scatter(
                embedding[absent_mask, 0], embedding[absent_mask, 1],
                c='lightgrey', s=self.point_size * 0.6, alpha=0.4, 
                edgecolors='none', label=f'Below threshold (n={n_absent})'
            )
        
        # Plot present spots (colored by proportion)
        if present_mask.any():
            scatter2 = ax2.scatter(
                embedding[present_mask, 0], embedding[present_mask, 1],
                c=props[present_mask], cmap='viridis', 
                s=self.point_size, alpha=0.8, edgecolors='none'
            )
            cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
            cbar2.set_label("Proportion", fontsize=self.cbar_label_fontsize, fontweight='bold')
            cbar2.ax.tick_params(labelsize=self.tick_fontsize)
        
        ax2.set_xlabel("UMAP 1", fontsize=self.label_fontsize, fontweight='bold')
        ax2.set_ylabel("UMAP 2", fontsize=self.label_fontsize, fontweight='bold')
        ax2.set_title(f"{ct}\nProportion\n{title_suffix}", 
                     fontsize=self.title_fontsize, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=self.tick_fontsize)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                      markersize=8, label=f'Present (≥{self.presence_threshold:.0%}): n={n_present}'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey',
                      markersize=8, alpha=0.5, label=f'Below threshold: n={n_absent}')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
                  fontsize=10, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(os.path.join(self.state_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {filename}")
    
    def plot_spatial_state_map(self, ct: str, archetype_df: pd.DataFrame):
        """Plot spatial map of cell states with clear legend."""
        if self.spatial_coords is None:
            return
        
        safe_name = self._safe_filename(ct)
        props = self.prop_df[ct].values
        present_mask = props >= self.presence_threshold
        
        n_present = present_mask.sum()
        n_absent = (~present_mask).sum()
        
        if n_present < 5:
            print(f"    Skipping spatial state map for {ct}: only {n_present} spots above threshold")
            return
        
        # Get state values
        archetype_values = archetype_df.values
        if archetype_values.shape[1] > 1:
            pca = PCA(n_components=1, random_state=42)
            state_values = pca.fit_transform(archetype_values).flatten()
        else:
            state_values = archetype_values.flatten()
        
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
        
        # Absent spots (proportion below threshold)
        absent_mask = ~present_mask
        if absent_mask.any():
            patches = [RegularPolygon((self.spatial_coords[j, 0], self.spatial_coords[j, 1]),
                                    numVertices=6, radius=hex_radius, 
                                    orientation=self.hex_orientation_rad)
                    for j in np.where(absent_mask)[0]]
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
            cbar.set_label("Cell State (Archetype)", fontsize=10)
        
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
            vmin, vmax = 0, present_props.max()
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
            cbar.set_label("Proportion", fontsize=10)
        
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        ax2.axis('off')
        ax2.set_title(f"{ct}\nSpatial Proportion", fontsize=14, fontweight='bold')
        
        # ==================== Add unified legend below plots ====================
        legend_elements = []
        
        legend_elements.append(plt.Line2D([0], [0], marker='H', color='w', 
                                        markerfacecolor='steelblue', markersize=12,
                                        markeredgecolor='none',
                                        label=f'Present (β ≥ {self.presence_threshold:.0%}): n={n_present}'))
        
        if n_absent > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='H', color='w',
                                            markerfacecolor='#F5F5F5', markersize=12,
                                            markeredgecolor='#CCCCCC',
                                            label=f'Below threshold (β < {self.presence_threshold:.0%}): n={n_absent}'))
        
        if n_no_data > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='H', color='w',
                                            markerfacecolor='#E0E0E0', markersize=12,
                                            markeredgecolor='none', alpha=0.4,
                                            label=f'No count data: n={n_no_data}'))
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
                frameon=True, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(os.path.join(self.state_dir, f"spatial_state_{safe_name}.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: spatial_state_{safe_name}.png (present={n_present}, below_threshold={n_absent}, no_data={n_no_data})")
    
    def plot_state_summary(self, archetype_dict: dict):
        """Create summary plot of state distributions."""
        print("\nGenerating state summary plot...")
        
        n_types = len(self.cell_types)
        cols = min(4, n_types)
        rows = math.ceil(n_types / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = [axes] if n_types == 1 else axes.flatten()
        
        for i, ct in enumerate(self.cell_types):
            ax = axes[i]
            
            archetype_df = archetype_dict[ct]
            props = self.prop_df[ct].values
            present_mask = props >= self.presence_threshold
            
            if present_mask.sum() < 5:
                ax.text(0.5, 0.5, f"{ct}\n(insufficient data)", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            archetype_values = archetype_df.values[present_mask]
            if archetype_values.shape[1] > 1:
                pca = PCA(n_components=1, random_state=42)
                state_values = pca.fit_transform(archetype_values).flatten()
            else:
                state_values = archetype_values.flatten()
            
            ax.hist(state_values, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
            ax.set_xlabel("Cell State (Archetype)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{ct}\n(n={present_mask.sum()})", fontsize=11, fontweight='bold')
            
            mean_state = np.mean(state_values)
            ax.axvline(mean_state, color='red', linestyle='--', linewidth=1.5, label=f'μ={mean_state:.2f}')
            ax.legend(fontsize=8)
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle("Cell State Distribution Summary (Starfysh Archetypes)", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.state_dir, "state_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: state_summary.png")
    
    def run_full_analysis(self, archetype_dict: dict):
        """
        Run complete cell state analysis with BOTH UMAP types.
        
        Generates:
        1. Cell-type-specific UMAP (continuum_*.png, continuum_full_*.png)
           - UMAP computed using ONLY that cell type's archetype data
           
        2. Global UMAP (continuum_global_*.png, continuum_full_global_*.png)
           - UMAP computed using ALL cell types' archetype data combined
           
        3. Spatial state maps (spatial_state_*.png)
        4. State distribution summary (state_summary.png)
        """
        print("\n" + "="*70)
        print("CELL STATE VISUALIZATION AND TRACKING (STARFYSH)")
        print("="*70)
        
        # 1. Save all outputs
        self.save_all_outputs(archetype_dict)
        
        # 2. Compute GLOBAL UMAP embedding (all cell types combined)
        print("\n" + "-"*50)
        print("COMPUTING GLOBAL UMAP (all cell types combined)")
        print("-"*50)
        embedding_global = self.compute_global_umap_embedding(archetype_dict)
        
        # 3. Generate per-cell-type visualizations
        print("\n" + "-"*50)
        print("GENERATING PER-CELL-TYPE VISUALIZATIONS")
        print("-"*50)
        
        for ct in self.cell_types:
            print(f"\n  Processing: {ct}")
            archetype_df = archetype_dict[ct]
            
            # 3a. Compute CELL-TYPE-SPECIFIC UMAP (only this cell type's data)
            embedding_ct_specific = self.compute_celltype_specific_umap(archetype_df, ct)
            
            # 3b. Plot cell-type-specific UMAP (filtered spots only)
            self.plot_cell_type_continuum(ct, archetype_df, embedding_ct_specific, plot_type="specific")
            
            # 3c. Plot cell-type-specific UMAP (all spots, present colored, absent grey)
            self.plot_cell_type_continuum_full(ct, archetype_df, embedding_ct_specific, plot_type="specific")
            
            # 3d. Plot global UMAP (filtered spots only)
            self.plot_cell_type_continuum(ct, archetype_df, embedding_global, plot_type="global")
            
            # 3e. Plot global UMAP (all spots, present colored, absent grey)
            self.plot_cell_type_continuum_full(ct, archetype_df, embedding_global, plot_type="global")
            
            # 3f. Plot spatial state map
            self.plot_spatial_state_map(ct, archetype_df)
        
        # 4. Summary plot
        self.plot_state_summary(archetype_dict)
        
        print("\n" + "="*70)
        print("CELL STATE ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nOutput directory: {self.state_dir}")
        print("\nGenerated files per cell type:")
        print("  - continuum_*.png          : Cell-type-specific UMAP (filtered spots)")
        print("  - continuum_full_*.png     : Cell-type-specific UMAP (all spots)")
        print("  - continuum_global_*.png   : Global UMAP (filtered spots)")
        print("  - continuum_full_global_*.png : Global UMAP (all spots)")
        print("  - spatial_state_*.png      : Spatial cell state maps")
        print("  - state_summary.png        : Distribution summary")


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
    min_cells_per_type = 2 # Change to 3 or 5 if desired
    cell_counts = adata.obs['cell_type'].value_counts()
    valid_types = cell_counts[cell_counts >= min_cells_per_type].index
    
    if len(valid_types) < len(cell_counts):
        dropped = set(cell_counts.index) - set(valid_types)
        print(f"  Dropping {len(dropped)} cell types with < {min_cells_per_type} cells: {list(dropped)}")
        adata = adata[adata.obs['cell_type'].isin(valid_types)].copy()
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


def prepare_gene_signatures(adata_sc=None, signature_file=None, 
                           method='marker_genes', n_markers=50,
                           cell_type_col='cell_type'):
    """
    Prepare gene signatures for Starfysh.
    
    Starfysh is designed to work with pre-defined gene signatures.
    These can come from:
    1. A provided signature file (CSV with genes as rows, cell types as columns)
    2. Derived from scRNA-seq reference using marker gene detection
    3. Database signatures (PanglaoDB, CellMarker, etc.)
    
    Args:
        adata_sc: Optional scRNA-seq AnnData (used if no signature_file)
        signature_file: Path to CSV file with gene signatures
                       Format: genes as index, cell types as columns
                       Values: 1/0 (binary) or expression weights
        method: 'marker_genes' or 'mean_expression' (if deriving from scRNA-seq)
        n_markers: Number of marker genes per cell type
        cell_type_col: Column name for cell type annotations
        
    Returns:
        DataFrame with gene signatures (genes x cell_types)
    """
    print("\n" + "="*50)
    print("PREPARING GENE SIGNATURES")
    print("="*50)
    
    # Option 1: Load pre-defined signatures from file
    if signature_file is not None and os.path.exists(signature_file):
        print(f"\nLoading signatures from file: {signature_file}")
        sig_df = pd.read_csv(signature_file, index_col=0)
        print(f"  Loaded: {sig_df.shape[0]} genes x {sig_df.shape[1]} cell types")
        print(f"  Cell types: {list(sig_df.columns)}")
        return sig_df
    
    # Option 2: Derive signatures from scRNA-seq reference
    if adata_sc is None:
        raise ValueError(
            "Either signature_file or adata_sc must be provided!\n"
            "Starfysh can work with:\n"
            "  1. Pre-defined gene signatures (--signature_file)\n"
            "  2. Signatures derived from scRNA-seq reference"
        )
    
    print("\nDeriving signatures from scRNA-seq reference...")
    
    cell_types = adata_sc.obs[cell_type_col].unique()
    print(f"  Cell types found: {list(cell_types)}")
    
    # Normalize for marker detection
    adata_norm = adata_sc.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    
    if method == 'marker_genes':
        print(f"\n  Finding top {n_markers} marker genes per cell type...")
        print("  Method: Wilcoxon rank-sum test")
        
        # Find marker genes using Wilcoxon rank-sum test
        sc.tl.rank_genes_groups(adata_norm, cell_type_col, method='wilcoxon', 
                                n_genes=n_markers, use_raw=False)
        
        # Build signature dictionary
        signature_dict = {}
        all_markers = set()
        
        for ct in cell_types:
            # Get top markers for this cell type
            markers = list(adata_norm.uns['rank_genes_groups']['names'][ct][:n_markers])
            scores = list(adata_norm.uns['rank_genes_groups']['scores'][ct][:n_markers])
            pvals = list(adata_norm.uns['rank_genes_groups']['pvals_adj'][ct][:n_markers])
            
            # Filter by significance
            significant_markers = [
                m for m, p in zip(markers, pvals) 
                if p < 0.05  # Only significantly differentially expressed
            ]
            
            if len(significant_markers) < 10:
                print(f"    Warning: {ct} has only {len(significant_markers)} significant markers")
                significant_markers = markers[:max(10, len(significant_markers))]
            
            signature_dict[ct] = significant_markers
            all_markers.update(significant_markers)
            
            print(f"    {ct}: {len(significant_markers)} markers "
                  f"(top: {significant_markers[:3]}...)")
        
        all_markers = sorted(list(all_markers))
        
        # Create binary signature matrix
        sig_matrix = pd.DataFrame(0.0, index=all_markers, columns=cell_types)
        for ct, markers in signature_dict.items():
            for gene in markers:
                if gene in sig_matrix.index:
                    sig_matrix.loc[gene, ct] = 1.0
        
        print(f"\n  Binary signature matrix: {sig_matrix.shape[0]} genes x {sig_matrix.shape[1]} cell types")
        
    elif method == 'mean_expression':
        print(f"\n  Computing mean expression signatures...")
        
        # Use mean expression as signature (continuous values)
        sig_matrix = pd.DataFrame(index=adata_norm.var_names, columns=cell_types, dtype=float)
        
        for ct in cell_types:
            mask = adata_norm.obs[cell_type_col] == ct
            mean_expr = np.array(adata_norm[mask].X.mean(axis=0)).flatten()
            sig_matrix[ct] = mean_expr
            print(f"    {ct}: mean expr range [{mean_expr.min():.2f}, {mean_expr.max():.2f}]")
        
        # Keep only genes with meaningful expression
        max_expr = sig_matrix.max(axis=1)
        sig_matrix = sig_matrix[max_expr > 0.1]
        
        print(f"\n  Expression signature matrix: {sig_matrix.shape[0]} genes x {sig_matrix.shape[1]} cell types")
    
    elif method == 'specificity_weighted':
        print(f"\n  Computing specificity-weighted signatures...")
        
        # More sophisticated: weight by specificity (how unique to cell type)
        sc.tl.rank_genes_groups(adata_norm, cell_type_col, method='wilcoxon', 
                                n_genes=n_markers * 2, use_raw=False)
        
        sig_matrix = pd.DataFrame(0.0, index=adata_norm.var_names, columns=cell_types)
        
        for ct in cell_types:
            markers = list(adata_norm.uns['rank_genes_groups']['names'][ct][:n_markers])
            scores = list(adata_norm.uns['rank_genes_groups']['scores'][ct][:n_markers])
            
            # Normalize scores to [0, 1]
            scores = np.array(scores)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
            for gene, score in zip(markers, scores):
                if gene in sig_matrix.index:
                    sig_matrix.loc[gene, ct] = score
        
        # Keep only genes that are markers for at least one type
        marker_genes = sig_matrix.index[sig_matrix.max(axis=1) > 0]
        sig_matrix = sig_matrix.loc[marker_genes]
        
        print(f"\n  Weighted signature matrix: {sig_matrix.shape[0]} genes x {sig_matrix.shape[1]} cell types")
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: marker_genes, mean_expression, specificity_weighted")
    
    return sig_matrix


def load_database_signatures(database='panglaodb', tissue=None, species='human'):
    """
    Load gene signatures from public databases.
    
    This is how Starfysh is often used in practice - with curated
    marker gene lists from databases rather than derived from scRNA-seq.
    
    Args:
        database: 'panglaodb', 'cellmarker', or path to custom file
        tissue: Filter by tissue (optional)
        species: 'human' or 'mouse'
        
    Returns:
        DataFrame with gene signatures
    """
    print(f"\nLoading signatures from {database}...")
    
    if database == 'panglaodb':
        # PanglaoDB marker genes
        # In practice, you would download from: https://panglaodb.se/
        url = "https://panglaodb.se/markers/PanglaoDB_markers_27_Mar_2020.tsv.gz"
        print(f"  Note: Download markers from {url}")
        print("  Then provide via --signature_file")
        raise NotImplementedError("Please download PanglaoDB markers and provide via --signature_file")
        
    elif database == 'cellmarker':
        # CellMarker database
        # http://bio-bigdata.hrbmu.edu.cn/CellMarker/
        print("  Note: Download markers from CellMarker database")
        print("  Then provide via --signature_file")
        raise NotImplementedError("Please download CellMarker data and provide via --signature_file")
    
    else:
        # Custom file
        if os.path.exists(database):
            return pd.read_csv(database, index_col=0)
        else:
            raise FileNotFoundError(f"Signature file not found: {database}")


# ==============================================================================
# STARFYSH ALTERNATIVE IMPLEMENTATION (NMF-based)
# ==============================================================================

def run_starfysh_alternative(adata_st, adata_sc, signature_matrix, cell_types,
                              n_archetypes=8, max_epochs=500, seed=42):
    """
    Alternative Starfysh-style implementation using archetypal analysis.
    
    This implements the core concepts of Starfysh:
    1. Archetypal analysis to find extreme points in expression space
    2. Probabilistic assignment of spots to archetypes
    3. Deconvolution based on archetype signatures
    """
    print("\n" + "="*70)
    print("RUNNING NMF-BASED DECONVOLUTION (Starfysh Alternative)")
    print("="*70)
    
    # Get expression matrix
    X = adata_st.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)
    
    # Get signature matrix
    S = signature_matrix.values.astype(np.float32)
    
    n_spots = X.shape[0]
    n_genes = X.shape[1]
    n_types = len(cell_types)
    
    print(f"\nData dimensions:")
    print(f"  Expression matrix: {X.shape} (spots x genes)")
    print(f"  Signature matrix: {S.shape} (genes x cell_types)")
    print(f"  Cell types: {n_types}")
    print(f"  Archetypes per type: {n_archetypes}")
    
    # Step 1: Initial deconvolution using NNLS
    print("\n  Step 1: Initial NNLS deconvolution...")
    proportions = np.zeros((n_spots, n_types))
    
    for i in range(n_spots):
        if i % 1000 == 0:
            print(f"    Processing spot {i}/{n_spots}...")
        try:
            coef, _ = nnls(S, X[i])
            proportions[i] = coef
        except:
            proportions[i] = np.ones(n_types) / n_types
    
    # Normalize to sum to 1
    row_sums = proportions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    proportions = proportions / row_sums
    
    print(f"    Initial proportions computed for {n_spots} spots")
    
    # Step 2: Archetypal analysis per cell type
    print("\n  Step 2: Archetypal analysis per cell type...")
    archetype_weights = {}
    
    for ct_idx, ct in enumerate(cell_types):
        print(f"    Processing {ct}...")
        
        # Get spots where this cell type is present
        ct_props = proportions[:, ct_idx]
        present_mask = ct_props > 0.01
        n_present = present_mask.sum()
        
        if n_present < n_archetypes:
            # Not enough spots, use simple representation
            n_comp = max(1, min(3, n_present))
            archetype_weights[ct] = np.zeros((n_spots, n_comp))
            print(f"      Only {n_present} spots with presence, using {n_comp} components")
            continue
        
        # Weight expression by proportion
        weighted_expr = X * ct_props[:, np.newaxis]
        
        # Run NMF to find archetypes
        n_comp = min(n_archetypes, n_present - 1, n_genes - 1)
        n_comp = max(1, n_comp)
        
        nmf = NMF(n_components=n_comp, random_state=seed, max_iter=500, init='nndsvda')
        try:
            W = nmf.fit_transform(weighted_expr + 1e-10)
            print(f"      NMF converged with {n_comp} components")
        except Exception as e:
            print(f"      NMF failed: {e}, using zeros")
            W = np.zeros((n_spots, n_comp))
        
        archetype_weights[ct] = W
    
    # Step 3: Refine proportions using archetype information
    print("\n  Step 3: Refining proportions (10 iterations)...")
    
    refined_props = proportions.copy()
    
    # Iterative refinement
    for iteration in range(10):
        # Reconstruct expression using current proportions
        reconstruction = np.zeros_like(X)
        
        for ct_idx, ct in enumerate(cell_types):
            ct_signature = S[:, ct_idx]
            ct_props = refined_props[:, ct_idx]
            reconstruction += np.outer(ct_props, ct_signature)
        
        # Update proportions based on reconstruction error
        for i in range(n_spots):
            residual = X[i] - reconstruction[i]
            
            # Adjust proportions slightly based on residual
            for ct_idx in range(n_types):
                correlation = np.corrcoef(residual, S[:, ct_idx])[0, 1]
                if np.isfinite(correlation):
                    refined_props[i, ct_idx] *= (1 + 0.1 * correlation)
        
        # Re-normalize
        row_sums = refined_props.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        refined_props = refined_props / row_sums
        
        if iteration % 3 == 0:
            print(f"    Iteration {iteration+1}/10 complete")
    
    # Ensure non-negative
    refined_props = np.maximum(refined_props, 0)
    row_sums = refined_props.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    refined_props = refined_props / row_sums
    
    print("\n  Refinement complete!")
    
    # Package outputs
    inference_outputs = {
        'prop': refined_props,
        'archetype_weights': archetype_weights,
        'cell_types': cell_types,
        'reconstruction': np.dot(refined_props, S.T)
    }
    
    # Create a simple model object
    class StarfyshModel:
        def __init__(self, proportions, archetype_weights, cell_types):
            self.proportions = proportions
            self.archetype_weights = archetype_weights
            self.cell_types = cell_types
            self.cell_type_mapping = {ct: i for i, ct in enumerate(cell_types)}
        
        def get_proportions(self):
            return self.proportions
        
        def get_archetype_weights(self, cell_type):
            return self.archetype_weights.get(cell_type, None)
    
    model = StarfyshModel(refined_props, archetype_weights, cell_types)
    
    print("\n" + "="*70)
    print("NMF-BASED DECONVOLUTION COMPLETE")
    print("="*70)
    
    return model, inference_outputs


def run_starfysh_deconvolution(adata_st, adata_sc, signature_matrix, 
                                n_archetypes=8, max_epochs=500, 
                                learning_rate=0.01, seed=42):
    """
    Run Starfysh spatial deconvolution.
    
    If Starfysh package is available, uses native API.
    Otherwise, falls back to NMF-based alternative implementation.
    
    Args:
        adata_st: Spatial AnnData
        adata_sc: Single-cell AnnData (for reference)
        signature_matrix: Gene signature matrix (genes x cell_types)
        n_archetypes: Number of archetypes per cell type
        max_epochs: Training epochs
        learning_rate: Learning rate
        seed: Random seed
        
    Returns:
        model: Trained model
        inference_outputs: Dictionary with inference results
        cell_types: List of cell type names
    """
    set_seeds(seed)
    
    # Get common genes between ST data and signatures
    common_genes = adata_st.var_names.intersection(signature_matrix.index)
    print(f"\nCommon genes between ST and signatures: {len(common_genes)}")
    
    if len(common_genes) < 50:
        raise ValueError(f"Too few common genes ({len(common_genes)})")
    
    # Subset to common genes
    adata_st_sub = adata_st[:, common_genes].copy()
    sig_sub = signature_matrix.loc[common_genes]
    
    # Normalize ST data
    adata_st_norm = adata_st_sub.copy()
    sc.pp.normalize_total(adata_st_norm, target_sum=1e4)
    sc.pp.log1p(adata_st_norm)
    
    cell_types = list(sig_sub.columns)
    n_cell_types = len(cell_types)
    
    print(f"\nDeconvolution configuration:")
    print(f"  - Cell types: {n_cell_types}")
    print(f"  - Archetypes per type: {n_archetypes}")
    print(f"  - Genes: {len(common_genes)}")
    print(f"  - Spots: {adata_st_sub.n_obs}")
    print(f"  - Max epochs: {max_epochs}")
    
    if STARFYSH_AVAILABLE:
        print("\n  Using native Starfysh package...")
        try:
            # Try using Starfysh API
            gene_sig = sig_sub.copy()
            
            model = starfysh.Starfysh(
                adata=adata_st_norm,
                gene_sig=gene_sig,
                n_archetypes=n_archetypes,
                random_state=seed
            )
            
            print("\nTraining Starfysh model...")
            model.train(
                max_epochs=max_epochs,
                lr=learning_rate,
                batch_size=min(128, adata_st_norm.n_obs)
            )
            
            print("\nRunning inference...")
            inference_outputs = model.inference()
            
            proportions = model.get_cell_type_proportions()
            inference_outputs['prop'] = proportions
            inference_outputs['cell_types'] = cell_types
            
            print("\nStarfysh training complete!")
            return model, inference_outputs, cell_types
            
        except Exception as e:
            print(f"\nStarfysh native API failed: {e}")
            print("Falling back to alternative implementation...")
    
    # Use alternative NMF-based implementation
    model, inference_outputs = run_starfysh_alternative(
        adata_st_norm, adata_sc, sig_sub, cell_types,
        n_archetypes=n_archetypes, max_epochs=max_epochs, seed=seed
    )
    
    return model, inference_outputs, cell_types


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Starfysh Spatial Deconvolution with Cell State Tracking")
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
    parser.add_argument("--n_archetypes", type=int, default=8, help="Number of archetypes per cell type")
    parser.add_argument("--n_markers", type=int, default=50, help="Number of marker genes per cell type")
    parser.add_argument("--max_epochs", type=int, default=500, help="Max training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", default='0', help="GPU ID or 'cpu'")
    parser.add_argument("--skip_cell_states", action='store_true', help="Skip cell state analysis")
    parser.add_argument("--presence_threshold", type=float, default=0.05, help="Threshold for cell type presence (default: 0.05 = 5%%)")
    parser.add_argument("--n_state_components", type=int, default=10, help="Number of state latent dimensions")
    parser.add_argument("--hex_orientation", type=float, default=0.0, 
                        help="Hexagon rotation angle in degrees (default: 0.0)")
    # Signature options
    parser.add_argument("--signature_file", default=None, 
                       help="Path to pre-defined gene signatures CSV (genes x cell_types). "
                            "If provided, scRNA-seq reference is not required.")
    parser.add_argument("--signature_method", default='marker_genes',
                       choices=['marker_genes', 'mean_expression', 'specificity_weighted'],
                       help="Method for deriving signatures from scRNA-seq (if no signature_file)")
    
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
    print("STARFYSH SPATIAL DECONVOLUTION PIPELINE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Starfysh available: {STARFYSH_AVAILABLE}")
    print(f"  - Hexagon orientation: {args.hex_orientation}°")
    print(f"  - Presence threshold: {args.presence_threshold:.1%}")
    print(f"  - Number of archetypes: {args.n_archetypes}")
    print(f"  - Number of markers: {args.n_markers}")
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

    # 4. Prepare gene signatures
    signature_matrix = prepare_gene_signatures(
        adata_sc, 
        method=args.signature_method, 
        n_markers=args.n_markers
    )

    # 5. Run Starfysh deconvolution
    model, inference_outputs, cell_types = run_starfysh_deconvolution(
        adata_st, adata_sc, signature_matrix,
        n_archetypes=args.n_archetypes,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed
    )

    # 6. Extract Results using the extractor
    print("\n" + "="*70)
    print("EXTRACTING OUTPUTS")
    print("="*70)
    
    extractor = StarfyshCellStateExtractor(model, adata_st, inference_outputs, cell_types)
    
    # Get proportions
    print("\nExtracting cell type proportions...")
    prop_df = extractor.get_proportions()
    prop_df.to_csv(args.output_csv)
    print(f"  Saved: {args.output_csv}")
    
    # Get archetype states
    print("\nExtracting cell-type-specific archetype weights...")
    archetype_dict = extractor.get_archetype_weights(n_components=args.n_state_components)
    
    # Print extraction summary
    print("\n" + "-"*50)
    print("EXTRACTION SUMMARY:")
    print("-"*50)
    print(f"  Proportions shape: {prop_df.shape}")
    print(f"  Cell types: {len(cell_types)}")
    for ct in cell_types:
        arch_shape = archetype_dict[ct].shape
        arch_var = archetype_dict[ct].var().sum()
        print(f"    {ct}: archetype shape={arch_shape}, total_variance={arch_var:.4f}")

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
            cell_state_viz.run_full_analysis(archetype_dict)
        except Exception as e:
            print(f"\nWarning: Cell state analysis error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping cell state visualization (--skip_cell_states flag)")

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📊 MAIN OUTPUTS:")
    print(f"  └─ Proportions: {args.output_csv}")
    print(f"  └─ Main heatmap: {args.output_plot}")
    print(f"  └─ Spatial maps: {os.path.join(output_dir, 'spatial_intensity_maps.png')}")
    print(f"  └─ Dominant type: {os.path.join(output_dir, 'spatial_dominant_type.png')}")
    print(f"  └─ Co-occurrence: {os.path.join(output_dir, 'cooccurrence_heatmap.png')}")
    
    if not args.skip_cell_states:
        state_dir = os.path.join(output_dir, "cell_states")
        print(f"\n🧬 CELL STATE OUTPUTS (in {state_dir}):")
        print(f"  └─ archetype_weights_*.csv: Cell-type-specific archetype weights (filtered to present spots)")
        print(f"  └─ cell_state_summary.csv: Summary statistics")
        print(f"\n📈 UMAP VISUALIZATIONS (per cell type):")
        print(f"  └─ continuum_*.png          : Cell-type-specific UMAP (filtered spots)")
        print(f"  └─ continuum_full_*.png     : Cell-type-specific UMAP (all spots)")
        print(f"  └─ continuum_global_*.png   : Global UMAP (filtered spots)")
        print(f"  └─ continuum_full_global_*.png : Global UMAP (all spots)")
        print(f"  └─ spatial_state_*.png      : Spatial cell state maps")
        print(f"  └─ state_summary.png        : Distribution summary")
    
    print(f"\n⚙️  VISUALIZATION SETTINGS:")
    print(f"  └─ Hexagon orientation: {args.hex_orientation}°")
    print(f"  └─ Presence threshold: {args.presence_threshold:.1%}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()