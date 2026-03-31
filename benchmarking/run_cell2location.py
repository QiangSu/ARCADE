#!/usr/bin/env python3
"""
================================================================================
Cell2location Spatial Deconvolution Pipeline
================================================================================

With spatial visualization outputs matching STVAE/RCTD scripts:
- spatial_intensity_maps.png
- spatial_dominant_type.png
- cooccurrence_heatmap.png
================================================================================
"""

import argparse
import sys
import os
import re
import warnings
import math
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib
matplotlib.use('Agg')  # Must be BEFORE importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree

# Cell2location imports
from cell2location.models import RegressionModel, Cell2location

# Set global settings
import scvi
scvi.settings.seed = 42
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
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'legend.fontsize': 16,
    'legend.title_fontsize': 16,
    'figure.titlesize': 20,
    'figure.titleweight': 'bold',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ==============================================================================
# VISUALIZATION UTILITIES
# ==============================================================================

class VisualizationUtils:
    """Visualization utilities for spatial deconvolution results."""
    
    @staticmethod
    def _calculate_hex_radius(coords: np.ndarray, scale_factor: float = 0.6) -> float:
        """
        Auto-calculate hexagon radius from spot spacing.
        Uses nearest-neighbor distance to estimate appropriate size.
        """
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
            annot_kws={
                'size': 10,
                'weight': 'bold',
            }
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
        hex_orientation: int = 0
    ):
        """
        Generates spatial maps with HEXAGONAL spot markers:
        1. A grid of plots, one per cell type, showing proportion intensity.
        2. A dominant cell type map.
        """
        import matplotlib.ticker as ticker
        
        cell_types = df_props.columns
        n_types = len(cell_types)
        
        # Determine coordinate range
        if coords_full is not None:
            coords_for_limits = coords_full
        else:
            coords_for_limits = coords
        
        hex_radius = VisualizationUtils._calculate_hex_radius(coords_for_limits, scale_factor=0.55)
        
        x_min = coords_for_limits[:, 0].min() - hex_radius * 2
        x_max = coords_for_limits[:, 0].max() + hex_radius * 2
        y_min = coords_for_limits[:, 1].min() - hex_radius * 2
        y_max = coords_for_limits[:, 1].max() + hex_radius * 2
        
        def add_background_spots(ax, coords_full, matched_mask, hex_radius, orientation_deg):
            """Add grey hexagons for unmatched spots as background."""
            if coords_full is None or matched_mask is None:
                return
            
            unmatched_mask = ~matched_mask
            if not unmatched_mask.any():
                return
            
            unmatched_coords = coords_full[unmatched_mask]
            
            patches = []
            for x, y in unmatched_coords:
                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=hex_radius,
                    orientation=np.radians(orientation_deg)
                )
                patches.append(hexagon)
            
            collection = PatchCollection(
                patches, 
                facecolors='lightgrey', 
                edgecolors='none',
                alpha=0.3
            )
            ax.add_collection(collection)
        
        # =====================================================================
        # 1. Grid Plot (Intensity per type) - HEXAGONAL
        # =====================================================================
        cols = 4
        rows = math.ceil(n_types / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        axes = axes.flatten() if n_types > 1 else [axes]
        
        for i, ct in enumerate(cell_types):
            ax = axes[i]
            values = df_props[ct].values
            
            add_background_spots(ax, coords_full, matched_mask, hex_radius, hex_orientation)
            
            vmin, vmax = values.min(), values.max()
            if vmax - vmin < 1e-8:
                vmax = vmin + 1e-8
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.plasma

            patches = []
            colors = []
            for j, (x, y) in enumerate(coords):
                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=hex_radius,
                    orientation=np.radians(hex_orientation)
                )
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
            
            cbar = plt.colorbar(
                sm, 
                ax=ax, 
                fraction=0.046, 
                pad=0.04,
                shrink=0.85,
                aspect=15
            )
            
            cbar.locator = ticker.MaxNLocator(nbins=5)
            cbar.formatter = ticker.FormatStrFormatter('%.2f')
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=10)
            
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spatial_intensity_maps.png"), dpi=300)
        plt.close()

        # =====================================================================
        # 2. Dominant Cell Type Map - HEXAGONAL
        # =====================================================================
        dominant_idx = np.argmax(df_props.values, axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.colormaps.get_cmap('tab20').resampled(n_types)
        
        add_background_spots(ax, coords_full, matched_mask, hex_radius, hex_orientation)
        
        patches = []
        colors = []
        for j, (x, y) in enumerate(coords):
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=np.radians(hex_orientation)
            )
            patches.append(hexagon)
            colors.append(cmap(dominant_idx[j] / max(n_types - 1, 1)))
        
        collection = PatchCollection(patches, facecolors=colors, edgecolors='none')
        ax.add_collection(collection)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        
        handles = [
            plt.Line2D([0], [0], marker='H', color='w', 
                      markerfacecolor=cmap(i / max(n_types - 1, 1)),
                      label=cell_types[i], markersize=12, markeredgecolor='none')
            for i in range(n_types)
        ]
        
        if coords_full is not None and matched_mask is not None and (~matched_mask).any():
            handles.append(
                plt.Line2D([0], [0], marker='H', color='w',
                          markerfacecolor='lightgrey', alpha=0.5,
                          label='No count data', markersize=12, markeredgecolor='none')
            )
        
        ax.legend(handles=handles, title="Cell Type", 
                 bbox_to_anchor=(1.02, 1), loc='upper left',
                 frameon=True, fontsize=10)
        
        plt.title("Dominant Cell Type per Spot", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spatial_dominant_type.png"), dpi=300, bbox_inches='tight')
        plt.close()


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Cell2location for Spatial Deconvolution")
    
    # Input Arguments
    parser.add_argument("--sc_counts", type=str, required=True,
                        help="Path to single-cell counts CSV (genes x cells or cells x genes)")
    parser.add_argument("--sc_labels", type=str, required=True,
                        help="Path to single-cell labels CSV")
    parser.add_argument("--st_counts", type=str, required=True,
                        help="Path to spatial counts CSV")
    parser.add_argument("--st_coords", type=str, default=None,
                        help="Path to spatial coordinates CSV (index=barcodes, columns=[x, y])")
    
    # Output Arguments
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save predicted proportions CSV")
    parser.add_argument("--output_plot", type=str, required=True,
                        help="Path to save heatmap plot")
    
    # Hyperparameters
    parser.add_argument("--n_hvg", type=int, default=12000,
                        help="Number of genes to use (Cell2location works best with many genes)")
    parser.add_argument("--max_epochs_sc", type=int, default=250,
                        help="Max epochs for reference signature training (default: 250)")
    parser.add_argument("--max_epochs_st", type=int, default=30000,
                        help="Max epochs for spatial mapping (default: 30000)")
    parser.add_argument("--cells_per_spot", type=int, default=30,
                        help="Expected number of cells per spatial spot (N_cells_per_location)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID (e.g., '0' or 'cuda' or 'cpu')")
    parser.add_argument("--hex_orientation", type=int, default=0,
                        help="Hexagon orientation angle in degrees (0 = flat-top, 30 = pointy-top)")
    
    return parser.parse_args()


# ==============================================================================
# COORDINATE LOADING
# ==============================================================================

def load_coordinates(st_coords_path, st_barcodes):
    """
    Load spatial coordinates and match to ST barcodes.
    
    Returns:
        spatial_coords: coordinates for matched spots only
        spatial_coords_full: all coordinates from file
        matched_mask: boolean mask indicating which coords matched
    """
    print(f"\nLoading ST coordinates: {st_coords_path}")
    
    # First, peek to detect if there's a header
    with open(st_coords_path, 'r') as f:
        first_line = f.readline().strip()
    
    first_fields = first_line.split(',')
    first_field = first_fields[0].strip()
    
    # Heuristic: detect if first line is a header
    looks_like_barcode = bool(re.match(r'^[ACGT]+-?\d*$', first_field, re.IGNORECASE))
    looks_like_spot_id = bool(re.match(r'^spot_\d+$', first_field, re.IGNORECASE))
    is_numeric = first_field.replace('.', '').replace('-', '').isdigit()
    
    has_header = not (looks_like_barcode or looks_like_spot_id or is_numeric)
    
    print(f"  First line: {first_line[:80]}...")
    print(f"  Detected header: {has_header}")
    
    if has_header:
        coords_df = pd.read_csv(st_coords_path, index_col=0)
    else:
        coords_df = pd.read_csv(st_coords_path, header=None)
        
        # Assign column names based on number of columns
        n_cols = coords_df.shape[1]
        if n_cols == 6:
            coords_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
        elif n_cols == 5:
            coords_df.columns = ['barcode', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
        elif n_cols == 4:
            coords_df.columns = ['barcode', 'col1', 'x', 'y']
        elif n_cols == 3:
            coords_df.columns = ['barcode', 'x', 'y']
        else:
            coords_df.columns = ['barcode'] + [f'col{i}' for i in range(1, n_cols)]
        
        coords_df = coords_df.set_index('barcode')
    
    print(f"  Coords shape: {coords_df.shape}")
    print(f"  Coords columns: {list(coords_df.columns)}")
    print(f"  First 3 coord barcodes: {list(coords_df.index[:3])}")
    
    # Determine which columns to use for spatial coordinates
    if 'pxl_col' in coords_df.columns and 'pxl_row' in coords_df.columns:
        coord_cols = ['pxl_col', 'pxl_row']
        print(f"  Using pixel coordinates (pxl_col, pxl_row)")
    elif 'x' in coords_df.columns and 'y' in coords_df.columns:
        coord_cols = ['x', 'y']
        print(f"  Using x, y coordinates")
    elif 'array_col' in coords_df.columns and 'array_row' in coords_df.columns:
        coord_cols = ['array_col', 'array_row']
        print(f"  Using array coordinates (array_col, array_row)")
    else:
        numeric_cols = coords_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            coord_cols = list(numeric_cols[-2:])
            print(f"  Using last 2 numeric columns: {coord_cols}")
        else:
            raise ValueError(f"Could not determine spatial columns from: {list(coords_df.columns)}")
    
    # Get full coordinates
    spatial_coords_full = coords_df[coord_cols].values.astype(np.float32)
    
    # Match to ST barcodes
    st_barcodes_set = set(st_barcodes)
    coords_barcodes = list(coords_df.index)
    
    matched_mask = np.array([b in st_barcodes_set for b in coords_barcodes])
    print(f"  Matched spots: {matched_mask.sum()} / {len(coords_barcodes)}")
    
    if matched_mask.sum() == 0:
        print(f"  WARNING: No barcode matches!")
        print(f"    ST barcodes examples: {list(st_barcodes)[:3]}")
        print(f"    Coord barcodes examples: {list(coords_barcodes)[:3]}")
        raise ValueError("No matching barcodes between ST data and coordinates!")
    
    # Get coordinates for matched spots, maintaining order of st_barcodes
    coords_dict = dict(zip(coords_barcodes, spatial_coords_full))
    spatial_coords = np.array([coords_dict[b] for b in st_barcodes if b in coords_dict])
    
    print(f"  Spatial coords range: X=[{spatial_coords[:,0].min():.1f}, {spatial_coords[:,0].max():.1f}], "
          f"Y=[{spatial_coords[:,1].min():.1f}, {spatial_coords[:,1].max():.1f}]")
    
    return spatial_coords, spatial_coords_full, matched_mask


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(sc_path, sc_labels_path, st_path, st_coords_path=None):
    """
    Load and format SC and ST data with flexible format handling.
    
    Returns:
        adata_sc: AnnData for single-cell
        adata_st: AnnData for spatial (with spatial coords in obsm if provided)
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # =========================================================================
    # LOAD SC COUNTS
    # =========================================================================
    print(f"Loading SC counts: {sc_path}")
    sc_df = pd.read_csv(sc_path, index_col=0)
    print(f"  SC shape: {sc_df.shape}")
    print(f"  First 3 barcodes: {list(sc_df.index[:3])}")
    print(f"  First 3 genes: {list(sc_df.columns[:3])}")
    
    # =========================================================================
    # LOAD SC LABELS
    # =========================================================================
    print(f"\nLoading SC labels: {sc_labels_path}")
    lbl_df = pd.read_csv(sc_labels_path, index_col=0)
    print(f"  Labels shape: {lbl_df.shape}")
    print(f"  First 3 label barcodes: {list(lbl_df.index[:3])}")
    
    # Find label column
    label_col = None
    for col_name in ['cell_type', 'celltype', 'CellType', 'cluster', 'Celltype', 'cell_types', 'label']:
        if col_name in lbl_df.columns:
            label_col = col_name
            break
    if label_col is None:
        label_col = lbl_df.columns[0]
    print(f"  Using label column: '{label_col}'")
    
    # Align SC with labels
    common_cells = sc_df.index.intersection(lbl_df.index)
    print(f"  Common cells between counts and labels: {len(common_cells)}")
    
    if len(common_cells) == 0:
        print(f"  ERROR: No overlap!")
        print(f"    SC index examples: {list(sc_df.index[:5])}")
        print(f"    Label index examples: {list(lbl_df.index[:5])}")
        raise ValueError("No common cell barcodes between sc_counts and sc_labels!")
    
    adata_sc = anndata.AnnData(sc_df.loc[common_cells].astype(np.float32))
    adata_sc.obs['cell_type'] = lbl_df.loc[common_cells, label_col].values
    
    print(f"  Final SC AnnData: {adata_sc.shape}")
    print(f"  Cell types ({adata_sc.obs['cell_type'].nunique()}): {list(adata_sc.obs['cell_type'].unique()[:5])}...")
    
    # =========================================================================
    # LOAD ST COUNTS (handles both 'barcode' column and index formats)
    # =========================================================================
    print(f"\nLoading ST counts: {st_path}")
    
    # Peek at first row to detect format
    st_df_peek = pd.read_csv(st_path, nrows=1)
    first_col = st_df_peek.columns[0].lower().strip()
    
    if first_col == 'barcode':
        print(f"  Detected 'barcode' as explicit column")
        st_df = pd.read_csv(st_path)
        st_df = st_df.set_index('barcode')
    elif first_col == '' or first_col.startswith('unnamed'):
        print(f"  Detected standard index format")
        st_df = pd.read_csv(st_path, index_col=0)
    else:
        print(f"  First column is '{st_df_peek.columns[0]}', trying index_col=0")
        st_df = pd.read_csv(st_path, index_col=0)
    
    print(f"  ST shape: {st_df.shape}")
    print(f"  First 3 spot IDs: {list(st_df.index[:3])}")
    print(f"  First 3 genes: {list(st_df.columns[:3])}")
    
    # =========================================================================
    # FIND COMMON GENES
    # =========================================================================
    sc_genes = set(adata_sc.var_names)
    st_genes = set(st_df.columns)
    common_genes = list(sc_genes.intersection(st_genes))
    common_genes.sort()
    
    print(f"\nGene overlap:")
    print(f"  SC genes: {len(sc_genes)}")
    print(f"  ST genes: {len(st_genes)}")
    print(f"  Common genes: {len(common_genes)}")
    
    if len(common_genes) < 100:
        raise ValueError(f"Too few common genes ({len(common_genes)}). Check gene name formats.")
    
    # Subset to common genes
    adata_sc = adata_sc[:, common_genes].copy()
    adata_st = anndata.AnnData(st_df[common_genes].astype(np.float32))
    
    # =========================================================================
    # LOAD ST COORDINATES (if provided)
    # =========================================================================
    if st_coords_path and os.path.exists(st_coords_path):
        try:
            spatial_coords, spatial_coords_full, matched_mask = load_coordinates(
                st_coords_path, list(adata_st.obs_names)
            )
            adata_st.obsm['spatial'] = spatial_coords
            # Store full coords info for visualization
            adata_st.uns['spatial_full'] = spatial_coords_full
            adata_st.uns['matched_mask'] = matched_mask
        except Exception as e:
            print(f"  WARNING: Could not load coordinates: {e}")
    else:
        print(f"\nNo coordinates file provided or found, skipping spatial info")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATA LOADING COMPLETE")
    print("=" * 60)
    print(f"  SC AnnData: {adata_sc.shape[0]} cells x {adata_sc.shape[1]} genes")
    print(f"  ST AnnData: {adata_st.shape[0]} spots x {adata_st.shape[1]} genes")
    print(f"  Cell types: {list(adata_sc.obs['cell_type'].unique())}")
    if 'spatial' in adata_st.obsm:
        print(f"  Spatial coordinates: Available")
    else:
        print(f"  Spatial coordinates: Not available")
    
    return adata_sc, adata_st


# ==============================================================================
# PREPROCESSING
# ==============================================================================

def preprocess_for_c2l(adata_sc, adata_st, n_genes=12000):
    """
    Cell2location requires raw integer counts.
    """
    print("\n" + "=" * 60)
    print(f"PREPROCESSING (Targeting {n_genes} genes)")
    print("=" * 60)
    
    # 1. Filter genes in SC
    sc.pp.filter_cells(adata_sc, min_genes=50)
    sc.pp.filter_genes(adata_sc, min_cells=10)
    
    # 2. Identify common genes first
    common_genes = adata_sc.var_names.intersection(adata_st.var_names)
    adata_sc = adata_sc[:, common_genes].copy()
    adata_st = adata_st[:, common_genes].copy()
    
    # 3. Filter mitochondrial genes
    adata_sc.var['mt'] = adata_sc.var_names.str.startswith('MT-') | adata_sc.var_names.str.startswith('mt-')
    adata_sc = adata_sc[:, ~adata_sc.var['mt']].copy()
    
    # Update ST to match
    common_genes = adata_sc.var_names.intersection(adata_st.var_names)
    adata_st = adata_st[:, common_genes].copy()
    
    # 4. Select top genes by total counts
    if adata_sc.n_vars > n_genes:
        print(f"Subsetting to top {n_genes} genes based on total counts...")
        sc.pp.calculate_qc_metrics(adata_sc, inplace=True)
        top_genes = adata_sc.var['total_counts'].sort_values(ascending=False).head(n_genes).index
        adata_sc = adata_sc[:, top_genes].copy()
        adata_st = adata_st[:, top_genes].copy()
        
    print(f"Final Data Shapes: SC {adata_sc.shape}, ST {adata_st.shape}")
    
    # Ensure Integer counts
    from scipy.sparse import issparse
    if issparse(adata_sc.X):
        adata_sc.X = np.array(adata_sc.X.todense())
    adata_sc.X = np.round(adata_sc.X).astype(int)
    
    if issparse(adata_st.X):
        adata_st.X = np.array(adata_st.X.todense())
    adata_st.X = np.round(adata_st.X).astype(int)
    
    # Print data statistics
    print(f"SC counts - min: {adata_sc.X.min()}, max: {adata_sc.X.max()}, mean: {adata_sc.X.mean():.2f}")
    print(f"ST counts - min: {adata_st.X.min()}, max: {adata_st.X.max()}, mean: {adata_st.X.mean():.2f}")
    print(f"Cell types: {adata_sc.obs['cell_type'].unique().tolist()}")
        
    return adata_sc, adata_st


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def train_reference_model(adata_sc, max_epochs=250, gpu="0"):
    """
    Step 1: Estimate reference signatures using RegressionModel.
    """
    print("\n" + "=" * 60)
    print("STEP 1: TRAINING REFERENCE SIGNATURES")
    print("=" * 60)
    
    # Determine accelerator
    accelerator_mode = "cpu" if gpu == "cpu" else "gpu"
    print(f"Training on: {accelerator_mode}")

    # Get unique cell types
    cell_types = adata_sc.obs['cell_type'].unique().tolist()
    print(f"Training on {len(cell_types)} cell types: {cell_types}")

    # Setup SC Anndata
    RegressionModel.setup_anndata(
        adata_sc, 
        layer=None,
        labels_key='cell_type'
    )
    
    mod = RegressionModel(adata_sc)
    
    # Train
    print(f"Training RegressionModel for {max_epochs} epochs...")
    mod.train(max_epochs=max_epochs, batch_size=2500, accelerator=accelerator_mode)
    
    # Export Posterior
    print("Exporting posterior samples...")
    adata_sc = mod.export_posterior(
        adata_sc, 
        sample_kwargs={'num_samples': 1000, 'batch_size': 2500}
    )
    
    # Debug output
    print(f"\nKeys in adata_sc.varm: {list(adata_sc.varm.keys())}")
    
    # Extract signatures from varm
    if 'means_per_cluster_mu_fg' in adata_sc.varm:
        sig_df = pd.DataFrame(
            adata_sc.varm['means_per_cluster_mu_fg'],
            index=adata_sc.var_names
        )
        
        print(f"Original column names: {sig_df.columns.tolist()}")
        
        # Clean column names
        clean_names = []
        for col in sig_df.columns:
            if isinstance(col, str) and col.startswith('means_per_cluster_mu_fg_'):
                clean_names.append(col.replace('means_per_cluster_mu_fg_', ''))
            else:
                clean_names.append(str(col))
        
        sig_df.columns = clean_names
        print(f"Cleaned column names: {sig_df.columns.tolist()}")
        
        print(f"\nSignature statistics:")
        print(f"  Min: {sig_df.values.min():.4f}")
        print(f"  Max: {sig_df.values.max():.4f}")
        print(f"  Mean: {sig_df.values.mean():.4f}")
        
        if sig_df.values.max() == 0:
            print("WARNING: All signatures are zero! Training may have failed.")
        
        return sig_df
    
    else:
        raise KeyError(f"Could not find 'means_per_cluster_mu_fg' in varm. Available: {list(adata_sc.varm.keys())}")


def clean_cell2location_columns(df, cell_types):
    """
    Clean Cell2location output column names by removing prefixes.
    """
    prefixes_to_remove = [
        'q05cell_abundance_w_sf_',
        'q05_cell_abundance_w_sf_',
        'means_cell_abundance_w_sf_',
        'meanscell_abundance_w_sf_',
        'stds_cell_abundance_w_sf_',
        'stdscell_abundance_w_sf_',
        'q95cell_abundance_w_sf_',
        'q95_cell_abundance_w_sf_',
    ]
    
    new_columns = []
    for col in df.columns:
        col_str = str(col)
        cleaned = col_str
        
        for prefix in prefixes_to_remove:
            if col_str.startswith(prefix):
                cleaned = col_str[len(prefix):]
                break
        
        if cleaned == col_str:
            for ct in cell_types:
                if ct in col_str:
                    cleaned = ct
                    break
        
        new_columns.append(cleaned)
    
    df.columns = new_columns
    return df


def train_spatial_mapping(adata_st, inf_aver_df, cells_per_spot=30, max_epochs=30000, gpu="0"):
    """
    Step 2: Map signatures to spatial spots using Cell2location.
    """
    print("\n" + "=" * 60)
    print("STEP 2: SPATIAL MAPPING (Cell2location)")
    print("=" * 60)
    
    # Determine accelerator
    accelerator_mode = "cpu" if gpu == "cpu" else "gpu"
    print(f"Training on: {accelerator_mode}")

    # Ensure ST data has matching genes
    common_genes = adata_st.var_names.intersection(inf_aver_df.index)
    print(f"Common genes between ST and reference: {len(common_genes)}")
    
    if len(common_genes) < 100:
        raise ValueError(f"Too few common genes ({len(common_genes)}). Check gene name format.")
    
    adata_st = adata_st[:, common_genes].copy()
    inf_aver_df = inf_aver_df.loc[common_genes, :]
    
    cell_types = inf_aver_df.columns.tolist()
    print(f"Cell types for mapping: {cell_types}")
    
    # Ensure integer counts
    from scipy.sparse import issparse
    if issparse(adata_st.X):
        adata_st.X = np.array(adata_st.X.todense())
    adata_st.X = np.round(adata_st.X).astype(int)
    
    print(f"\nSignature matrix shape: {inf_aver_df.shape}")
    print(f"Signature sum per cell type:\n{inf_aver_df.sum(axis=0)}")
    print(f"\nST data shape: {adata_st.shape}")
    print(f"ST total counts per spot: min={adata_st.X.sum(axis=1).min()}, max={adata_st.X.sum(axis=1).max()}")
    
    # Setup ST Anndata
    Cell2location.setup_anndata(adata_st, layer=None)
    
    detection_alpha = 200
    
    # Initialize Model
    mod = Cell2location(
        adata_st, 
        cell_state_df=inf_aver_df, 
        N_cells_per_location=cells_per_spot,
        detection_alpha=detection_alpha
    )
    
    print(f"\nModel parameters:")
    print(f"  N_cells_per_location: {cells_per_spot}")
    print(f"  detection_alpha: {detection_alpha}")
    print(f"  max_epochs: {max_epochs}")
    
    print(f"\nTraining Cell2location for {max_epochs} epochs...")
    mod.train(
        max_epochs=max_epochs,
        batch_size=None,
        train_size=1.0,
        accelerator=accelerator_mode
    )
    
    print("\nExtracting posterior...")
    adata_st = mod.export_posterior(
        adata_st, 
        sample_kwargs={'num_samples': 1000, 'batch_size': adata_st.n_obs}
    )
    
    print(f"\nKeys in adata_st.obsm: {list(adata_st.obsm.keys())}")
    
    # Extract abundance from obsm
    abundance_key = 'q05_cell_abundance_w_sf'
    if abundance_key not in adata_st.obsm:
        abundance_key = 'means_cell_abundance_w_sf'
    
    raw_abundance = adata_st.obsm[abundance_key]
    
    print(f"\nUsing key: {abundance_key}")
    print(f"Type of raw_abundance: {type(raw_abundance)}")
    
    if isinstance(raw_abundance, pd.DataFrame):
        print(f"Raw abundance is a DataFrame with columns: {raw_abundance.columns.tolist()}")
        abundance_df = raw_abundance.copy()
        abundance_df = clean_cell2location_columns(abundance_df, cell_types)
        print(f"Cleaned columns: {abundance_df.columns.tolist()}")
    else:
        print(f"Raw abundance shape: {raw_abundance.shape}")
        print(f"Raw abundance stats: min={raw_abundance.min():.4f}, max={raw_abundance.max():.4f}, mean={raw_abundance.mean():.4f}")
        
        abundance_df = pd.DataFrame(
            raw_abundance,
            index=adata_st.obs_names,
            columns=cell_types
        )
    
    print(f"\nAbundance DataFrame shape: {abundance_df.shape}")
    print(f"Abundance DataFrame dtypes: {abundance_df.dtypes.unique()}")
    print(f"Abundance stats: min={abundance_df.values.min():.4f}, max={abundance_df.values.max():.4f}")
    print(f"Abundance per cell type (sum across spots):\n{abundance_df.sum(axis=0)}")
    
    return abundance_df, adata_st


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    args = parse_arguments()
    
    # Create output directory
    output_dir = os.path.dirname(args.output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set PyTorch Device
    try:
        import torch
        if args.gpu == "cpu":
            print("Force using CPU.")
        elif torch.cuda.is_available():
            print(f"Using GPU: {args.gpu}")
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    except Exception as e:
        print(f"Note: Torch device setup warning: {e}")

    # 1. Load Data (now returns 2 values, coordinates handled internally)
    adata_sc, adata_st = load_data(
        args.sc_counts, 
        args.sc_labels, 
        args.st_counts,
        args.st_coords  # Pass coords path here
    )
    
    # 2. Preprocess
    adata_sc, adata_st = preprocess_for_c2l(adata_sc, adata_st, n_genes=args.n_hvg)
    
    # 3. Train Reference (SC)
    inf_aver_df = train_reference_model(adata_sc, max_epochs=args.max_epochs_sc, gpu=args.gpu)
    
    # 4. Train Spatial (ST)
    abundance_df, adata_st = train_spatial_mapping(
        adata_st, 
        inf_aver_df, 
        cells_per_spot=args.cells_per_spot,
        max_epochs=args.max_epochs_st,
        gpu=args.gpu
    )
    
    # 5. Convert Abundance to Proportions
    row_sums = abundance_df.sum(axis=1)
    print(f"\nRow sums (total abundance per spot): min={row_sums.min():.4f}, max={row_sums.max():.4f}")
    
    if (row_sums == 0).any():
        print(f"WARNING: {(row_sums == 0).sum()} spots have zero total abundance!")
        proportions_df = abundance_df.div(row_sums.replace(0, 1), axis=0)
        proportions_df.loc[row_sums == 0] = 1.0 / len(abundance_df.columns)
    else:
        proportions_df = abundance_df.div(row_sums, axis=0)
    
    proportions_df = proportions_df.fillna(0)
    
    prop_sums = proportions_df.sum(axis=1)
    print(f"Proportion row sums: min={prop_sums.min():.4f}, max={prop_sums.max():.4f}")
    
    # 6. Save Results
    print("\n" + "=" * 60)
    print(f"SAVING RESULTS to {output_dir}")
    print("=" * 60)
    
    print(f"Saving proportions to {args.output_csv}")
    proportions_df.to_csv(args.output_csv)
    
    abundance_path = args.output_csv.replace('.csv', '_absolute_abundance.csv')
    abundance_df.to_csv(abundance_path)
    print(f"Saved absolute abundances to {abundance_path}")
    
    # 7. Generate Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # 7a. Proportion Heatmap (original output)
    print(f"Generating heatmap to {args.output_plot}")

    n_spots = len(proportions_df)
    if n_spots > 100:
        print(f"  -> {n_spots} spots is too large for heatmap. Creating summary bar chart.")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        mean_props = proportions_df.mean().sort_values(ascending=False)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(mean_props)))
        
        bars = ax.bar(range(len(mean_props)), mean_props.values, color=colors)
        ax.set_xticks(range(len(mean_props)))
        ax.set_xticklabels(mean_props.index, rotation=45, ha='right')
        ax.set_ylabel("Mean Proportion")
        ax.set_xlabel("Cell Type")
        ax.set_title(f"Average Cell Type Proportions (n={n_spots} spots)")
        ax.set_ylim(0, min(1.0, mean_props.max() * 1.2))
        
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=150)
        plt.close()
    else:
        plt.figure(figsize=(12, max(8, n_spots * 0.3)))
        sns.heatmap(
            proportions_df, 
            cmap='viridis', 
            yticklabels=True,
            vmin=0, vmax=1
        )
        plt.title("Predicted Cell Type Proportions (Cell2location)")
        plt.xlabel("Cell Types")
        plt.ylabel("Spots")
        plt.tight_layout()
        plt.savefig(args.output_plot, dpi=150)
        plt.close()

    print("- Saved proportions visualization")
    
    # 7b. Co-occurrence Heatmap
    cooccurrence_path = os.path.join(output_dir, "cooccurrence_heatmap.png")
    VisualizationUtils.plot_cooccurrence(proportions_df, cooccurrence_path)
    print("- Saved cooccurrence_heatmap.png")
    
    # 7c. Spatial Maps (if coordinates available)
    spatial_coords = None
    spatial_coords_full = None
    matched_mask = None
    
    if 'spatial' in adata_st.obsm:
        spatial_coords = adata_st.obsm['spatial']
        if 'spatial_full' in adata_st.uns:
            spatial_coords_full = adata_st.uns['spatial_full']
        if 'matched_mask' in adata_st.uns:
            matched_mask = adata_st.uns['matched_mask']
    
    if spatial_coords is not None:
        # Ensure proportions_df index matches adata_st
        proportions_df_aligned = proportions_df.loc[adata_st.obs_names]
        
        VisualizationUtils.plot_spatial_maps(
            proportions_df_aligned,
            spatial_coords,
            output_dir,
            coords_full=spatial_coords_full,
            matched_mask=matched_mask,
            hex_orientation=args.hex_orientation
        )
        print("- Saved spatial_intensity_maps.png")
        print("- Saved spatial_dominant_type.png")
    else:
        print("- Skipping spatial maps (no coordinates provided)")
    
    # Print Summary Statistics
    print("\n" + "=" * 60)
    print("PROPORTION SUMMARY")
    print("=" * 60)
    print("\nAverage proportions per cell type:")
    print(proportions_df.mean().round(4))
    print("\nMax proportion per cell type:")
    print(proportions_df.max().round(4))
    
    print("\n" + "=" * 60)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Proportions saved to: {args.output_csv}")
    print(f"Abundances saved to: {abundance_path}")
    print(f"Heatmap saved to: {args.output_plot}")
    if spatial_coords is not None:
        print(f"Spatial maps saved to: {output_dir}/spatial_*.png")
    print(f"Co-occurrence heatmap saved to: {cooccurrence_path}")


if __name__ == "__main__":
    main()