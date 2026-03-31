#!/usr/bin/env python3
"""
================================================================================
Tangram Spatial Deconvolution Pipeline with Cell State Continuum Analysis
================================================================================

Enhanced version that outputs:
1. Cell type proportions (deconvolution)
2. Cell state continuum UMAPs for each cell type
3. Spatial mapping quality metrics
4. Spatial visualization maps:
   - spatial_intensity_maps.png
   - spatial_dominant_type.png
   - cooccurrence_heatmap.png
   - spatial_state_<cell_type>.png (NEW: Spatial in situ mapping of states)

Supports TWO input formats:
1. Real Visium data: --st_coords points to separate coordinate file
2. Simulated data: Coordinates embedded in counts CSV (row/col columns or spot names)
   - When --st_coords is omitted or set to "auto", coordinates are extracted from data

Key Addition: Cell state analysis via mapped cell embeddings per spot

Cell Type Order: Alphabetical ordering in all image outputs
================================================================================
"""

import argparse
import sys
import os
import re
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tangram as tg
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.ticker as ticker

# Optional imports for dimensionality reduction
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. Will use t-SNE instead.")

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ==============================================================================
# GLOBAL SETTINGS & PLOTTING CONFIGURATION
# ==============================================================================

sc.settings.verbosity = 0
sc.settings.seed = 42
warnings.filterwarnings('ignore')

# MATCHING REFERENCE STYLING
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

# Image rotation: Set to 0 to keep image upright  
IMAGE_ROTATION = 0  # Controls whole coordinate system rotation

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Tangram for Spatial Deconvolution with Cell State Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real Visium data (with coordinate file):
  python run_tangram.py --sc_counts sc.csv --sc_labels labels.csv \\
      --st_counts st.csv --st_coords tissue_positions.csv \\
      --output_csv results/proportions.csv --output_plot results/heatmap.png

  # Simulated data (coordinates in counts file):
  python run_tangram.py --sc_counts sc.csv --sc_labels labels.csv \\
      --st_counts simulated_spots.csv \\
      --output_csv results/proportions.csv --output_plot results/heatmap.png

  # Simulated data (explicit auto mode):
  python run_tangram.py --sc_counts sc.csv --sc_labels labels.csv \\
      --st_counts simulated_spots.csv --st_coords auto \\
      --output_csv results/proportions.csv --output_plot results/heatmap.png
        """
    )
    
    # Input Arguments
    parser.add_argument("--sc_counts", type=str, required=True,
                        help="Path to single-cell counts CSV")
    parser.add_argument("--sc_labels", type=str, required=True,
                        help="Path to single-cell labels CSV")
    parser.add_argument("--st_counts", type=str, required=True,
                        help="Path to spatial counts CSV")
    parser.add_argument("--st_coords", type=str, required=False, default=None,
                        help="Path to spatial coordinates CSV. Use 'auto' or omit for simulated data "
                             "with embedded coordinates (row/col columns or spot_row_col names)")
    
    # Output Arguments
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save predicted proportions CSV")
    parser.add_argument("--output_plot", type=str, required=True,
                        help="Path to save heatmap plot")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for all outputs (default: derived from output_csv)")
    
    # Tangram Specific Hyperparameters
    parser.add_argument("--n_markers", type=int, default=150,
                        help="Number of marker genes per cell type (default: 150)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs (default: 1000)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID (e.g., '0' or 'cpu')")
    
    # Cell filtering parameters
    parser.add_argument("--min_cells_per_type", type=int, default=10,
                        help="Minimum cells per cell type for marker detection (default: 10)")
    
    # Cell State Analysis Parameters
    parser.add_argument("--n_pcs", type=int, default=30,
                        help="Number of PCs for cell state embedding (default: 30)")
    parser.add_argument("--min_proportion", type=float, default=0.05,
                        help="Minimum proportion threshold for cell state analysis (default: 0.05)")
    parser.add_argument("--umap_neighbors", type=int, default=30,
                        help="Number of neighbors for UMAP (default: 30)")
    parser.add_argument("--skip_state_analysis", action="store_true",
                        help="Skip cell state continuum analysis")
    
    # Data format arguments
    parser.add_argument("--transpose_sc", action="store_true",
                        help="Transpose SC counts matrix")
    parser.add_argument("--transpose_st", action="store_true",
                        help="Transpose ST counts matrix")
    parser.add_argument("--hex_orientation", type=int, default=0,
                        help="Hexagon orientation angle in degrees (0 = flat-top, 30 = pointy-top)")
    
    # Legacy arguments (kept for compatibility)
    parser.add_argument("--cells_per_spot", type=int, default=None, 
                        help="Ignored in Tangram")
    parser.add_argument("--max_epochs_st", type=int, default=None, 
                        help="Ignored (uses --epochs)")
    
    return parser.parse_args()


# ==============================================================================
# COORDINATE EXTRACTION FUNCTIONS (For Simulated Data)
# ==============================================================================

def extract_coordinates_from_index(spot_names):
    """
    Extract coordinates from spot names like 'spot_0_1' or '0x1' or 'spot_r0_c1'.
    
    Returns:
        DataFrame with columns ['barcode', 'x', 'y'] or None if extraction fails
    """
    print("  Attempting to extract coordinates from spot names...")
    
    # Common patterns for spot names with embedded coordinates
    patterns = [
        r'spot_(\d+)_(\d+)',      # spot_0_1
        r'(\d+)x(\d+)',           # 0x1
        r'(\d+)_(\d+)',           # 0_1
        r'spot_r(\d+)_c(\d+)',    # spot_r0_c1
        r'r(\d+)_c(\d+)',         # r0_c1
        r'row(\d+)_col(\d+)',     # row0_col1
    ]
    
    coords = []
    matched_pattern = None
    
    for pattern in patterns:
        coords = []
        for name in spot_names:
            match = re.search(pattern, str(name))
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                coords.append((name, col, row))  # x=col, y=row for visualization
            else:
                break  # Pattern doesn't match all spots
        
        if len(coords) == len(spot_names):
            matched_pattern = pattern
            break
    
    if matched_pattern and len(coords) == len(spot_names):
        print(f"    Successfully extracted coordinates using pattern: {matched_pattern}")
        df = pd.DataFrame(coords, columns=['barcode', 'x', 'y'])
        print(f"    Coordinate range: x=[{df['x'].min()}, {df['x'].max()}], y=[{df['y'].min()}, {df['y'].max()}]")
        return df
    
    print("    Could not extract coordinates from spot names")
    return None


def extract_coordinates_from_columns(st_df):
    """
    Extract coordinates from row/col columns in the dataframe.
    
    Returns:
        DataFrame with columns ['barcode', 'x', 'y'] or None if extraction fails
    """
    print("  Attempting to extract coordinates from data columns...")
    
    # Look for row/col columns (case-insensitive)
    col_lower = {c.lower(): c for c in st_df.columns}
    
    row_col = None
    col_col = None
    
    # Check for row column
    for name in ['row', 'array_row', 'spot_row', 'y', 'y_coord']:
        if name in col_lower:
            row_col = col_lower[name]
            break
    
    # Check for col column
    for name in ['col', 'array_col', 'spot_col', 'column', 'x', 'x_coord']:
        if name in col_lower:
            col_col = col_lower[name]
            break
    
    if row_col is not None and col_col is not None:
        print(f"    Found coordinate columns: row='{row_col}', col='{col_col}'")
        df = pd.DataFrame({
            'barcode': st_df.index.astype(str),
            'x': st_df[col_col].values.astype(float),
            'y': st_df[row_col].values.astype(float)
        })
        print(f"    Coordinate range: x=[{df['x'].min()}, {df['x'].max()}], y=[{df['y'].min()}, {df['y'].max()}]")
        return df, [row_col, col_col]  # Return columns to drop
    
    print("    Could not find coordinate columns in data")
    return None, []


def load_coordinates(filepath, st_barcodes=None, st_df=None):
    """
    Load spatial coordinates from file OR extract from data.
    
    Supports:
    1. Separate coordinate file (filepath provided and exists)
    2. Coordinates embedded in spot names (filepath is None or 'auto')
    3. Coordinates as columns in data (filepath is None or 'auto')
    
    Args:
        filepath: Path to coordinates file, 'auto', or None
        st_barcodes: List of spot barcodes from count matrix
        st_df: Original ST dataframe (for extracting coordinate columns)
    
    Returns:
        coords_matched: DataFrame [barcode, x, y] aligned with st_barcodes
        coords_full: Numpy array [N, 2] of ALL spots (for background plotting)
    """
    print(f"\n{'='*70}")
    print("LOADING SPATIAL COORDINATES")
    print(f"{'='*70}")
    
    # Case 1: Auto-detect from data (no file provided or 'auto')
    if filepath is None or filepath.lower() == 'auto':
        print("  Mode: Auto-detect coordinates from data")
        
        # Try extracting from columns first
        if st_df is not None:
            coords_df, cols_to_drop = extract_coordinates_from_columns(st_df)
            if coords_df is not None:
                coords_full = coords_df[['x', 'y']].values
                return coords_df, coords_full, cols_to_drop
        
        # Try extracting from spot names
        if st_barcodes is not None:
            coords_df = extract_coordinates_from_index(st_barcodes)
            if coords_df is not None:
                coords_full = coords_df[['x', 'y']].values
                return coords_df, coords_full, []
        
        # Generate grid coordinates as fallback
        print("  Warning: Could not extract coordinates. Generating grid layout...")
        n_spots = len(st_barcodes) if st_barcodes is not None else 100
        grid_size = int(np.ceil(np.sqrt(n_spots)))
        coords = []
        for i, barcode in enumerate(st_barcodes if st_barcodes is not None else range(n_spots)):
            row = i // grid_size
            col = i % grid_size
            coords.append((str(barcode), col, row))
        
        coords_df = pd.DataFrame(coords, columns=['barcode', 'x', 'y'])
        coords_full = coords_df[['x', 'y']].values
        return coords_df, coords_full, []
    
    # Case 2: Load from file
    print(f"  Mode: Load from file: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Coordinate file not found: {filepath}")
    
    coords_df = None
    
    # Try reading with header first
    try:
        coords_df = pd.read_csv(filepath)
        
        # Check if it looks like tissue_positions_list.csv (6 columns, no header)
        if coords_df.shape[1] == 6 and coords_df.columns[0] not in ['barcode', 'Barcode', 'spot_id']:
            # This is likely tissue_positions_list.csv read incorrectly
            coords_df = pd.read_csv(filepath, header=None)
            coords_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
            
            full_df = pd.DataFrame({
                'barcode': coords_df['barcode'].astype(str),
                'x': coords_df['pxl_col'].astype(float),
                'y': coords_df['pxl_row'].astype(float)
            })
        else:
            # Check for standard column names
            barcode_col = None
            x_col = None
            y_col = None
            
            for col in coords_df.columns:
                col_lower = col.lower()
                if col_lower in ['barcode', 'spot_id', 'cell_id', 'spot']:
                    barcode_col = col
                elif col_lower in ['x', 'pxl_col', 'imagecol', 'col', 'x_coord']:
                    x_col = col
                elif col_lower in ['y', 'pxl_row', 'imagerow', 'row', 'y_coord']:
                    y_col = col
            
            # If first column is unnamed, it might be the index/barcode
            if barcode_col is None and coords_df.columns[0] == 'Unnamed: 0':
                barcode_col = 'Unnamed: 0'
            
            # Use index as barcode if no barcode column found
            if barcode_col is None:
                coords_df['barcode'] = coords_df.index.astype(str)
                barcode_col = 'barcode'
            
            # Try to find coordinate columns if not found
            if x_col is None or y_col is None:
                numeric_cols = coords_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    if x_col is None:
                        x_col = numeric_cols[0]
                    if y_col is None:
                        y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            
            if x_col is None or y_col is None:
                raise ValueError(f"Could not identify coordinate columns. Found: {list(coords_df.columns)}")
            
            full_df = pd.DataFrame({
                'barcode': coords_df[barcode_col].astype(str),
                'x': coords_df[x_col].astype(float),
                'y': coords_df[y_col].astype(float)
            })

        # Store ALL coordinates for background plotting
        coords_full = full_df[['x', 'y']].values
        
        # If count matrix barcodes are provided, filter and align
        if st_barcodes is not None:
            # Handle potential suffix mismatches (e.g., "-1")
            coord_bcs = set(full_df['barcode'])
            count_bcs = set(st_barcodes)
            
            # Check if we need to align suffixes
            if len(coord_bcs) > 0 and len(count_bcs) > 0:
                example_coord = next(iter(coord_bcs))
                example_count = next(iter(count_bcs))
                
                if str(example_count).endswith("-1") and not str(example_coord).endswith("-1"):
                    print("    Adding '-1' suffix to coordinate barcodes to match counts")
                    full_df['barcode'] = full_df['barcode'] + "-1"
                elif str(example_coord).endswith("-1") and not str(example_count).endswith("-1"):
                    print("    Removing '-1' suffix from coordinate barcodes to match counts")
                    full_df['barcode'] = full_df['barcode'].str.replace("-1", "", regex=False)

            # Filter to matched spots
            matched_df = full_df[full_df['barcode'].isin(count_bcs)].copy()
            
            # Reindex to match the order of st_barcodes
            matched_df = matched_df.set_index('barcode').reindex(st_barcodes)
            
            # Handle missing coordinates
            if matched_df.isnull().any().any():
                n_miss = matched_df.isnull().any(axis=1).sum()
                print(f"    Warning: {n_miss} spots in count matrix have no coordinates")
                matched_df = matched_df.fillna(0)
            
            matched_df = matched_df.reset_index()
            matched_df = matched_df.rename(columns={'index': 'barcode'})
            
            print(f"    Coordinate file: {len(coords_full)} total spots")
            print(f"    Matched to data: {len(matched_df)} spots")
            
            return matched_df, coords_full, []
        
        return full_df, coords_full, []
        
    except Exception as e:
        print(f"  Error loading coordinates: {e}")
        raise


def rotate_coordinates(x, y, angle_degrees):
    """Rotate coordinates by given angle around centroid."""
    if angle_degrees == 0:
        return x, y
    
    angle_rad = np.radians(angle_degrees)
    cx, cy = np.mean(x), np.mean(y)
    
    x_centered = x - cx
    y_centered = y - cy
    
    x_rot = x_centered * np.cos(angle_rad) - y_centered * np.sin(angle_rad)
    y_rot = x_centered * np.sin(angle_rad) + y_centered * np.cos(angle_rad)
    
    return x_rot + cx, y_rot + cy


def _calculate_hex_radius(coords: np.ndarray, orientation: int = 30) -> float:
    """
    Calculates the appropriate hexagon radius for perfect tessellation (no gaps).
    
    For tessellating hexagons:
    - Pointy-top (orientation=30): horizontal spacing = sqrt(3) * radius
    - Flat-top (orientation=0): horizontal spacing = 1.5 * radius
    
    Args:
        coords: Nx2 array of spot coordinates
        orientation: Hexagon orientation in degrees (30=pointy-top, 0=flat-top)
    
    Returns:
        Radius value for hexagons that tessellate perfectly
    """
    if len(coords) < 2:
        return 10.0
    
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nn_distances = distances[:, 1]
    valid_dists = nn_distances[nn_distances > 1e-6]
    
    if len(valid_dists) == 0:
        return 10.0
    
    median_spacing = np.median(valid_dists)
    
    # Calculate radius based on hexagon orientation
    # For perfect tessellation, nearest neighbor distance = sqrt(3) * radius (for standard hex grid)
    if orientation == 30:  # Pointy-top
        # In a pointy-top hex grid, horizontal neighbors are sqrt(3) * radius apart
        radius = median_spacing / np.sqrt(3)
    else:  # Flat-top
        # In a flat-top hex grid, horizontal neighbors are 1.5 * radius apart (edge-to-edge)
        # But diagonal neighbors are sqrt(3) * radius apart
        radius = median_spacing / np.sqrt(3)
    
    # Small overlap factor to prevent anti-aliasing gaps in PNG output
    radius *= 1.000
    
    return radius


def _add_background_spots(ax, coords_full, hex_radius):
    """
    Adds grey background hexagons for ALL spots (Reference Style).
    """
    if coords_full is None or len(coords_full) == 0:
        return

    patches = []
    for (x, y) in coords_full:
        hexagon = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_radius,
            orientation=np.radians(30)
        )
        patches.append(hexagon)
    
    collection = PatchCollection(
        patches, 
        facecolors='lightgrey', 
        edgecolors='none', 
        alpha=0.3,
        zorder=0
    )
    ax.add_collection(collection)


def load_data(sc_path, sc_labels_path, st_path, transpose_sc=False, transpose_st=False):
    """
    Load and format SC and ST data.
    
    Returns:
        adata_sc: AnnData for single-cell
        adata_st: AnnData for spatial
        st_df_original: Original ST dataframe (for coordinate extraction)
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # --- LOAD SC ---
    print(f"Loading SC counts: {sc_path}")
    sc_df = pd.read_csv(sc_path, index_col=0)
    print(f"  SC counts raw shape: {sc_df.shape}")
    
    if transpose_sc:
        print("  Transposing SC matrix (user requested)...")
        sc_df = sc_df.T
    
    print(f"  SC counts final shape: {sc_df.shape} (rows=cells x columns=genes)")
    
    # --- LOAD LABELS ---
    print(f"\nLoading SC labels: {sc_labels_path}")
    lbl_df = pd.read_csv(sc_labels_path, index_col=0)
    print(f"  Labels shape: {lbl_df.shape}")
    
    # Find label column
    label_col = next(
        (c for c in ['cell_type', 'celltype', 'CellType', 'cluster', 'annotation'] 
         if c in lbl_df.columns), 
        lbl_df.columns[0]
    )
    print(f"  Using label column: '{label_col}'")

    # Convert indices to string for comparison
    sc_df.index = sc_df.index.astype(str)
    lbl_df.index = lbl_df.index.astype(str)
    
    print(f"  SC index examples: {list(sc_df.index[:3])}")
    print(f"  Label index examples: {list(lbl_df.index[:3])}")
    
    # Find common cells
    common = sc_df.index.intersection(lbl_df.index)
    print(f"  Common cells between counts and labels: {len(common)}")
    
    if len(common) == 0:
        # Check if indices might match by position (same length)
        if len(sc_df) == len(lbl_df):
            print("  WARNING: No matching indices, but same number of cells.")
            print("  Assuming counts and labels are in the same order...")
            # Use sc_df index for both
            lbl_df = lbl_df.copy()
            lbl_df.index = sc_df.index
            common = sc_df.index
        else:
            print(f"\n  ERROR: Index mismatch!")
            print(f"    SC has {len(sc_df)} cells")
            print(f"    Labels has {len(lbl_df)} cells")
            print(f"    SC index examples: {list(sc_df.index[:5])}")
            print(f"    Label index examples: {list(lbl_df.index[:5])}")
            raise ValueError("No common cells found and different lengths - cannot match!")
    
    if len(common) < len(sc_df):
        print(f"  Note: Only {len(common)}/{len(sc_df)} cells have matching labels")
    
    # Subset to common cells
    sc_df = sc_df.loc[common]
    lbl_df = lbl_df.loc[common]
    
    # Create AnnData
    adata_sc = anndata.AnnData(sc_df.astype(np.float32))
    adata_sc.obs['cell_type'] = lbl_df.loc[adata_sc.obs_names, label_col].values
    
    print(f"\n  SC AnnData shape: {adata_sc.shape} (cells x genes)")
    print(f"  Cell types: {adata_sc.obs['cell_type'].nunique()}")
    
    # Print cell type distribution (alphabetically sorted for display)
    ct_counts = adata_sc.obs['cell_type'].value_counts()
    print(f"  Cell type distribution:")
    for ct in sorted(ct_counts.index):
        print(f"    {ct}: {ct_counts[ct]}")
    
    # --- LOAD ST ---
    print(f"\nLoading ST counts: {st_path}")
    st_df = pd.read_csv(st_path, index_col=0)
    print(f"  ST counts raw shape: {st_df.shape}")
    
    # Store original for coordinate extraction
    st_df_original = st_df.copy()
    
    if transpose_st:
        print("  Transposing ST matrix (user requested)...")
        st_df = st_df.T
        st_df_original = st_df_original.T
    else:
        # Check gene overlap to determine correct orientation
        sc_genes = set(adata_sc.var_names)
        col_gene_overlap = len(set(str(c) for c in st_df.columns) & sc_genes)
        row_gene_overlap = len(set(str(r) for r in st_df.index) & sc_genes)
        
        print(f"  Gene overlap check: {col_gene_overlap} in columns, {row_gene_overlap} in rows")
        
        if row_gene_overlap > col_gene_overlap and row_gene_overlap > 100:
            print("  Auto-transposing ST matrix (genes detected in rows)")
            st_df = st_df.T
            st_df_original = st_df_original.T
    
    adata_st = anndata.AnnData(st_df.astype(np.float32))
    
    print(f"  ST AnnData shape: {adata_st.shape} (spots x genes)")
    
    # Check final gene overlap
    common_genes = set(adata_sc.var_names) & set(adata_st.var_names)
    print(f"\n  Common genes between SC and ST: {len(common_genes)}")
    
    if len(common_genes) < 100:
        print("  WARNING: Very few common genes!")
        print(f"    SC genes (first 5): {list(adata_sc.var_names[:5])}")
        print(f"    ST genes (first 5): {list(adata_st.var_names[:5])}")
    
    return adata_sc, adata_st, st_df_original


def preprocess_tangram(adata_sc, adata_st, n_markers=150, min_cells_per_type=10):
    """
    Tangram preprocessing: normalization and marker selection.
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING (Normalization & Marker Selection)")
    print("=" * 70)
    
    # Store raw counts for later
    adata_sc.layers['counts'] = adata_sc.X.copy()
    adata_st.layers['counts'] = adata_st.X.copy()
    
    # --- FILTER CELL TYPES WITH TOO FEW CELLS ---
    print(f"\nFiltering cell types with < {min_cells_per_type} cells...")
    cell_type_counts = adata_sc.obs['cell_type'].value_counts()
    
    print("  Cell type counts:")
    for ct in sorted(cell_type_counts.index):
        count = cell_type_counts[ct]
        status = "✓" if count >= min_cells_per_type else "✗ (excluded)"
        print(f"    {ct}: {count} cells {status}")
    
    valid_cell_types = cell_type_counts[cell_type_counts >= min_cells_per_type].index.tolist()
    excluded_cell_types = cell_type_counts[cell_type_counts < min_cells_per_type].index.tolist()
    
    if len(excluded_cell_types) > 0:
        print(f"\n  Excluding {len(excluded_cell_types)} cell types with < {min_cells_per_type} cells:")
        for ct in sorted(excluded_cell_types):
            print(f"    - {ct} ({cell_type_counts[ct]} cells)")
    
    if len(valid_cell_types) == 0:
        raise ValueError(f"No cell types have >= {min_cells_per_type} cells! Lower --min_cells_per_type")
    
    original_n_cells = adata_sc.n_obs
    adata_sc = adata_sc[adata_sc.obs['cell_type'].isin(valid_cell_types)].copy()
    print(f"\n  Kept {adata_sc.n_obs}/{original_n_cells} cells from {len(valid_cell_types)} cell types")
    
    # 1. Normalize SC
    print("\nNormalizing Single Cell data...")
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    
    # 2. Normalize ST
    print("Normalizing Spatial data...")
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    
    # 3. Compute PCA on SC for cell state analysis
    print("Computing PCA on single-cell data for state analysis...")
    sc.pp.highly_variable_genes(adata_sc, n_top_genes=2000, flavor='seurat_v3', 
                                 layer='counts', subset=False)
    sc.pp.pca(adata_sc, n_comps=50, use_highly_variable=True)
    
    # 4. Find Marker Genes
    print(f"Finding top {n_markers} marker genes per cell type...")
    sc.tl.rank_genes_groups(adata_sc, groupby="cell_type", use_raw=False)
    
    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).head(n_markers)
    markers = list(set(markers_df.values.flatten()))
    
    # 5. Intersect with Spatial Genes
    common_markers = list(set(markers).intersection(set(adata_st.var_names)))
    
    print(f"  Total unique markers in SC: {len(markers)}")
    print(f"  Markers present in ST: {len(common_markers)}")
    
    if len(common_markers) < 10:
        raise ValueError("Too few common marker genes!")
        
    # 6. Prepare Tangram Data 
    tg.pp_adatas(adata_sc, adata_st, genes=common_markers)
    
    return adata_sc, adata_st


def run_tangram(adata_sc, adata_st, epochs=1000, gpu="0"):
    """Run Tangram mapping and get cell-to-spot mapping matrix."""
    print("\n" + "=" * 70)
    print("RUNNING TANGRAM MAPPING")
    print("=" * 70)
    
    # Set Device
    device = "cpu"
    if gpu != "cpu":
        if torch.cuda.is_available():
            device = f"cuda:{gpu}"
            print(f"Using GPU: {device}")
        else:
            print("GPU requested but not available. Using CPU.")
    else:
        print("Using CPU.")
    
    # Map Cells to Space
    print(f"Starting mapping ({epochs} epochs)...")
    
    ad_map = tg.map_cells_to_space(
        adata_sc, 
        adata_st,
        mode="cells",
        density_prior='rna_count_based', 
        num_epochs=epochs,
        device=device
    )
    
    print("Mapping complete.")
    
    # Project Cell Types
    print("Projecting cell type annotations to space...")
    tg.project_cell_annotations(ad_map, adata_st, annotation="cell_type")
    
    # Get results
    prediction_df = adata_st.obsm['tangram_ct_pred']
    
    # Sort columns alphabetically for consistent ordering
    prediction_df = prediction_df[sorted(prediction_df.columns)]
    
    # Store mapping matrix for cell state analysis
    mapping_matrix = ad_map.X  
    
    return prediction_df, mapping_matrix, ad_map


# =============================================================================
# SPATIAL VISUALIZATION FUNCTIONS
# =============================================================================

def plot_spatial_intensity_maps(proportions_df, coords_df, output_path, coords_full=None, 
                                 hexagon_orientation=30, image_rotation=0):
    """
    Generates hexagonal maps matching VisualizationUtils.plot_spatial_maps.
    
    Args:
        proportions_df: DataFrame with cell type proportions per spot
        coords_df: DataFrame with barcode, x, y columns
        output_path: Path to save the figure
        coords_full: Full coordinate array for background spots
        hexagon_orientation: Hexagon orientation in degrees (30=pointy-top, 0=flat-top)
        image_rotation: Rotation angle for entire image (0=upright)
    """
    print(f"Generating spatial intensity maps (Hexagonal)...")
    import math

    # 1. Prepare Coordinates
    if coords_full is None:
        coords_full = coords_df[['x', 'y']].values

    bg_x, bg_y = coords_full[:, 0], coords_full[:, 1]
    
    # Only rotate if image_rotation is non-zero
    if image_rotation != 0:
        bg_x, bg_y = rotate_coordinates(bg_x, bg_y, image_rotation)
    
    coords_full_rot = np.column_stack([bg_x, bg_y])
    
    hex_radius = _calculate_hex_radius(coords_full_rot, orientation=hexagon_orientation)

    # Prepare Foreground
    common_spots = proportions_df.index.intersection(coords_df['barcode'])
    if len(common_spots) == 0:
        print("  Warning: No matching spots found for visualization")
        return
    
    coords_matched = coords_df.set_index('barcode').loc[common_spots]
    matched_x, matched_y = coords_matched['x'].values, coords_matched['y'].values
    
    if image_rotation != 0:
        matched_x, matched_y = rotate_coordinates(matched_x, matched_y, image_rotation)
    
    coords_data = np.column_stack([matched_x, matched_y])
    
    # 2. Setup Grid - ALPHABETICALLY SORTED cell types
    cell_types = sorted(proportions_df.columns.tolist())
    n_types = len(cell_types)
    cols = 4
    rows = math.ceil(n_types / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_types > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Set bounds
    x_min, x_max = coords_full_rot[:, 0].min(), coords_full_rot[:, 0].max()
    y_min, y_max = coords_full_rot[:, 1].min(), coords_full_rot[:, 1].max()
    pad = hex_radius * 2

    for i, ct in enumerate(cell_types):
        ax = axes[i]
        values = proportions_df.loc[common_spots, ct].values
        
        # A. Background - grey hexagons for all spots
        bg_patches = []
        for (x, y) in coords_full_rot:
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=np.radians(hexagon_orientation)  # Use hexagon orientation
            )
            bg_patches.append(hexagon)
        
        bg_collection = PatchCollection(
            bg_patches, 
            facecolors='lightgrey', 
            edgecolors='none', 
            alpha=0.3,
            zorder=0
        )
        ax.add_collection(bg_collection)
        
        # B. Foreground - colored hexagons for data spots
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.plasma
        
        patches = []
        colors = []
        for j, (x, y) in enumerate(coords_data):
            hexagon = RegularPolygon(
                (x, y), 
                numVertices=6, 
                radius=hex_radius, 
                orientation=np.radians(hexagon_orientation)  # Use hexagon orientation
            )
            patches.append(hexagon)
            colors.append(cmap(norm(values[j])))
        
        collection = PatchCollection(patches, facecolors=colors, edgecolors='none', zorder=1)
        ax.add_collection(collection)
        
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
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

    for i in range(n_types, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_spatial_dominant_type(proportions_df, coords_df, output_path, coords_full=None,
                                hexagon_orientation=30, image_rotation=0):
    """
    Generates dominant type map with tessellating hexagons.
    
    Args:
        proportions_df: DataFrame with cell type proportions per spot
        coords_df: DataFrame with barcode, x, y columns
        output_path: Path to save the figure
        coords_full: Full coordinate array for background spots
        hexagon_orientation: Hexagon orientation in degrees (30=pointy-top, 0=flat-top)
        image_rotation: Rotation angle for entire image (0=upright)
    """
    print(f"Generating spatial dominant type map (Hexagonal)...")
    
    # 1. Prepare Coordinates
    if coords_full is None:
        coords_full = coords_df[['x', 'y']].values
    
    bg_x, bg_y = coords_full[:, 0], coords_full[:, 1]
    if image_rotation != 0:
        bg_x, bg_y = rotate_coordinates(bg_x, bg_y, image_rotation)
    coords_full_rot = np.column_stack([bg_x, bg_y])
    
    hex_radius = _calculate_hex_radius(coords_full_rot, orientation=hexagon_orientation)
    
    # 2. Prepare Foreground
    common_spots = proportions_df.index.intersection(coords_df['barcode'])
    if len(common_spots) == 0:
        print("  Warning: No matching spots found for visualization")
        return
    
    coords_matched = coords_df.set_index('barcode').loc[common_spots]
    matched_x, matched_y = coords_matched['x'].values, coords_matched['y'].values
    if image_rotation != 0:
        matched_x, matched_y = rotate_coordinates(matched_x, matched_y, image_rotation)
    coords_data = np.column_stack([matched_x, matched_y])

    # 3. Determine Dominant - use ALPHABETICALLY SORTED cell types for consistent color mapping
    props_matched = proportions_df.loc[common_spots]
    cell_types = sorted(props_matched.columns.tolist())  # Alphabetically sorted
    n_types = len(cell_types)
    
    # Reorder columns alphabetically before getting dominant indices
    props_sorted = props_matched[cell_types]
    dominant_indices = np.argmax(props_sorted.values, axis=1)
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    cmap = plt.colormaps.get_cmap('tab20').resampled(n_types)

    # A. Background - grey hexagons
    bg_patches = []
    for (x, y) in coords_full_rot:
        hexagon = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=hex_radius,
            orientation=np.radians(hexagon_orientation)
        )
        bg_patches.append(hexagon)
    
    bg_collection = PatchCollection(
        bg_patches, 
        facecolors='lightgrey', 
        edgecolors='none', 
        alpha=0.3,
        zorder=0
    )
    ax.add_collection(bg_collection)
    
    # B. Foreground - colored by dominant type
    patches = []
    colors = []
    for j, (x, y) in enumerate(coords_data):
        type_idx = dominant_indices[j]
        color_val = type_idx / max(n_types - 1, 1) if n_types > 1 else 0
        hexagon = RegularPolygon(
            (x, y), 
            numVertices=6, 
            radius=hex_radius, 
            orientation=np.radians(hexagon_orientation)
        )
        patches.append(hexagon)
        colors.append(cmap(color_val))

    collection = PatchCollection(patches, facecolors=colors, edgecolors='none', zorder=1)
    ax.add_collection(collection)
    
    # Bounds
    x_min, x_max = coords_full_rot[:, 0].min(), coords_full_rot[:, 0].max()
    y_min, y_max = coords_full_rot[:, 1].min(), coords_full_rot[:, 1].max()
    pad = hex_radius * 2
    
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    
    # Legend - alphabetically sorted
    handles = []
    for i, ct in enumerate(cell_types):
        cval = i / max(n_types - 1, 1) if n_types > 1 else 0
        handles.append(plt.Line2D([0], [0], marker='H', color='w', markerfacecolor=cmap(cval),
                      label=ct, markersize=12, markeredgecolor='none'))
    
    if len(coords_full_rot) > len(coords_data):
        handles.append(plt.Line2D([0], [0], marker='H', color='w', markerfacecolor='lightgrey',
                      alpha=0.5, label='No count data', markersize=12, markeredgecolor='none'))

    ax.legend(handles=handles, title="Cell Type", bbox_to_anchor=(1.02, 1), 
              loc='upper left', frameon=True, fontsize=10)
    
    plt.title("Dominant Cell Type per Spot", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cooccurrence_heatmap(proportions_df, output_path):
    """
    Generates a Pearson correlation heatmap (Reference Style).
    Cell types are ordered alphabetically.
    """
    print(f"Generating co-occurrence heatmap...")
    
    # Sort columns alphabetically
    sorted_cols = sorted(proportions_df.columns.tolist())
    proportions_sorted = proportions_df[sorted_cols]
    
    plt.figure(figsize=(10, 8))
    corr = proportions_sorted.corr(method='pearson')
    
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
    print(f"  Saved: {output_path}")


# =============================================================================
# CELL STATE ANALYSIS FUNCTIONS
# =============================================================================

def compute_spot_cell_states(adata_sc, adata_st, mapping_matrix, proportions_df, 
                              n_pcs=30, min_proportion=0.05):
    """
    Compute cell-type-specific states for each spot.
    """
    print("\n" + "=" * 70)
    print("COMPUTING CELL TYPE STATES PER SPOT")
    print("=" * 70)
    
    # Get cell type information
    cell_types = adata_sc.obs['cell_type'].values
    unique_cell_types = sorted(adata_sc.obs['cell_type'].unique())  # Alphabetically sorted
    
    # Get PCA embeddings from single cells
    if 'X_pca' not in adata_sc.obsm:
        print("Computing PCA...")
        sc.pp.pca(adata_sc, n_comps=n_pcs)
    
    sc_pca = adata_sc.obsm['X_pca'][:, :n_pcs]
    
    # Normalize mapping matrix per spot
    mapping_normalized = mapping_matrix / (mapping_matrix.sum(axis=0, keepdims=True) + 1e-10)
    
    n_spots = adata_st.n_obs
    spot_states = {}
    state_variance = {}
    
    print(f"Computing states for {len(unique_cell_types)} cell types across {n_spots} spots...")
    
    for ct in unique_cell_types:
        ct_mask = (cell_types == ct)
        ct_indices = np.where(ct_mask)[0]
        
        if len(ct_indices) == 0:
            continue
            
        ct_pca = sc_pca[ct_indices]
        ct_mapping = mapping_normalized[ct_indices]
        
        weights_sum = ct_mapping.sum(axis=0, keepdims=True).T
        weights_sum = np.maximum(weights_sum, 1e-10)
        
        weighted_states = ct_mapping.T @ ct_pca
        spot_states[ct] = weighted_states / weights_sum
        
        weighted_sq = ct_mapping.T @ (ct_pca ** 2) / weights_sum
        state_variance[ct] = weighted_sq - (spot_states[ct] ** 2)
        
        print(f"  {ct}: computed states for {n_spots} spots")
    
    state_metadata = {
        'cell_types': unique_cell_types,
        'n_spots': n_spots,
        'n_pcs': n_pcs,
        'proportions': proportions_df,
        'variance': state_variance
    }
    
    return spot_states, state_metadata


def analyze_state_heterogeneity(spot_states, proportions_df, min_proportion=0.05):
    """
    Analyze whether cell states are heterogeneous or uniform across spots.
    """
    print("\n" + "=" * 70)
    print("ANALYZING STATE HETEROGENEITY")
    print("=" * 70)
    
    report = {}
    
    # Process in alphabetical order
    for ct in sorted(spot_states.keys()):
        states = spot_states[ct]
        ct_props = proportions_df[ct].values if ct in proportions_df.columns else np.zeros(states.shape[0])
        significant_mask = ct_props >= min_proportion
        n_significant = significant_mask.sum()
        
        if n_significant < 10:
            report[ct] = {
                'n_spots_analyzed': n_significant,
                'variability': np.nan,
                'is_uniform': True,
                'reason': 'Too few spots with significant proportion'
            }
            continue
        
        sig_states = states[significant_mask]
        
        total_var = np.var(sig_states, axis=0).sum()
        
        centroid = sig_states.mean(axis=0)
        distances = np.linalg.norm(sig_states - centroid, axis=1)
        cv = distances.std() / (distances.mean() + 1e-10)
        
        state_scale = np.var(sig_states)
        is_uniform = total_var < 0.1 * state_scale * sig_states.shape[1]
        
        report[ct] = {
            'n_spots_analyzed': n_significant,
            'total_variance': total_var,
            'mean_distance_to_centroid': distances.mean(),
            'distance_cv': cv,
            'variability': cv,
            'is_uniform': is_uniform
        }
        
        status = "UNIFORM" if is_uniform else "HETEROGENEOUS"
        print(f"  {ct}: {status} (CV={cv:.3f}, n_spots={n_significant})")
    
    return report


def create_continuum_plots(spot_states, proportions_df, state_report, output_dir,
                           min_proportion=0.05, umap_neighbors=30):
    """
    Create cell state continuum UMAP plots for each cell type.
    """
    print("\n" + "=" * 70)
    print("GENERATING CELL STATE CONTINUUM PLOTS")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in alphabetical order
    for ct in sorted(spot_states.keys()):
        states = spot_states[ct]
        
        if ct not in proportions_df.columns:
            continue
            
        ct_props = proportions_df[ct].values
        significant_mask = ct_props >= min_proportion
        n_significant = significant_mask.sum()
        
        if n_significant < 20:
            print(f"  {ct}: Skipping (only {n_significant} spots with proportion >= {min_proportion})")
            continue
        
        sig_states = states[significant_mask]
        sig_props = ct_props[significant_mask]
        
        # Dimensionality reduction
        if HAS_UMAP and n_significant > umap_neighbors:
            reducer = umap.UMAP(
                n_neighbors=min(umap_neighbors, n_significant - 1),
                min_dist=0.3,
                random_state=42
            )
            embedding = reducer.fit_transform(sig_states)
            method = "UMAP"
        else:
            if n_significant > 50:
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_significant // 3))
            else:
                reducer = PCA(n_components=2)
            embedding = reducer.fit_transform(sig_states)
            method = "t-SNE" if n_significant > 50 else "PCA"
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Colored by proportion
        sc1 = axes[0].scatter(
            embedding[:, 0], embedding[:, 1],
            c=sig_props, cmap='viridis', s=30, alpha=0.7
        )
        axes[0].set_title(f'{ct} - Colored by Proportion')
        axes[0].set_xlabel(f'{method} 1')
        axes[0].set_ylabel(f'{method} 2')
        plt.colorbar(sc1, ax=axes[0], label='Proportion')
        
        # Plot 2: Colored by first principal state component
        state_pc1 = sig_states[:, 0]
        sc2 = axes[1].scatter(
            embedding[:, 0], embedding[:, 1],
            c=state_pc1, cmap='coolwarm', s=30, alpha=0.7
        )
        axes[1].set_title(f'{ct} - Colored by State PC1')
        axes[1].set_xlabel(f'{method} 1')
        axes[1].set_ylabel(f'{method} 2')
        plt.colorbar(sc2, ax=axes[1], label='State PC1')
        
        # Plot 3: Colored by state magnitude
        centroid = sig_states.mean(axis=0)
        distances = np.linalg.norm(sig_states - centroid, axis=1)
        sc3 = axes[2].scatter(
            embedding[:, 0], embedding[:, 1],
            c=distances, cmap='plasma', s=30, alpha=0.7
        )
        axes[2].set_title(f'{ct} - Distance from Mean State')
        axes[2].set_xlabel(f'{method} 1')
        axes[2].set_ylabel(f'{method} 2')
        plt.colorbar(sc3, ax=axes[2], label='Distance')
        
        if ct in state_report:
            status = "UNIFORM" if state_report[ct].get('is_uniform', True) else "HETEROGENEOUS"
            cv = state_report[ct].get('variability', 0)
            fig.suptitle(f'{ct} Cell State Continuum ({status}, CV={cv:.3f})', fontsize=14)
        
        plt.tight_layout()
        
        safe_ct_name = ct.replace('/', '_').replace(' ', '_').replace('+', 'pos').replace('-', 'neg')
        plot_path = os.path.join(output_dir, f'continuum_{safe_ct_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_path}")
    
    # Create combined plot
    create_full_continuum_plot(
        spot_states, proportions_df, state_report, 
        output_dir, min_proportion, umap_neighbors
    )


def create_full_continuum_plot(spot_states, proportions_df, state_report,
                                output_dir, min_proportion=0.05, umap_neighbors=30):
    """
    Create a combined UMAP showing all spots, with various coloring options.
    """
    print("\n  Creating full continuum plot (all cell types combined)...")
    
    n_spots = proportions_df.shape[0]
    cell_types = sorted(spot_states.keys())  # Alphabetically sorted
    
    dominant_ct = proportions_df.idxmax(axis=1).values
    
    first_ct = cell_types[0]
    n_pcs = spot_states[first_ct].shape[1]
    
    # Create proportion-weighted combined states
    weighted_combined = np.zeros((n_spots, n_pcs))
    for ct in cell_types:
        if ct in proportions_df.columns:
            weights = proportions_df[ct].values.reshape(-1, 1)
            weighted_combined += weights * spot_states[ct]
    
    # Dimensionality reduction
    if HAS_UMAP and n_spots > umap_neighbors:
        reducer = umap.UMAP(
            n_neighbors=min(umap_neighbors, n_spots - 1),
            min_dist=0.3,
            random_state=42
        )
        embedding = reducer.fit_transform(weighted_combined)
        method = "UMAP"
    else:
        if n_spots > 50:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_spots // 3))
            method = "t-SNE"
        else:
            reducer = PCA(n_components=2)
            method = "PCA"
        embedding = reducer.fit_transform(weighted_combined)
    
    # Create multi-panel figure
    n_cell_types = len(cell_types)
    n_cols = min(4, n_cell_types + 2)
    n_rows = (n_cell_types + 2 + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    
    # Panel 1: Colored by dominant cell type
    ax1 = fig.add_subplot(n_rows, n_cols, 1)
    unique_cts = sorted(proportions_df.columns)  # Alphabetically sorted
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cts)))
    
    # FIXED LINE HERE: added "ct" to the enumeration
    ct_to_color = {ct: colors[i] for i, ct in enumerate(unique_cts)}
    
    for ct in unique_cts:
        mask = dominant_ct == ct
        if mask.sum() > 0:
            ax1.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=[ct_to_color[ct]], label=ct, s=20, alpha=0.7
            )
    ax1.set_title('Colored by Dominant Cell Type')
    ax1.set_xlabel(f'{method} 1')
    ax1.set_ylabel(f'{method} 2')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6)
    
    # Panel 2: Colored by entropy
    ax2 = fig.add_subplot(n_rows, n_cols, 2)
    props_array = proportions_df.values
    spot_entropy = np.array([entropy(p + 1e-10) for p in props_array])
    sc2 = ax2.scatter(
        embedding[:, 0], embedding[:, 1],
        c=spot_entropy, cmap='viridis', s=20, alpha=0.7
    )
    ax2.set_title('Colored by Cell Type Entropy (Mixing)')
    ax2.set_xlabel(f'{method} 1')
    ax2.set_ylabel(f'{method} 2')
    plt.colorbar(sc2, ax=ax2, label='Entropy')
    
    # Remaining panels: One per cell type (alphabetically)
    for idx, ct in enumerate(cell_types[:min(n_cols * n_rows - 2, len(cell_types))]):
        ax = fig.add_subplot(n_rows, n_cols, idx + 3)
        
        if ct in proportions_df.columns:
            ct_props = proportions_df[ct].values
            
            sc = ax.scatter(
                embedding[:, 0], embedding[:, 1],
                c=ct_props, cmap='Reds', s=20, alpha=0.7,
                vmin=0, vmax=max(0.1, ct_props.max())
            )
            plt.colorbar(sc, ax=ax, label='Proportion')
            
            if ct in state_report:
                status = "U" if state_report[ct].get('is_uniform', True) else "H"
                ax.set_title(f'{ct} [{status}]', fontsize=10)
            else:
                ax.set_title(ct, fontsize=10)
        
        ax.set_xlabel(f'{method} 1')
        ax.set_ylabel(f'{method} 2')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'continuum_full_all_celltypes.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {plot_path}")


def save_cell_states(spot_states, state_report, proportions_df, output_dir):
    """Save cell state data to files."""
    print("\n" + "=" * 70)
    print("SAVING CELL STATE DATA")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save state matrices as NPZ (alphabetically sorted keys)
    state_arrays = {ct: spot_states[ct] for ct in sorted(spot_states.keys())}
    npz_path = os.path.join(output_dir, 'spot_cell_states.npz')
    np.savez(npz_path, **state_arrays)
    print(f"  Saved state matrices: {npz_path}")
    
    # Save heterogeneity report as CSV (sorted by cell type)
    report_df = pd.DataFrame(state_report).T
    report_df = report_df.sort_index()
    report_path = os.path.join(output_dir, 'state_heterogeneity_report.csv')
    report_df.to_csv(report_path)
    print(f"  Saved heterogeneity report: {report_path}")
    
    # Save per-cell-type state CSVs (alphabetically)
    for ct in sorted(spot_states.keys()):
        states = spot_states[ct]
        safe_ct_name = ct.replace('/', '_').replace(' ', '_').replace('+', 'pos').replace('-', 'neg')
        
        cols = [f'State_PC{i+1}' for i in range(states.shape[1])]
        state_df = pd.DataFrame(states, columns=cols, index=proportions_df.index)
        
        if ct in proportions_df.columns:
            state_df['proportion'] = proportions_df[ct].values
        
        csv_path = os.path.join(output_dir, f'states_{safe_ct_name}.csv')
        state_df.to_csv(csv_path)
    
    print(f"  Saved individual cell type state files")


def create_spatial_state_maps(spot_states, proportions_df, coords_df, output_dir, coords_full=None,
                              hexagon_orientation=30, image_rotation=0, min_proportion=0.05):
    """
    Create spatial maps of cell states (State PC1) and proportions for each cell type in situ.
    """
    print("\n  Generating spatial state maps (in situ)...")
    
    if coords_full is None:
        coords_full = coords_df[['x', 'y']].values

    bg_x, bg_y = coords_full[:, 0], coords_full[:, 1]
    if image_rotation != 0:
        bg_x, bg_y = rotate_coordinates(bg_x, bg_y, image_rotation)
    coords_full_rot = np.column_stack([bg_x, bg_y])
    hex_radius = _calculate_hex_radius(coords_full_rot, orientation=hexagon_orientation)

    common_spots = proportions_df.index.intersection(coords_df['barcode'])
    if len(common_spots) == 0:
        return

    coords_matched = coords_df.set_index('barcode').loc[common_spots]
    matched_x, matched_y = coords_matched['x'].values, coords_matched['y'].values
    if image_rotation != 0:
        matched_x, matched_y = rotate_coordinates(matched_x, matched_y, image_rotation)
    coords_data = np.column_stack([matched_x, matched_y])

    # Align common spots with array indices
    common_indices = [proportions_df.index.get_loc(spot) for spot in common_spots]

    x_min, x_max = coords_full_rot[:, 0].min(), coords_full_rot[:, 0].max()
    y_min, y_max = coords_full_rot[:, 1].min(), coords_full_rot[:, 1].max()
    pad = hex_radius * 2

    for ct in sorted(spot_states.keys()):
        if ct not in proportions_df.columns:
            continue

        ct_props = proportions_df[ct].values[common_indices]
        ct_states = spot_states[ct][common_indices, 0]  # Use State PC1 as primary state metric

        present_mask = ct_props >= min_proportion
        n_present = present_mask.sum()
        n_absent = (~present_mask).sum()

        if n_present < 5:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        def add_background(ax):
            bg_patches = [RegularPolygon((x, y), numVertices=6, radius=hex_radius, 
                                         orientation=np.radians(hexagon_orientation))
                          for x, y in coords_full_rot]
            ax.add_collection(PatchCollection(bg_patches, facecolors='#E0E0E0', 
                                              edgecolors='none', alpha=0.4, zorder=0))

        # ==================== Panel 1: State Map ====================
        ax1 = axes[0]
        add_background(ax1)

        if n_absent > 0:
            absent_patches = [RegularPolygon((coords_data[j, 0], coords_data[j, 1]), numVertices=6, 
                                             radius=hex_radius, orientation=np.radians(hexagon_orientation))
                              for j in np.where(~present_mask)[0]]
            ax1.add_collection(PatchCollection(absent_patches, facecolors='#F5F5F5', edgecolors='#CCCCCC', 
                                               linewidths=0.3, alpha=0.7, zorder=1))

        present_state = ct_states[present_mask]
        vmin, vmax = present_state.min(), present_state.max()
        if vmax - vmin < 1e-8: vmax = vmin + 1e-8
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.coolwarm

        present_patches = [RegularPolygon((coords_data[j, 0], coords_data[j, 1]), numVertices=6, 
                                          radius=hex_radius, orientation=np.radians(hexagon_orientation))
                           for j in np.where(present_mask)[0]]
        ax1.add_collection(PatchCollection(present_patches, facecolors=cmap(norm(present_state)), 
                                           edgecolors='none', zorder=2))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04, shrink=0.7)
        cbar.set_label("Cell State (PC1)", fontsize=10)

        ax1.set_xlim(x_min - pad, x_max + pad)
        ax1.set_ylim(y_min - pad, y_max + pad)
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        ax1.axis('off')
        ax1.set_title(f"{ct}\nSpatial Cell State", fontsize=14, fontweight='bold')

        # ==================== Panel 2: Proportion Map ====================
        ax2 = axes[1]
        add_background(ax2)

        if n_absent > 0:
            absent_patches = [RegularPolygon((coords_data[j, 0], coords_data[j, 1]), numVertices=6, 
                                             radius=hex_radius, orientation=np.radians(hexagon_orientation))
                              for j in np.where(~present_mask)[0]]
            ax2.add_collection(PatchCollection(absent_patches, facecolors='#F5F5F5', edgecolors='#CCCCCC', 
                                               linewidths=0.3, alpha=0.7, zorder=1))

        present_props = ct_props[present_mask]
        vmin, vmax = 0, present_props.max()
        if vmax < 1e-8: vmax = 1e-8
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.plasma

        present_patches = [RegularPolygon((coords_data[j, 0], coords_data[j, 1]), numVertices=6, 
                                          radius=hex_radius, orientation=np.radians(hexagon_orientation))
                           for j in np.where(present_mask)[0]]
        ax2.add_collection(PatchCollection(present_patches, facecolors=cmap(norm(present_props)), 
                                           edgecolors='none', zorder=2))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04, shrink=0.7)
        cbar.set_label("Proportion", fontsize=10)

        ax2.set_xlim(x_min - pad, x_max + pad)
        ax2.set_ylim(y_min - pad, y_max + pad)
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        ax2.axis('off')
        ax2.set_title(f"{ct}\nSpatial Proportion", fontsize=14, fontweight='bold')

        safe_ct_name = ct.replace('/', '_').replace(' ', '_').replace('+', 'pos').replace('-', 'neg')
        out_path = os.path.join(output_dir, f"spatial_state_{safe_ct_name}.png")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {out_path}")


def main():
    args = parse_arguments()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.output_csv)
    os.makedirs(args.output_dir, exist_ok=True)
    
    continuum_dir = os.path.join(args.output_dir, 'cell_state_continuum')
    
    # 1. Load Data
    adata_sc, adata_st, st_df_original = load_data(
        args.sc_counts, args.sc_labels, args.st_counts,
        transpose_sc=args.transpose_sc, transpose_st=args.transpose_st
    )
    
    # 2. Load/Extract Spatial Coordinates
    coords_df, coords_full, cols_to_drop = load_coordinates(
        args.st_coords, 
        st_barcodes=adata_st.obs_names,
        st_df=st_df_original
    )
    
    # Store spatial info in adata_st
    adata_st.obsm['spatial'] = coords_df.set_index('barcode').loc[adata_st.obs_names][['x', 'y']].values
    adata_st.uns['spatial_coords_df'] = coords_df
    adata_st.uns['spatial_coords_full'] = coords_full
    
    # Remove coordinate columns from gene expression if they were embedded
    if cols_to_drop:
        genes_to_keep = [g for g in adata_st.var_names if g not in cols_to_drop]
        if len(genes_to_keep) < adata_st.n_vars:
            print(f"  Removing {len(cols_to_drop)} coordinate columns from expression data")
            adata_st = adata_st[:, genes_to_keep].copy()
    
    # 3. Preprocess
    adata_sc, adata_st = preprocess_tangram(
        adata_sc, adata_st, 
        n_markers=args.n_markers,
        min_cells_per_type=args.min_cells_per_type
    )
    
    # 4. Run Tangram
    epochs = args.epochs
    if args.max_epochs_st is not None:
        epochs = args.max_epochs_st
        
    proportions_df, mapping_matrix, ad_map = run_tangram(
        adata_sc, adata_st, epochs=epochs, gpu=args.gpu
    )
    
    # 5. Save Proportions
    print("\n" + "=" * 70)
    print("SAVING DECONVOLUTION RESULTS")
    print("=" * 70)
    
    # Sort columns alphabetically before saving
    proportions_df = proportions_df[sorted(proportions_df.columns)]
    
    # Save raw density
    density_path = args.output_csv.replace('.csv', '_density.csv')
    proportions_df.to_csv(density_path)
    print(f"  Saved density scores: {density_path}")
    
    # Normalize to proportions
    row_sums = proportions_df.sum(axis=1)
    norm_proportions = proportions_df.div(row_sums.replace(0, 1), axis=0)
    norm_proportions = norm_proportions.fillna(0)
    
    # Ensure alphabetical order in normalized proportions
    norm_proportions = norm_proportions[sorted(norm_proportions.columns)]
    norm_proportions.to_csv(args.output_csv)
    print(f"  Saved normalized proportions: {args.output_csv}")
    
    # 6. Proportion Heatmap
    n_spots = len(norm_proportions)
    if n_spots > 100:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Sort by name (alphabetically) for consistent display
        mean_props = norm_proportions.mean()
        mean_props = mean_props.sort_index()  # Alphabetical order
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(mean_props)))
        ax.barh(range(len(mean_props)), mean_props.values, color=colors)
        ax.set_yticks(range(len(mean_props)))
        ax.set_yticklabels(mean_props.index)
        ax.set_xlabel('Average Proportion')
        ax.set_title(f'Average Cell Type Proportions (Tangram, n={n_spots} spots)')
        plt.tight_layout()
    else:
        # Sort columns alphabetically for heatmap
        plt.figure(figsize=(12, max(8, n_spots * 0.3)))
        heatmap_data = norm_proportions[sorted(norm_proportions.columns)]
        sns.heatmap(
            heatmap_data, 
            cmap='viridis', 
            yticklabels=True,
            vmin=0, vmax=1
        )
        plt.title("Predicted Cell Type Proportions (Tangram)")
        plt.xlabel("Cell Types")
        plt.ylabel("Spots")
        plt.tight_layout()
    
    plt.savefig(args.output_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved proportion heatmap: {args.output_plot}")
    
    # 7. Spatial Visualization Maps
    print("\n" + "=" * 70)
    print("GENERATING SPATIAL VISUALIZATIONS")
    print("=" * 70)
    
    intensity_path = os.path.join(args.output_dir, 'spatial_intensity_maps.png')
    plot_spatial_intensity_maps(
        norm_proportions, 
        coords_df, 
        intensity_path, 
        coords_full=coords_full, 
        hexagon_orientation=args.hex_orientation,
        image_rotation=IMAGE_ROTATION
    )
    
    dominant_path = os.path.join(args.output_dir, 'spatial_dominant_type.png')
    plot_spatial_dominant_type(
        norm_proportions, 
        coords_df, 
        dominant_path, 
        coords_full=coords_full, 
        hexagon_orientation=args.hex_orientation,
        image_rotation=IMAGE_ROTATION
    )
    
    # Co-occurrence heatmap
    cooccurrence_path = os.path.join(args.output_dir, 'cooccurrence_heatmap.png')
    plot_cooccurrence_heatmap(norm_proportions, cooccurrence_path)

    # 8. Cell State Analysis
    if not args.skip_state_analysis:
        spot_states, state_metadata = compute_spot_cell_states(
            adata_sc, adata_st, mapping_matrix, norm_proportions,
            n_pcs=args.n_pcs, min_proportion=args.min_proportion
        )
        
        state_report = analyze_state_heterogeneity(
            spot_states, norm_proportions, min_proportion=args.min_proportion
        )
        
        save_cell_states(spot_states, state_report, norm_proportions, continuum_dir)
        
        create_continuum_plots(
            spot_states, norm_proportions, state_report, continuum_dir,
            min_proportion=args.min_proportion, umap_neighbors=args.umap_neighbors
        )
        
        # --- NEW SPATIAL IN SITU MAPS FOR CELL STATES ---
        create_spatial_state_maps(
            spot_states, norm_proportions, coords_df, continuum_dir,
            coords_full=coords_full, hexagon_orientation=args.hex_orientation,
            image_rotation=IMAGE_ROTATION, min_proportion=args.min_proportion
        )
        
    else:
        print("\n  Skipping cell state analysis (--skip_state_analysis)")
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Proportions: {args.output_csv}")
    print(f"  Heatmap: {args.output_plot}")
    print(f"  Spatial intensity maps: {intensity_path}")
    print(f"  Spatial dominant type: {dominant_path}")
    print(f"  Co-occurrence heatmap: {cooccurrence_path}")
    if not args.skip_state_analysis:
        print(f"  Cell states: {continuum_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()