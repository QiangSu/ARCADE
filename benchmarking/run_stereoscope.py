#!/usr/bin/env python3
"""
================================================================================
Stereoscope Spatial Deconvolution Pipeline (scvi-tools)
================================================================================

Enhanced with Spatial Visualization outputs:
1. Cell type proportions (CSV & Heatmap)
2. Spatial visualization maps:
   - spatial_intensity_maps.png (Hexagonal)
   - spatial_dominant_type.png (Hexagonal)
   - cooccurrence_heatmap.png

Supports TWO input formats:
1. Real Visium data: --st_coords points to separate coordinate file
2. Simulated data: Coordinates embedded in counts CSV (row/col columns)
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
import matplotlib.ticker as ticker
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree

# SCVI / Stereoscope imports
import scvi
from scvi.external import RNAStereoscope, SpatialStereoscope

# ==============================================================================
# GLOBAL SETTINGS & PLOTTING CONFIGURATION
# ==============================================================================
scvi.settings.seed = 42
warnings.filterwarnings('ignore')

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
IMAGE_ROTATION = 0

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run Stereoscope for Spatial Deconvolution (Real & Simulated Data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real Visium data (separate coordinate file):
  python run_stereoscope.py --sc_counts sc.csv --sc_labels labels.csv \\
      --st_counts st_counts.csv --st_coords tissue_positions.csv \\
      --output_csv results/proportions.csv --output_plot results/heatmap.png

  # Simulated data (coordinates in counts file):
  python run_stereoscope.py --sc_counts sc.csv --sc_labels labels.csv \\
      --st_counts simulated_spots.csv \\
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
    parser.add_argument("--st_coords", type=str, default=None,
                        help="Path to spatial coordinates CSV (optional for simulated data)")
    
    # Output Arguments
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save predicted proportions CSV")
    parser.add_argument("--output_plot", type=str, required=True,
                        help="Path to save heatmap plot")
    
    # Hyperparameters
    parser.add_argument("--n_hvg", type=int, default=5000,
                        help="Number of Highly Variable Genes (default: 5000)")
    parser.add_argument("--max_epochs_sc", type=int, default=100,
                        help="Epochs for Single-Cell training (default: 100)")
    parser.add_argument("--max_epochs_st", type=int, default=1000,
                        help="Epochs for Spatial training (default: 1000)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID (e.g., '0' or 'cpu')")
    parser.add_argument("--hex_orientation", type=int, default=0,
                        help="Hexagon orientation angle in degrees (0 = flat-top, 30 = pointy-top)")
    
    return parser.parse_args()


# =============================================================================
# COORDINATE LOADING FUNCTIONS
# =============================================================================

def load_coordinates(filepath, st_barcodes=None):
    """
    Load spatial coordinates from external file.
    
    Returns:
        matched_df: DataFrame [barcode, x, y] aligned with st_barcodes
        coords_full: Numpy array [N, 2] of ALL spots (for background)
    """
    print(f"Loading coordinates from: {filepath}")
    
    try:
        coords_df = pd.read_csv(filepath)
        
        # Handle Visium tissue_positions_list.csv (no header, 6 columns)
        if coords_df.shape[1] == 6 and coords_df.columns[0] not in ['barcode', 'Barcode', 'spot_id']:
            coords_df = pd.read_csv(filepath, header=None)
            coords_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row', 'pxl_col']
            full_df = pd.DataFrame({
                'barcode': coords_df['barcode'],
                'x': coords_df['pxl_col'].astype(float),
                'y': coords_df['pxl_row'].astype(float)
            })
        else:
            # Standard CSV detection
            barcode_col = next((c for c in coords_df.columns 
                               if c.lower() in ['barcode', 'spot_id', 'cell_id']), None)
            x_col = next((c for c in coords_df.columns 
                         if c.lower() in ['x', 'pxl_col', 'col', 'x_coord']), None)
            y_col = next((c for c in coords_df.columns 
                         if c.lower() in ['y', 'pxl_row', 'row', 'y_coord']), None)
            
            # Handle unnamed index column
            if barcode_col is None and coords_df.columns[0] == 'Unnamed: 0':
                barcode_col = 'Unnamed: 0'
            
            if barcode_col is None:
                coords_df['barcode'] = coords_df.index.astype(str)
                barcode_col = 'barcode'
                
            full_df = pd.DataFrame({
                'barcode': coords_df[barcode_col].astype(str),
                'x': coords_df[x_col].astype(float),
                'y': coords_df[y_col].astype(float)
            })

        # Store ALL coordinates for background
        coords_full = full_df[['x', 'y']].values
        
        # Match to ST Data barcodes
        if st_barcodes is not None:
            coord_bcs = set(full_df['barcode'])
            count_bcs = set(st_barcodes)
            
            # Handle Visium suffix (-1)
            if len(coord_bcs) > 0 and len(count_bcs) > 0:
                ex_coord = next(iter(coord_bcs))
                ex_count = next(iter(count_bcs))
                if str(ex_count).endswith("-1") and not str(ex_coord).endswith("-1"):
                    full_df['barcode'] = full_df['barcode'] + "-1"
                elif str(ex_coord).endswith("-1") and not str(ex_count).endswith("-1"):
                    full_df['barcode'] = full_df['barcode'].str.replace("-1", "", regex=False)

            matched_df = full_df[full_df['barcode'].isin(count_bcs)].copy()
            matched_df = matched_df.set_index('barcode').reindex(st_barcodes)
            
            if matched_df.isnull().any().any():
                n_missing = matched_df.isnull().any(axis=1).sum()
                print(f"  Warning: {n_missing} spots have no coordinates.")
                matched_df = matched_df.fillna(0)
            
            matched_df = matched_df.reset_index().rename(columns={'index': 'barcode'})
            return matched_df, coords_full
        
        return full_df, coords_full
        
    except Exception as e:
        print(f"Error loading coordinates: {e}")
        raise


def extract_coordinates_from_index(spot_indices):
    """
    Extract row/col coordinates from spot index names.
    Handles formats like: 'spot_10x20', '10x20', 'spot_10_20', '10_20'
    
    Returns:
        coords_df: DataFrame with [barcode, x, y]
        coords_array: numpy array of [x, y] coordinates
    """
    rows = []
    cols = []
    valid_indices = []
    
    patterns = [
        r'spot_(\d+)x(\d+)',      # spot_10x20
        r'^(\d+)x(\d+)$',          # 10x20
        r'spot_(\d+)_(\d+)',       # spot_10_20
        r'^(\d+)_(\d+)$',          # 10_20
        r'r(\d+)c(\d+)',           # r10c20
        r'R(\d+)C(\d+)',           # R10C20
    ]
    
    for idx in spot_indices:
        idx_str = str(idx)
        matched = False
        
        for pattern in patterns:
            match = re.search(pattern, idx_str)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                rows.append(row)
                cols.append(col)
                valid_indices.append(idx)
                matched = True
                break
        
        if not matched:
            # Try to extract any two numbers
            numbers = re.findall(r'\d+', idx_str)
            if len(numbers) >= 2:
                rows.append(int(numbers[0]))
                cols.append(int(numbers[1]))
                valid_indices.append(idx)
    
    if len(valid_indices) == 0:
        return None, None
    
    # Convert to pixel-like coordinates (scale for visualization)
    rows = np.array(rows)
    cols = np.array(cols)
    
    # Scale to reasonable pixel coordinates
    x_coords = cols * 100  # col -> x
    y_coords = rows * 100  # row -> y
    
    coords_df = pd.DataFrame({
        'barcode': valid_indices,
        'x': x_coords,
        'y': y_coords
    })
    
    coords_array = np.column_stack([x_coords, y_coords])
    
    return coords_df, coords_array


def extract_coordinates_from_columns(st_df):
    """
    Extract coordinates from row/col columns in the dataframe.
    
    Returns:
        coords_df: DataFrame with [barcode, x, y]
        coords_array: numpy array of coordinates
        gene_columns: list of gene column names (excluding coordinate columns)
    """
    # Identify coordinate columns
    row_col = next((c for c in st_df.columns 
                   if c.lower() in ['row', 'array_row', 'spot_row']), None)
    col_col = next((c for c in st_df.columns 
                   if c.lower() in ['col', 'column', 'array_col', 'spot_col']), None)
    
    if row_col is None or col_col is None:
        return None, None, None
    
    # Extract coordinates
    rows = st_df[row_col].values
    cols = st_df[col_col].values
    
    # Convert to pixel coordinates
    x_coords = cols * 100
    y_coords = rows * 100
    
    coords_df = pd.DataFrame({
        'barcode': st_df.index.astype(str),
        'x': x_coords,
        'y': y_coords
    })
    
    coords_array = np.column_stack([x_coords, y_coords])
    
    # Identify gene columns (exclude coordinate columns)
    non_gene_cols = {row_col, col_col}
    gene_columns = [c for c in st_df.columns if c not in non_gene_cols]
    
    return coords_df, coords_array, gene_columns


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data(sc_path, sc_labels_path, st_path, st_coords_path=None):
    """
    Load single-cell and spatial transcriptomics data.
    
    Handles two scenarios:
    1. Real data: Separate coordinate file provided
    2. Simulated data: Coordinates embedded in counts file
    
    Returns:
        adata_sc: AnnData for single-cell reference
        adata_st: AnnData for spatial data (with spatial info in .obsm and .uns)
    """
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Load Single-Cell Data
    # -------------------------------------------------------------------------
    print(f"Loading SC counts: {sc_path}")
    sc_df = pd.read_csv(sc_path, index_col=0)
    print(f"  Raw SC shape: {sc_df.shape}")
    
    # Smart auto-transpose for SC data
    sc_df = smart_transpose(sc_df, data_type="SC")
    
    # Load labels
    lbl_df = pd.read_csv(sc_labels_path, index_col=0)
    label_col = next((c for c in ['cell_type', 'celltype', 'CellType', 'cluster', 'annotation'] 
                     if c in lbl_df.columns), lbl_df.columns[0])
    
    # Match cells
    common_cells = sc_df.index.intersection(lbl_df.index)
    if len(common_cells) == 0:
        print("  Warning: No common indices between SC counts and labels. Matching by order.")
        common_cells = sc_df.index[:len(lbl_df)]
        lbl_df.index = common_cells
    
    print(f"  SC cells with labels: {len(common_cells)}")
    
    adata_sc = anndata.AnnData(sc_df.loc[common_cells])
    adata_sc.obs['cell_type'] = lbl_df.loc[common_cells, label_col].values
    
    print(f"  SC shape: {adata_sc.shape} (cells × genes)")
    print(f"  Cell types: {adata_sc.obs['cell_type'].nunique()}")
    print(f"  SC genes (first 5): {list(adata_sc.var_names[:5])}")
    
    # Store SC gene set for ST orientation detection
    sc_genes = set(adata_sc.var_names)
    
    # -------------------------------------------------------------------------
    # Load Spatial Data
    # -------------------------------------------------------------------------
    print(f"\nLoading ST counts: {st_path}")
    st_df = pd.read_csv(st_path, index_col=0)
    print(f"  Raw ST shape: {st_df.shape}")
    
    # Smart auto-transpose for ST data using SC genes as reference
    st_df = smart_transpose(st_df, data_type="ST", reference_genes=sc_genes)
    
    coords_df = None
    coords_full = None
    gene_columns = list(st_df.columns)
    
    # -------------------------------------------------------------------------
    # Coordinate Extraction Strategy
    # -------------------------------------------------------------------------
    
    # Strategy 1: External coordinate file provided
    if st_coords_path is not None and os.path.exists(st_coords_path):
        print(f"  Loading coordinates from external file: {st_coords_path}")
        coords_df, coords_full = load_coordinates(st_coords_path, st_barcodes=st_df.index)
        print(f"  Loaded {len(coords_df)} coordinate entries")
    
    # Strategy 2: Check for row/col columns in the counts file
    if coords_df is None:
        print("  Checking for coordinate columns in counts file...")
        coords_df, coords_full, extracted_genes = extract_coordinates_from_columns(st_df)
        
        if coords_df is not None:
            print(f"  Found row/col columns! Extracted {len(coords_df)} coordinates")
            gene_columns = extracted_genes
        else:
            print("  No row/col columns found")
    
    # Strategy 3: Parse coordinates from index names
    if coords_df is None:
        print("  Attempting to parse coordinates from spot names...")
        coords_df, coords_full = extract_coordinates_from_index(st_df.index)
        
        if coords_df is not None:
            print(f"  Parsed coordinates from {len(coords_df)} spot names")
        else:
            print("  Could not parse coordinates from spot names")
    
    # Strategy 4: Generate grid coordinates as fallback
    if coords_df is None:
        print("  WARNING: No coordinates found. Generating grid layout for visualization.")
        n_spots = len(st_df)
        grid_size = int(np.ceil(np.sqrt(n_spots)))
        
        x_coords = np.array([(i % grid_size) * 100 for i in range(n_spots)])
        y_coords = np.array([(i // grid_size) * 100 for i in range(n_spots)])
        
        coords_df = pd.DataFrame({
            'barcode': st_df.index.astype(str),
            'x': x_coords,
            'y': y_coords
        })
        coords_full = np.column_stack([x_coords, y_coords])
        print(f"  Generated {grid_size}x{grid_size} grid coordinates")
    
    # -------------------------------------------------------------------------
    # Create Spatial AnnData
    # -------------------------------------------------------------------------
    # Filter to gene columns only
    gene_columns = [c for c in gene_columns if c in st_df.columns]
    adata_st = anndata.AnnData(st_df[gene_columns])
    
    print(f"  ST shape: {adata_st.shape} (spots × genes)")
    print(f"  ST genes (first 5): {list(adata_st.var_names[:5])}")
    
    # Verify gene overlap
    common_genes = set(adata_sc.var_names) & set(adata_st.var_names)
    print(f"  Common genes with SC: {len(common_genes)}")
    
    if len(common_genes) < 100:
        print("\n  ERROR: Very few common genes!")
        print(f"  SC genes (first 10): {list(adata_sc.var_names[:10])}")
        print(f"  ST genes (first 10): {list(adata_st.var_names[:10])}")
        raise ValueError(f"Only {len(common_genes)} common genes. Check data orientation.")
    
    # Store spatial information in AnnData
    if coords_df is not None:
        # Ensure coords_df is aligned with adata_st
        coords_df_indexed = coords_df.set_index('barcode')
        matched_coords = coords_df_indexed.reindex(adata_st.obs_names)
        
        if matched_coords.isnull().any().any():
            n_missing = matched_coords.isnull().any(axis=1).sum()
            print(f"  Warning: {n_missing} spots missing coordinates, filling with zeros")
            matched_coords = matched_coords.fillna(0)
        
        adata_st.obsm['spatial'] = matched_coords[['x', 'y']].values
        adata_st.uns['spatial_coords_df'] = coords_df
        adata_st.uns['spatial_coords_full'] = coords_full
        
        print(f"  Spatial coordinates stored in adata_st.obsm['spatial']")
    
    return adata_sc, adata_st


def smart_transpose(df, data_type="ST", reference_genes=None):
    """
    Intelligently determine if a dataframe needs transposition.
    
    Uses multiple heuristics:
    1. Gene name matching with reference (if provided)
    2. Barcode/spot name pattern detection
    3. Common gene name patterns
    
    Args:
        df: DataFrame to potentially transpose
        data_type: "SC" or "ST" for logging
        reference_genes: Set of known gene names (from SC data)
    
    Returns:
        DataFrame in correct orientation (samples × genes)
    """
    print(f"  {data_type} format detection:")
    
    col_sample = [str(c) for c in df.columns[:20]]
    row_sample = [str(r) for r in df.index[:20]]
    
    # Heuristic 1: Check if columns/rows match reference genes
    if reference_genes is not None and len(reference_genes) > 0:
        col_gene_overlap = len(set(df.columns.astype(str)) & reference_genes)
        row_gene_overlap = len(set(df.index.astype(str)) & reference_genes)
        
        print(f"    Reference gene overlap - Columns: {col_gene_overlap}, Rows: {row_gene_overlap}")
        
        if col_gene_overlap > 100 and col_gene_overlap > row_gene_overlap:
            print(f"    → Columns are genes (correct orientation)")
            return df
        elif row_gene_overlap > 100 and row_gene_overlap > col_gene_overlap:
            print(f"    → Rows are genes, transposing...")
            return df.T
    
    # Heuristic 2: Check for barcode/spot patterns
    barcode_patterns = [
        r'^[ACGT]{16}-\d+$',      # Visium barcode: ACGCCTGACACGCGCT-1
        r'^spot_\d+_\d+$',         # Simulated: spot_0_1
        r'^spot_\d+x\d+$',         # Simulated: spot_0x1
        r'^\d+x\d+$',              # Grid: 0x1
        r'^cell_\d+$',             # cell_0
        r'^Cell\d+$',              # Cell0
    ]
    
    def matches_barcode_pattern(names):
        count = 0
        for name in names[:20]:
            for pattern in barcode_patterns:
                if re.match(pattern, str(name)):
                    count += 1
                    break
        return count
    
    col_barcode_matches = matches_barcode_pattern(df.columns)
    row_barcode_matches = matches_barcode_pattern(df.index)
    
    print(f"    Barcode pattern matches - Columns: {col_barcode_matches}/20, Rows: {row_barcode_matches}/20")
    
    if col_barcode_matches > row_barcode_matches and col_barcode_matches >= 5:
        print(f"    → Columns are barcodes, transposing...")
        return df.T
    elif row_barcode_matches > col_barcode_matches and row_barcode_matches >= 5:
        print(f"    → Rows are barcodes (correct orientation)")
        return df
    
    # Heuristic 3: Check for common gene name patterns
    gene_patterns = [
        r'^[A-Z][A-Z0-9]+$',       # Standard gene: TP53, BRCA1
        r'^[A-Z]+\d+$',            # Gene with number: A2M, CD4
        r'^MT-',                    # Mitochondrial genes
        r'^RP[SL]\d+',             # Ribosomal genes
        r'^LINC\d+',               # Long non-coding RNA
    ]
    
    def matches_gene_pattern(names):
        count = 0
        for name in names[:50]:
            for pattern in gene_patterns:
                if re.match(pattern, str(name)):
                    count += 1
                    break
        return count
    
    col_gene_matches = matches_gene_pattern(df.columns)
    row_gene_matches = matches_gene_pattern(df.index)
    
    print(f"    Gene pattern matches - Columns: {col_gene_matches}/50, Rows: {row_gene_matches}/50")
    
    if col_gene_matches > row_gene_matches and col_gene_matches >= 10:
        print(f"    → Columns are genes (correct orientation)")
        return df
    elif row_gene_matches > col_gene_matches and row_gene_matches >= 10:
        print(f"    → Rows are genes, transposing...")
        return df.T
    
    # Heuristic 4: If nothing conclusive, keep original and warn
    print(f"    → Could not determine orientation conclusively, keeping original")
    print(f"    → If results are poor, try manually checking data format")
    
    return df


def preprocess_stereoscope(adata_sc, adata_st, n_hvg=5000):
    """
    Preprocess data for Stereoscope.
    """
    print("\n" + "=" * 60)
    print(f"PREPROCESSING (Top {n_hvg} HVGs)")
    print("=" * 60)
    
    # Filter genes
    sc.pp.filter_genes(adata_sc, min_cells=10)
    
    # Select HVGs
    target_genes = min(n_hvg, adata_sc.shape[1])
    sc.pp.highly_variable_genes(adata_sc, n_top_genes=target_genes, 
                                 flavor="seurat_v3", subset=False)
    hvg_list = adata_sc.var[adata_sc.var['highly_variable']].index
    
    # Find common genes
    common_genes = list(set(hvg_list).intersection(set(adata_st.var_names)))
    print(f"  HVGs selected: {len(hvg_list)}")
    print(f"  Common genes used: {len(common_genes)}")
    
    if len(common_genes) < 100:
        raise ValueError(f"Too few common genes ({len(common_genes)}). Check gene naming.")
    
    # Subset to common genes
    adata_sc = adata_sc[:, common_genes].copy()
    adata_st = adata_st[:, common_genes].copy()
    
    # Ensure integer counts for Negative Binomial model
    from scipy.sparse import issparse
    if issparse(adata_sc.X):
        adata_sc.X = np.array(adata_sc.X.todense())
    if issparse(adata_st.X):
        adata_st.X = np.array(adata_st.X.todense())
    
    adata_sc.X = np.round(adata_sc.X).astype(int)
    adata_st.X = np.round(adata_st.X).astype(int)
    
    print(f"  Final SC shape: {adata_sc.shape}")
    print(f"  Final ST shape: {adata_st.shape}")
    
    return adata_sc, adata_st


# =============================================================================
# MODEL TRAINING
# =============================================================================

def run_stereoscope(adata_sc, adata_st, epochs_sc=100, epochs_st=1000, gpu="0"):
    """
    Train Stereoscope model and get cell type proportions.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Training SC Reference Model")
    print("=" * 60)
    
    # Setup and train single-cell model
    RNAStereoscope.setup_anndata(adata_sc, labels_key="cell_type")
    sc_model = RNAStereoscope(adata_sc)
    
    # Configure accelerator
    accel = "gpu" if gpu != "cpu" else "cpu"
    devices = [int(gpu)] if gpu != "cpu" else "auto"
    
    sc_model.train(max_epochs=epochs_sc, accelerator=accel, devices=devices)
    
    print("\n" + "=" * 60)
    print("STEP 2: Training Spatial Mapping Model")
    print("=" * 60)
    
    # Setup and train spatial model
    SpatialStereoscope.setup_anndata(adata_st)
    st_model = SpatialStereoscope.from_rna_model(adata_st, sc_model)
    st_model.train(max_epochs=epochs_st, accelerator=accel, devices=devices)
    
    print("\nExtracting proportions...")
    proportions_df = st_model.get_proportions()
    proportions_df.index = adata_st.obs_names
    
    return proportions_df


# =============================================================================
# SPATIAL VISUALIZATION FUNCTIONS
# =============================================================================

def rotate_coordinates(x, y, angle_degrees):
    """Rotate coordinates around their centroid."""
    if angle_degrees == 0:
        return x, y
    angle_rad = np.radians(angle_degrees)
    cx, cy = np.mean(x), np.mean(y)
    x_rot = (x - cx) * np.cos(angle_rad) - (y - cy) * np.sin(angle_rad) + cx
    y_rot = (x - cx) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad) + cy
    return x_rot, y_rot


def _calculate_hex_radius(coords: np.ndarray, orientation: int = 0) -> float:
    """
    Calculates the appropriate hexagon radius for perfect tessellation (no gaps).
    
    Args:
        coords: Nx2 array of spot coordinates
        orientation: Hexagon orientation in degrees (30=pointy-top, 0=flat-top)
    """
    if len(coords) < 2:
        return 10.0
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    valid_dists = distances[:, 1][distances[:, 1] > 1e-6]
    
    if len(valid_dists) == 0:
        return 10.0
    
    median_spacing = np.median(valid_dists)
    
    # For perfect tessellation, nearest neighbor distance = sqrt(3) * radius
    radius = median_spacing / np.sqrt(3)
    
    return radius


def _add_background_spots(ax, coords_full, hex_radius, hex_orientation=0):
    """Add grey background hexagons for all tissue spots."""
    if coords_full is None or len(coords_full) == 0:
        return
    patches = [
        RegularPolygon((x, y), numVertices=6, radius=hex_radius, 
                       orientation=np.radians(hex_orientation))
        for x, y in coords_full
    ]
    collection = PatchCollection(patches, facecolors='lightgrey', 
                                  edgecolors='none', alpha=0.3, zorder=0)
    ax.add_collection(collection)


def plot_spatial_intensity_maps(proportions_df, coords_df, output_path, 
                                 coords_full=None, image_rotation=0, hex_orientation=0):
    """Generate spatial intensity maps for each cell type."""
    print(f"Generating spatial intensity maps...")
    import math
    
    if coords_full is None:
        coords_full = coords_df[['x', 'y']].values
    
    # Rotate background coordinates (image rotation)
    bg_x, bg_y = rotate_coordinates(coords_full[:, 0], coords_full[:, 1], image_rotation)
    coords_full_rot = np.column_stack([bg_x, bg_y])
    
    # Calculate radius based on spot orientation logic
    hex_radius = _calculate_hex_radius(coords_full_rot, orientation=hex_orientation)
    
    # Match spots
    common_spots = proportions_df.index.intersection(coords_df['barcode'])
    if len(common_spots) == 0:
        print("  Warning: No common spots found for intensity maps")
        return
    
    coords_matched = coords_df.set_index('barcode').loc[common_spots]
    matched_x, matched_y = rotate_coordinates(
        coords_matched['x'].values, coords_matched['y'].values, image_rotation
    )
    coords_data = np.column_stack([matched_x, matched_y])
    
    cell_types = proportions_df.columns.tolist()
    n_types = len(cell_types)
    cols = min(4, n_types)
    rows = math.ceil(n_types / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_types > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Calculate bounds
    pad = hex_radius * 2
    x_min, x_max = coords_full_rot[:, 0].min(), coords_full_rot[:, 0].max()
    y_min, y_max = coords_full_rot[:, 1].min(), coords_full_rot[:, 1].max()

    for i, ct in enumerate(cell_types):
        ax = axes[i]
        values = proportions_df.loc[common_spots, ct].values
        
        # Add background with hex orientation
        _add_background_spots(ax, coords_full_rot, hex_radius, hex_orientation)
        
        # Normalize values
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.plasma
        
        # Create hexagons with hex orientation
        patches = []
        colors = []
        for j, (x, y) in enumerate(coords_data):
            patches.append(RegularPolygon((x, y), numVertices=6, 
                                          radius=hex_radius, 
                                          orientation=np.radians(hex_orientation)))
            colors.append(cmap(norm(values[j])))
        
        ax.add_collection(PatchCollection(patches, facecolors=colors, 
                                          edgecolors='none', zorder=1))
        
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(ct, fontsize=14, fontweight='bold', pad=10)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.85, aspect=15)
        cbar.locator = ticker.MaxNLocator(nbins=5)
        cbar.formatter = ticker.FormatStrFormatter('%.2f')
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=10)

    # Hide empty subplots
    for i in range(n_types, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_spatial_dominant_type(proportions_df, coords_df, output_path, 
                                coords_full=None, image_rotation=0, hex_orientation=0):
    """Generate spatial map showing dominant cell type per spot."""
    print(f"Generating spatial dominant type map...")
    
    if coords_full is None:
        coords_full = coords_df[['x', 'y']].values
    
    # Rotate coordinates
    bg_x, bg_y = rotate_coordinates(coords_full[:, 0], coords_full[:, 1], image_rotation)
    coords_full_rot = np.column_stack([bg_x, bg_y])
    
    # Calculate radius
    hex_radius = _calculate_hex_radius(coords_full_rot, orientation=hex_orientation)
    
    # Match spots
    common_spots = proportions_df.index.intersection(coords_df['barcode'])
    if len(common_spots) == 0:
        print("  Warning: No common spots found for dominant type map")
        return
    
    coords_matched = coords_df.set_index('barcode').loc[common_spots]
    matched_x, matched_y = rotate_coordinates(
        coords_matched['x'].values, coords_matched['y'].values, image_rotation
    )
    coords_data = np.column_stack([matched_x, matched_y])

    # Get dominant type
    props_matched = proportions_df.loc[common_spots]
    dominant_indices = np.argmax(props_matched.values, axis=1)
    cell_types = props_matched.columns.tolist()
    n_types = len(cell_types)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.colormaps.get_cmap('tab20').resampled(n_types)

    # Background
    _add_background_spots(ax, coords_full_rot, hex_radius, hex_orientation)
    
    # Create hexagons
    patches = []
    colors = []
    for j, (x, y) in enumerate(coords_data):
        type_idx = dominant_indices[j]
        cval = type_idx / max(n_types - 1, 1) if n_types > 1 else 0
        patches.append(RegularPolygon((x, y), numVertices=6, 
                                      radius=hex_radius, 
                                      orientation=np.radians(hex_orientation)))
        colors.append(cmap(cval))

    ax.add_collection(PatchCollection(patches, facecolors=colors, 
                                      edgecolors='none', zorder=1))
    
    # Set bounds
    pad = hex_radius * 2
    x_min, x_max = coords_full_rot[:, 0].min(), coords_full_rot[:, 0].max()
    y_min, y_max = coords_full_rot[:, 1].min(), coords_full_rot[:, 1].max()
    
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    
    # Legend
    handles = []
    for i in range(n_types):
        cval = i / max(n_types - 1, 1) if n_types > 1 else 0
        handles.append(plt.Line2D([0], [0], marker='H', color='w', 
                                  markerfacecolor=cmap(cval),
                                  label=cell_types[i], markersize=12, 
                                  markeredgecolor='none'))
    
    if len(coords_full_rot) > len(coords_data):
        handles.append(plt.Line2D([0], [0], marker='H', color='w', 
                                  markerfacecolor='lightgrey',
                                  alpha=0.5, label='No count data', 
                                  markersize=12, markeredgecolor='none'))

    ax.legend(handles=handles, title="Cell Type", bbox_to_anchor=(1.02, 1), 
              loc='upper left', frameon=True, fontsize=10)
    plt.title("Dominant Cell Type per Spot", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cooccurrence_heatmap(proportions_df, output_path):
    """Generate co-occurrence correlation heatmap."""
    print(f"Generating co-occurrence heatmap...")
    plt.figure(figsize=(10, 8))
    corr = proportions_df.corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, fmt='.2f', square=True, linewidths=.5, annot_kws={'size': 8, 'weight': 'bold'})
    plt.title("Cell Type Co-occurrence (Correlation)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Saved: {output_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    args = parse_arguments()
    
    # Setup Output Directory
    output_dir = os.path.dirname(args.output_csv)
    if not output_dir:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU Configuration
    if args.gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # =========================================================================
    # 1. Load Data (handles both real and simulated)
    # =========================================================================
    adata_sc, adata_st = load_data(
        args.sc_counts, 
        args.sc_labels, 
        args.st_counts,
        args.st_coords  # Can be None for simulated data
    )
    
    # =========================================================================
    # 2. Extract spatial coordinates from adata_st
    # =========================================================================
    spatial_coords = None
    coords_df = None
    coords_full = None
    
    if 'spatial' in adata_st.obsm:
        spatial_coords = adata_st.obsm['spatial']
        coords_df = adata_st.uns.get('spatial_coords_df')
        coords_full = adata_st.uns.get('spatial_coords_full')
        print(f"\nSpatial coordinates available: {spatial_coords.shape}")
    else:
        print("\nWARNING: No spatial coordinates found. Spatial plots will be skipped.")
    
    # =========================================================================
    # 3. Preprocess
    # =========================================================================
    adata_sc, adata_st = preprocess_stereoscope(adata_sc, adata_st, n_hvg=args.n_hvg)
    
    # =========================================================================
    # 4. Run Stereoscope Model
    # =========================================================================
    proportions_df = run_stereoscope(
        adata_sc, adata_st, 
        epochs_sc=args.max_epochs_sc, 
        epochs_st=args.max_epochs_st, 
        gpu=args.gpu
    )
    
    # =========================================================================
    # 5. Save Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Normalize proportions
    proportions_df = proportions_df.div(proportions_df.sum(axis=1), axis=0).fillna(0)
    
    print(f"Saving proportions to {args.output_csv}")
    proportions_df.to_csv(args.output_csv)
    
    # =========================================================================
    # 6. Generate Summary Plot
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY PLOTS")
    print("=" * 60)

    n_spots = len(proportions_df)
    
    if n_spots > 100:
        print(f"  Dataset has {n_spots} spots. Generating summary bar plot.")
        fig, ax = plt.subplots(figsize=(12, 6))
        mean_props = proportions_df.mean().sort_values(ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(mean_props)))
        ax.barh(range(len(mean_props)), mean_props.values, color=colors)
        
        ax.set_yticks(range(len(mean_props)))
        ax.set_yticklabels(mean_props.index)
        ax.set_xlabel('Average Proportion')
        ax.set_title(f'Average Cell Type Proportions (Stereoscope, n={n_spots} spots)')
        plt.tight_layout()
    else:
        plt.figure(figsize=(12, max(8, n_spots * 0.3)))
        sns.heatmap(
            proportions_df, 
            cmap='viridis', 
            yticklabels=True,
            vmin=0, vmax=1
        )
        plt.title("Predicted Cell Type Proportions (Stereoscope)")
        plt.xlabel("Cell Types")
        plt.ylabel("Spots")
        plt.tight_layout()
    
    plt.savefig(args.output_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved proportion plot: {args.output_plot}")
    
    # =========================================================================
    # 7. Spatial Visualizations
    # =========================================================================
    if spatial_coords is not None and coords_df is not None:
        print("\n" + "=" * 60)
        print("GENERATING SPATIAL VISUALIZATIONS")
        print("=" * 60)
        
        # Spatial intensity maps
        intensity_path = os.path.join(output_dir, 'spatial_intensity_maps.png')
        plot_spatial_intensity_maps(
            proportions_df, coords_df, intensity_path, 
            coords_full=coords_full, 
            image_rotation=IMAGE_ROTATION,
            hex_orientation=args.hex_orientation
        )
        
        # Dominant type map
        dominant_path = os.path.join(output_dir, 'spatial_dominant_type.png')
        plot_spatial_dominant_type(
            proportions_df, coords_df, dominant_path, 
            coords_full=coords_full, 
            image_rotation=IMAGE_ROTATION,
            hex_orientation=args.hex_orientation
        )
        
        # Co-occurrence heatmap
        cooccurrence_path = os.path.join(output_dir, 'cooccurrence_heatmap.png')
        plot_cooccurrence_heatmap(proportions_df, cooccurrence_path)
    else:
        # Generate co-occurrence even without spatial
        print("\nGenerating co-occurrence heatmap (no spatial coordinates)...")
        cooccurrence_path = os.path.join(output_dir, 'cooccurrence_heatmap.png')
        plot_cooccurrence_heatmap(proportions_df, cooccurrence_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEREOSCOPE COMPLETE")
    print("=" * 60)
    print(f"  Proportions CSV: {args.output_csv}")
    print(f"  Summary Plot: {args.output_plot}")
    if spatial_coords is not None:
        print(f"  Spatial intensity maps: {os.path.join(output_dir, 'spatial_intensity_maps.png')}")
        print(f"  Dominant type map: {os.path.join(output_dir, 'spatial_dominant_type.png')}")
    print(f"  Co-occurrence heatmap: {os.path.join(output_dir, 'cooccurrence_heatmap.png')}")
    print("=" * 60)


if __name__ == "__main__":
    main()