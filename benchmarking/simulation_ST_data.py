#!/usr/bin/env python3
"""
DestVI Semi-Simulation Data Generation Pipeline

Input:
    - sc_counts.csv: scRNA-seq count matrix (cells x genes)
    - sc_labels.csv: cell type labels for each cell

Output (saved to output_dir):
    - simulated_st_counts.csv: simulated spatial transcriptomics counts (spots x genes)
    - simulated_st_coordinates.csv: spatial coordinates (SpaceRanger format)
    - ground_truth_proportions.csv: true cell type proportions for each spot
    - ground_truth_intensity_raw.csv: raw (unnormalized) cell type intensities
    - ground_truth_intensity_normalized.csv: normalized cell type intensities (sum to 1)
    - ground_truth_cell_states.csv: true cell state values for each spot and cell type
    - ground_truth_expression.csv: true cell-type-specific expression for each spot
    - spatial_ground_truth.png: Visualization of dominant types (Hexagonal)
    - spatial_intensity_maps.png: Visualization of proportion intensity per type (Hexagonal)
    - simulation_config.json: configuration parameters used
"""

import os
import json
import argparse
import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from scipy.special import softmax
from sklearn.decomposition import PCA
from pathlib import Path
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')


class DestVISemiSimulator:
    """
    Semi-simulation framework for generating paired scRNA-seq and ST data
    with known ground truth for benchmarking deconvolution methods.
    """
    
    def __init__(
        self,
        n_spots: int = 1000,
        cells_per_spot_range: tuple = (5, 20),
        n_cell_state_dims: int = 5,
        spatial_smoothness: float = 0.3,
        capture_efficiency: float = 0.1,
        grid_size: tuple = None,
        random_seed: int = 42
    ):
        """
        Initialize the simulator.
        """
        self.n_spots = n_spots
        self.cells_per_spot_range = cells_per_spot_range
        self.n_cell_state_dims = n_cell_state_dims
        self.spatial_smoothness = spatial_smoothness
        self.capture_efficiency = capture_efficiency
        self.random_seed = random_seed
        
        if grid_size is None:
            grid_dim = int(np.ceil(np.sqrt(n_spots)))
            self.grid_size = (grid_dim, grid_dim)
        else:
            self.grid_size = grid_size
            
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Will be populated during simulation
        self.sc_counts = None
        self.sc_labels = None
        self.cell_types = None
        self.n_genes = None
        self.cell_type_profiles = None
        self.cell_state_loadings = None
        self.coordinates = None
        
        # New: intensity ground truth
        self.ground_truth_intensity_raw = None
        self.ground_truth_intensity_normalized = None
        
    def load_sc_data(self, counts_path: str, labels_path: str):
        """Load scRNA-seq reference data."""
        print(f"Loading scRNA-seq counts from {counts_path}...")
        self.sc_counts = pd.read_csv(counts_path, index_col=0)
        
        print(f"Loading cell labels from {labels_path}...")
        self.sc_labels = pd.read_csv(labels_path, index_col=0)
        
        # Ensure alignment
        common_cells = self.sc_counts.index.intersection(self.sc_labels.index)
        self.sc_counts = self.sc_counts.loc[common_cells]
        self.sc_labels = self.sc_labels.loc[common_cells]
        
        # Get cell type column (assume first column or 'cell_type')
        if 'cell_type' in self.sc_labels.columns:
            label_col = 'cell_type'
        elif 'celltype' in self.sc_labels.columns:
            label_col = 'celltype'
        else:
            label_col = self.sc_labels.columns[0]
            
        self.cell_type_labels = self.sc_labels[label_col].values
        self.cell_types = np.unique(self.cell_type_labels)
        self.n_genes = self.sc_counts.shape[1]
        self.gene_names = self.sc_counts.columns.tolist()
        
        print(f"  Loaded {len(common_cells)} cells, {self.n_genes} genes")
        print(f"  Cell types: {self.cell_types}")
        
    def _compute_cell_type_profiles(self):
        """Compute mean expression profile for each cell type."""
        print("Computing cell type expression profiles...")
        profiles = {}
        
        for ct in self.cell_types:
            mask = self.cell_type_labels == ct
            ct_counts = self.sc_counts.values[mask]
            # Normalize to CPM then average
            cpm = ct_counts / (ct_counts.sum(axis=1, keepdims=True) + 1e-6) * 1e6
            profiles[ct] = cpm.mean(axis=0)
            
        self.cell_type_profiles = pd.DataFrame(profiles, index=self.gene_names)
        
    def _compute_cell_state_loadings(self):
        """Compute cell state dimensions using PCA within each cell type."""
        print("Computing cell state loadings via PCA...")
        self.cell_state_loadings = {}
        self.cell_state_ranges = {}
        
        for ct in self.cell_types:
            mask = self.cell_type_labels == ct
            ct_counts = self.sc_counts.values[mask]
            
            # Log-normalize
            ct_norm = np.log1p(ct_counts / (ct_counts.sum(axis=1, keepdims=True) + 1e-6) * 1e4)
            
            # PCA for cell state dimensions
            n_components = min(self.n_cell_state_dims, ct_counts.shape[0] - 1, self.n_genes)
            if n_components < 1:
                # Fallback for very rare cell types
                self.cell_state_loadings[ct] = np.zeros((self.n_genes, self.n_cell_state_dims))
                self.cell_state_ranges[ct] = {'mean': np.zeros(self.n_cell_state_dims), 
                                              'std': np.ones(self.n_cell_state_dims)}
                continue

            pca = PCA(n_components=n_components)
            cell_states = pca.fit_transform(ct_norm)
            
            # Pad if we couldn't get enough components
            comps = pca.components_.T
            if n_components < self.n_cell_state_dims:
                pad = self.n_cell_state_dims - n_components
                comps = np.pad(comps, ((0,0), (0, pad)))
                cell_states = np.pad(cell_states, ((0,0), (0, pad)))

            self.cell_state_loadings[ct] = comps
            self.cell_state_ranges[ct] = {
                'mean': cell_states.mean(axis=0),
                'std': cell_states.std(axis=0),
                'min': cell_states.min(axis=0),
                'max': cell_states.max(axis=0)
            }
    
    def _generate_barcodes(self, n):
        """Generate synthetic DNA barcodes."""
        bases = ['A', 'C', 'G', 'T']
        barcodes = set()
        while len(barcodes) < n:
            bc = "".join(random.choices(bases, k=16)) + "-1"
            barcodes.add(bc)
        return list(barcodes)

    def _generate_spatial_coordinates(self):
        """
        Generate spatial coordinates in SpaceRanger format.
        
        Generates a STAGGERED (Honeycomb) lattice.
        This allows hexagonal spots to fully tile the plane without gaps.
        """
        print("Generating spatial coordinates (Honeycomb Lattice)...")
        rows, cols = self.grid_size
        
        spot_barcodes = self._generate_barcodes(self.n_spots)
        data_rows = []
        
        # Helper lists for distance calculation (internal grid)
        grid_x = []
        grid_y = []
        
        spot_idx = 0
        pixel_scale_factor = 100  # Scale grid to fake pixels
        
        # Geometry constants for hexagonal tiling (Pointy Top Hexagons)
        # Horizontal spacing = 1.0
        # Vertical spacing = sqrt(3)/2 ~= 0.866
        # Odd rows shifted by 0.5
        vertical_scale = np.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if spot_idx >= self.n_spots:
                    break
                
                # Internal Coordinate Logic for Tiling
                # 1. Shift every odd row by 0.5 to create the "brick" or "honeycomb" pattern
                x_offset = 0.5 if i % 2 == 1 else 0.0
                
                x_internal = float(j) + x_offset
                y_internal = float(i) * vertical_scale
                
                grid_x.append(x_internal)
                grid_y.append(y_internal)
                
                # SpaceRanger style coordinates
                # In actual Visium, coordinates are also staggered
                pxl_row = int(y_internal * pixel_scale_factor + 2000)
                pxl_col = int(x_internal * pixel_scale_factor + 2000)
                
                data_rows.append({
                    'barcode': spot_barcodes[spot_idx],
                    'in_tissue': 1,
                    'array_row': i,
                    'array_col': j,
                    'pxl_row_in_fullres': pxl_row,
                    'pxl_col_in_fullres': pxl_col,
                    '_x_internal': x_internal,
                    '_y_internal': y_internal
                })
                spot_idx += 1
                
        self.coordinates = pd.DataFrame(data_rows)
        self.coordinates.set_index('barcode', inplace=True)
        
        # Compute distance matrix for spatial smoothing using the internal grid structure
        coords_internal = self.coordinates[['_x_internal', '_y_internal']].values
        self.distance_matrix = squareform(pdist(coords_internal))
        
    def _generate_spatial_proportions(self):
        """
        Generate spatially smooth cell type proportions with distinct clusters.
        
        This generates:
        1. Raw intensity values (logits transformed through spatial smoothing)
        2. Normalized proportions (softmax of raw intensities)
        """
        print("Generating clustered cell type proportions and intensities...")
        n_types = len(self.cell_types)
        
        # Length scale for spatial smoothing (controls blob size)
        length_scale = self.spatial_smoothness * self.grid_size[0]
        
        # Compute spatial covariance matrix
        K = np.exp(-self.distance_matrix**2 / (2 * length_scale**2 + 1e-6))
        
        # Generate smooth random fields
        L = np.linalg.cholesky(K + 1e-5 * np.eye(self.n_spots))
        logits = np.zeros((self.n_spots, n_types))
        
        for i in range(n_types):
            z = np.random.randn(self.n_spots)
            logits[:, i] = L @ z
        
        # Store raw intensity (before normalization)
        # Transform logits to positive values for "raw intensity" interpretation
        # Using exp to convert log-space values to intensity scale
        raw_intensity = np.exp(logits)
        
        self.ground_truth_intensity_raw = pd.DataFrame(
            raw_intensity,
            index=self.coordinates.index,
            columns=self.cell_types
        )
        
        # Also store the logits as an alternative raw representation
        self.ground_truth_logits = pd.DataFrame(
            logits,
            index=self.coordinates.index,
            columns=self.cell_types
        )
            
        # Apply Softmax with Temperature for normalized proportions
        temperature = 0.5 
        proportions = softmax(logits / temperature, axis=1)
        
        self.ground_truth_proportions = pd.DataFrame(
            proportions,
            index=self.coordinates.index,
            columns=self.cell_types
        )
        
        # Normalized intensity (row-normalized raw intensity)
        row_sums = raw_intensity.sum(axis=1, keepdims=True)
        normalized_intensity = raw_intensity / (row_sums + 1e-10)
        
        self.ground_truth_intensity_normalized = pd.DataFrame(
            normalized_intensity,
            index=self.coordinates.index,
            columns=self.cell_types
        )
        
        # Print summary statistics
        print(f"  Raw intensity range: [{raw_intensity.min():.4f}, {raw_intensity.max():.4f}]")
        print(f"  Raw intensity mean per type: {raw_intensity.mean(axis=0).round(4)}")
        
    def _generate_cell_states(self):
        """Generate cell state values for each spot and cell type."""
        print("Generating cell state values...")
        
        length_scale = self.spatial_smoothness * self.grid_size[0] / 3
        K = np.exp(-self.distance_matrix**2 / (2 * length_scale**2 + 1e-6))
        L = np.linalg.cholesky(K + 1e-5 * np.eye(self.n_spots))
        
        self.ground_truth_cell_states = {}
        
        for ct in self.cell_types:
            n_dims = self.cell_state_loadings[ct].shape[1]
            states = np.zeros((self.n_spots, n_dims))
            
            for d in range(n_dims):
                z = np.random.randn(self.n_spots)
                smooth_field = L @ z
                
                # Scale to observed range
                ct_range = self.cell_state_ranges[ct]
                states[:, d] = ct_range['mean'][d] + smooth_field * ct_range['std'][d]
                
            self.ground_truth_cell_states[ct] = states
            
        # Create combined DataFrame
        state_columns = []
        state_data = []
        for ct in self.cell_types:
            n_dims = self.ground_truth_cell_states[ct].shape[1]
            for d in range(n_dims):
                state_columns.append(f"{ct}_dim{d}")
                state_data.append(self.ground_truth_cell_states[ct][:, d])
                
        self.ground_truth_cell_states_df = pd.DataFrame(
            np.column_stack(state_data),
            index=self.coordinates.index,
            columns=state_columns
        )
        
    def _generate_expression(self):
        """Generate cell-type-specific expression for each spot."""
        print("Generating cell-type-specific expression...")
        
        self.ground_truth_expression = {}
        
        for ct in self.cell_types:
            base_expr = self.cell_type_profiles[ct].values
            loadings = self.cell_state_loadings[ct]
            states = self.ground_truth_cell_states[ct]
            
            # Expression = base + state modulation
            state_effect = states @ loadings.T
            
            # Combine
            expr = np.exp(np.log(base_expr + 1) + 0.1 * state_effect) - 1
            expr = np.maximum(expr, 0)
            
            self.ground_truth_expression[ct] = expr
            
    def _generate_st_counts(self):
        """Generate observed ST counts."""
        print("Generating ST counts...")
        
        n_cells_per_spot = np.random.randint(
            self.cells_per_spot_range[0],
            self.cells_per_spot_range[1] + 1,
            size=self.n_spots
        )
        
        mixed_expression = np.zeros((self.n_spots, self.n_genes))
        
        for i, ct in enumerate(self.cell_types):
            props = self.ground_truth_proportions[ct].values[:, np.newaxis]
            expr = self.ground_truth_expression[ct]
            mixed_expression += props * expr
            
        total_counts_target = n_cells_per_spot[:, np.newaxis] * mixed_expression
        total_counts_target *= self.capture_efficiency
        
        # Negative Binomial sampling
        overdispersion = 10
        p = overdispersion / (overdispersion + total_counts_target + 1e-6)
        
        # Approximate sampling (faster loop)
        counts = np.zeros_like(total_counts_target)
        for i in range(self.n_spots):
             counts[i, :] = np.random.negative_binomial(overdispersion, p[i, :])
                    
        self.st_counts = pd.DataFrame(
            counts.astype(int),
            index=self.coordinates.index,
            columns=self.gene_names
        )
        
        self.cells_per_spot = pd.Series(
            n_cells_per_spot,
            index=self.coordinates.index,
            name='n_cells'
        )
        
        print(f"  Total counts per spot: {self.st_counts.sum(axis=1).mean():.0f} (mean)")
        print(f"  Genes detected per spot: {(self.st_counts > 0).sum(axis=1).mean():.0f} (mean)")
        
    def simulate(self, counts_path: str, labels_path: str):
        """Run the full simulation pipeline."""
        self.load_sc_data(counts_path, labels_path)
        self._compute_cell_type_profiles()
        self._compute_cell_state_loadings()
        self._generate_spatial_coordinates()
        self._generate_spatial_proportions()
        self._generate_cell_states()
        self._generate_expression()
        self._generate_st_counts()
        
        print("\nSimulation complete!")
        print(f"  Generated {self.n_spots} spots")
        print(f"  {len(self.cell_types)} cell types")
        print(f"  {self.n_genes} genes")
        
        return self.get_results()
    
    def get_results(self):
        """Return all simulation results as a dictionary."""
        return {
            'st_counts': self.st_counts,
            'coordinates': self.coordinates,
            'ground_truth_proportions': self.ground_truth_proportions,
            'ground_truth_intensity_raw': self.ground_truth_intensity_raw,
            'ground_truth_intensity_normalized': self.ground_truth_intensity_normalized,
            'ground_truth_logits': self.ground_truth_logits,
            'ground_truth_cell_states': self.ground_truth_cell_states_df,
            'cells_per_spot': self.cells_per_spot,
            'cell_type_profiles': self.cell_type_profiles,
            'gene_names': self.gene_names,
            'cell_types': self.cell_types.tolist()
        }

    def _get_hex_radius(self, coords):
        """
        Helper to calculate the specific radius for touching hexagons
        based on the grid logic used in this simulator.
        """
        # 1. Find nearest neighbor distance (Center-to-Center)
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=2)
        nn_distances = distances[:, 1]
        
        # Use median to be robust
        center_to_center_dist = np.median(nn_distances)
        
        # 2. Calculate Radius for Tiling (Pointy Top)
        # Radius = Distance / sqrt(3)
        hex_radius = center_to_center_dist / np.sqrt(3)
        
        # Slightly bump radius to prevent floating point hairline gaps (overlap by 1%)
        return hex_radius * 1.01

    def plot_ground_truth(self, output_path):
        """
        Generate a HEXAGONAL spatial plot (Dominant Type).
        """
        print("Generating spatial ground truth image (Touching Hexagons)...")
        
        hex_angle = np.radians(0) 
        
        dominant_types = self.ground_truth_proportions.idxmax(axis=1)
        coords = self.coordinates[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
        cell_types = self.cell_types
        n_types = len(cell_types)
        
        hex_radius = self._get_hex_radius(coords)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Colormap
        cmap = plt.get_cmap('tab20')
        if n_types > 20:
             cmap = plt.get_cmap('nipy_spectral')
        
        type_to_color = {ct: cmap(i / max(n_types - 1, 1)) for i, ct in enumerate(cell_types)}
        
        patches = []
        colors = []
        
        for idx, barcode in enumerate(self.coordinates.index):
            x, y = coords[idx]
            dom_type = dominant_types.loc[barcode]
            color = type_to_color[dom_type]
            
            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_radius,
                orientation=hex_angle
            )
            patches.append(hexagon)
            colors.append(color)
            
        collection = PatchCollection(patches, facecolors=colors, edgecolors='none', linewidths=0)
        ax.add_collection(collection)
        
        # Setup Axes
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        pad = hex_radius * 2
        
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect('equal')
        ax.invert_yaxis() 
        ax.axis('off')
        
        # Legend
        handles = [
            Line2D([0], [0], marker='H', color='w', 
                   markerfacecolor=type_to_color[ct],
                   label=ct, markersize=12, markeredgecolor='none')
            for ct in cell_types
        ]
        
        ax.legend(handles=handles, title="Dominant Cell Type", 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title('Ground Truth: Dominant Cell Type Clusters')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved image: {os.path.basename(output_path)}")

    def plot_intensity_maps(self, output_path, use_raw=False):
        """
        Generate spatial maps with HEXAGONAL spot markers:
        A grid of plots, one per cell type, showing proportion/intensity.
        
        Args:
            output_path: Path to save the figure
            use_raw: If True, use raw intensity values; if False, use normalized proportions
        """
        data_type = "raw intensity" if use_raw else "normalized proportions"
        print(f"Generating spatial intensity maps ({data_type})...")
        
        coords = self.coordinates[['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
        cell_types = self.cell_types
        n_types = len(cell_types)
        
        # Select data source
        if use_raw:
            df_props = self.ground_truth_intensity_raw
        else:
            df_props = self.ground_truth_proportions
        
        # Calculate radius (exact same logic as ground truth plot for consistency)
        hex_radius = self._get_hex_radius(coords)
        hex_angle = np.radians(0)
        
        # Calculate limits
        x_min = coords[:, 0].min() - hex_radius * 2
        x_max = coords[:, 0].max() + hex_radius * 2
        y_min = coords[:, 1].min() - hex_radius * 2
        y_max = coords[:, 1].max() + hex_radius * 2
        
        # Grid setup
        cols = 4
        rows = math.ceil(n_types / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
        axes = axes.flatten() if n_types > 1 else [axes]
        
        for i, ct in enumerate(cell_types):
            ax = axes[i]
            values = df_props[ct].values
            
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
                    orientation=hex_angle
                )
                patches.append(hexagon)
                colors.append(cmap(norm(values[j])))
            
            collection = PatchCollection(patches, facecolors=colors, edgecolors='none', linewidths=0)
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
        
        title_suffix = "(Raw Intensity)" if use_raw else "(Normalized Proportions)"
        plt.suptitle(f'Ground Truth Cell Type Distribution {title_suffix}', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved image: {os.path.basename(output_path)}")

    def plot_intensity_comparison(self, output_path):
        """
        Generate a comparison plot showing raw vs normalized intensity distributions.
        """
        print("Generating intensity comparison plots...")
        
        n_types = len(self.cell_types)
        
        fig, axes = plt.subplots(2, n_types, figsize=(4 * n_types, 8))
        if n_types == 1:
            axes = axes.reshape(2, 1)
        
        for i, ct in enumerate(self.cell_types):
            # Raw intensity histogram
            ax1 = axes[0, i]
            raw_vals = self.ground_truth_intensity_raw[ct].values
            ax1.hist(raw_vals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax1.set_title(f'{ct}\n(Raw)', fontsize=12)
            ax1.set_xlabel('Intensity')
            ax1.set_ylabel('Frequency')
            ax1.axvline(raw_vals.mean(), color='red', linestyle='--', label=f'Mean: {raw_vals.mean():.2f}')
            ax1.legend(fontsize=8)
            
            # Normalized intensity histogram
            ax2 = axes[1, i]
            norm_vals = self.ground_truth_intensity_normalized[ct].values
            ax2.hist(norm_vals, bins=30, color='darkorange', edgecolor='black', alpha=0.7)
            ax2.set_title(f'{ct}\n(Normalized)', fontsize=12)
            ax2.set_xlabel('Proportion')
            ax2.set_ylabel('Frequency')
            ax2.axvline(norm_vals.mean(), color='red', linestyle='--', label=f'Mean: {norm_vals.mean():.2f}')
            ax2.legend(fontsize=8)
        
        plt.suptitle('Ground Truth Intensity Distributions: Raw vs Normalized', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved image: {os.path.basename(output_path)}")

    def save_results(self, output_dir: str):
        """Save all results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}/")
        
        # Save ST counts
        self.st_counts.index.name = None
        self.st_counts.to_csv(output_path / "simulated_st_counts.csv")
        print(f"  Saved: simulated_st_counts.csv")
        
        # Save coordinates in SpaceRanger format
        coord_export = self.coordinates[['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']]
        coord_export.to_csv(output_path / "simulated_st_coordinates.csv", header=False)
        print(f"  Saved: simulated_st_coordinates.csv")
        
        # Save ground truth proportions (softmax normalized)
        self.ground_truth_proportions.to_csv(output_path / "ground_truth_proportions.csv")
        print(f"  Saved: ground_truth_proportions.csv")
        
        # Save raw intensity (NEW)
        self.ground_truth_intensity_raw.to_csv(output_path / "ground_truth_intensity_raw.csv")
        print(f"  Saved: ground_truth_intensity_raw.csv")
        
        # Save normalized intensity (NEW)
        self.ground_truth_intensity_normalized.to_csv(output_path / "ground_truth_intensity_normalized.csv")
        print(f"  Saved: ground_truth_intensity_normalized.csv")
        
        # Save logits (pre-softmax values) (NEW)
        self.ground_truth_logits.to_csv(output_path / "ground_truth_logits.csv")
        print(f"  Saved: ground_truth_logits.csv")
        
        # Save ground truth cell states
        self.ground_truth_cell_states_df.to_csv(output_path / "ground_truth_cell_states.csv")
        print(f"  Saved: ground_truth_cell_states.csv")
        
        # Save cells per spot
        self.cells_per_spot.to_csv(output_path / "cells_per_spot.csv")
        print(f"  Saved: cells_per_spot.csv")
        
        # Save cell type profiles
        self.cell_type_profiles.to_csv(output_path / "cell_type_profiles.csv")
        print(f"  Saved: cell_type_profiles.csv")
        
        # Save ground truth expression
        expr_dir = output_path / "ground_truth_expression"
        expr_dir.mkdir(exist_ok=True)
        for ct in self.cell_types:
            expr_df = pd.DataFrame(
                self.ground_truth_expression[ct],
                index=self.coordinates.index,
                columns=self.gene_names
            )
            expr_df.to_csv(expr_dir / f"{ct}_expression.csv")
        print(f"  Saved: ground_truth_expression/ (per cell type)")
            
        # Save configuration
        config = {
            'n_spots': self.n_spots,
            'n_genes': self.n_genes,
            'n_cell_types': len(self.cell_types),
            'cell_types': self.cell_types.tolist(),
            'cells_per_spot_range': list(self.cells_per_spot_range),
            'n_cell_state_dims': self.n_cell_state_dims,
            'spatial_smoothness': self.spatial_smoothness,
            'capture_efficiency': self.capture_efficiency,
            'grid_size': list(self.grid_size),
            'random_seed': self.random_seed,
            'output_files': {
                'st_counts': 'simulated_st_counts.csv',
                'coordinates': 'simulated_st_coordinates.csv',
                'proportions': 'ground_truth_proportions.csv',
                'intensity_raw': 'ground_truth_intensity_raw.csv',
                'intensity_normalized': 'ground_truth_intensity_normalized.csv',
                'logits': 'ground_truth_logits.csv',
                'cell_states': 'ground_truth_cell_states.csv',
                'cells_per_spot': 'cells_per_spot.csv',
                'cell_type_profiles': 'cell_type_profiles.csv'
            }
        }
        with open(output_path / "simulation_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  Saved: simulation_config.json")

        # Generate and save visualizations
        self.plot_ground_truth(output_path / "spatial_ground_truth.png")
        self.plot_intensity_maps(output_path / "spatial_intensity_maps.png", use_raw=False)
        self.plot_intensity_maps(output_path / "spatial_intensity_maps_raw.png", use_raw=True)
        self.plot_intensity_comparison(output_path / "intensity_distribution_comparison.png")
        
        # Create summary statistics file
        summary = {
            'Total Spots': self.n_spots,
            'Total Genes': self.n_genes,
            'Cell Types': len(self.cell_types),
            'Mean Cells per Spot': float(self.cells_per_spot.mean()),
            'Mean Counts per Spot': float(self.st_counts.sum(axis=1).mean()),
            'Mean Genes Detected per Spot': float((self.st_counts > 0).sum(axis=1).mean()),
        }
        
        # Add per-cell-type statistics
        for ct in self.cell_types:
            summary[f'{ct}_mean_proportion'] = float(self.ground_truth_proportions[ct].mean())
            summary[f'{ct}_mean_raw_intensity'] = float(self.ground_truth_intensity_raw[ct].mean())
            summary[f'{ct}_std_raw_intensity'] = float(self.ground_truth_intensity_raw[ct].std())
        
        with open(output_path / "simulation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: simulation_summary.json")
        
        print(f"\nAll outputs saved to: {output_dir}/")
        print(f"\n{'='*60}")
        print("OUTPUT FILES SUMMARY:")
        print(f"{'='*60}")
        print(f"  CSV Files:")
        print(f"    - simulated_st_counts.csv: ST count matrix ({self.n_spots} x {self.n_genes})")
        print(f"    - simulated_st_coordinates.csv: Spatial coordinates (SpaceRanger format)")
        print(f"    - ground_truth_proportions.csv: Softmax-normalized proportions")
        print(f"    - ground_truth_intensity_raw.csv: Raw (unnormalized) intensities")
        print(f"    - ground_truth_intensity_normalized.csv: Row-normalized intensities")
        print(f"    - ground_truth_logits.csv: Pre-softmax logit values")
        print(f"    - ground_truth_cell_states.csv: Cell state dimensions per type")
        print(f"  Images:")
        print(f"    - spatial_ground_truth.png: Dominant cell type map")
        print(f"    - spatial_intensity_maps.png: Per-type normalized intensity maps")
        print(f"    - spatial_intensity_maps_raw.png: Per-type raw intensity maps")
        print(f"    - intensity_distribution_comparison.png: Raw vs normalized histograms")
        print(f"{'='*60}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DestVI Semi-Simulation: Generate simulated ST data with ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Files:
  Ground Truth CSVs:
    - ground_truth_proportions.csv      Softmax-normalized cell type proportions (sum=1)
    - ground_truth_intensity_raw.csv    Raw intensity values (unnormalized)
    - ground_truth_intensity_normalized.csv  Row-normalized intensity values
    - ground_truth_logits.csv           Pre-transformation logit values
    
  The difference between these files:
    - proportions: Uses softmax with temperature, creating sharper boundaries
    - intensity_raw: Exponential of spatial random field (always positive)
    - intensity_normalized: Raw intensity divided by row sum
    - logits: The underlying spatial random field values

Example:
  python simulate_st.py -c sc_counts.csv -l sc_labels.csv -o ./simulation_output
        """
    )
    
    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--counts', type=str, required=True, help='Path to scRNA-seq counts CSV')
    required.add_argument('-l', '--labels', type=str, required=True, help='Path to cell type labels CSV')
    required.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory')
    
    # Optional arguments
    parser.add_argument('--n_spots', type=int, default=1000, help='Number of spots (default: 1000)')
    parser.add_argument('--min_cells', type=int, default=5, help='Min cells per spot (default: 5)')
    parser.add_argument('--max_cells', type=int, default=20, help='Max cells per spot (default: 20)')
    parser.add_argument('--n_state_dims', type=int, default=5, help='Cell state dimensions (default: 5)')
    parser.add_argument('--spatial_smoothness', type=float, default=0.3, help='Spatial smoothness 0-1 (default: 0.3)')
    parser.add_argument('--capture_efficiency', type=float, default=0.1, help='Capture efficiency (default: 0.1)')
    parser.add_argument('--grid_rows', type=int, default=None, help='Grid rows (auto if not specified)')
    parser.add_argument('--grid_cols', type=int, default=None, help='Grid columns (auto if not specified)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("DestVI Semi-Simulation Data Generation")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Input counts: {args.counts}")
    print(f"  Input labels: {args.labels}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of spots: {args.n_spots}")
    print(f"  Cells per spot: {args.min_cells}-{args.max_cells}")
    print(f"  Spatial smoothness: {args.spatial_smoothness}")
    print(f"  Random seed: {args.seed}")
    print()
    
    grid_size = None
    if args.grid_rows is not None and args.grid_cols is not None:
        grid_size = (args.grid_rows, args.grid_cols)
    
    simulator = DestVISemiSimulator(
        n_spots=args.n_spots,
        cells_per_spot_range=(args.min_cells, args.max_cells),
        n_cell_state_dims=args.n_state_dims,
        spatial_smoothness=args.spatial_smoothness,
        capture_efficiency=args.capture_efficiency,
        grid_size=grid_size,
        random_seed=args.seed
    )
    
    simulator.simulate(counts_path=args.counts, labels_path=args.labels)
    simulator.save_results(args.output_dir)
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()