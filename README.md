# In ARCADE/README.md

# 01a_ARCADE_ref_optimizer: scRNA-seq Bayesian Optimization and Analysis


**01a_ARCADE_ref_optimizer** is an integrated, two-stage computational pipeline for single-cell RNA sequencing (scRNA-seq) analysis. It automates the discovery of optimal processing parameters using Bayesian Optimization (Stage 1) and then applies these parameters to a comprehensive downstream analysis workflow (Stage 2). The pipeline also features an optional multi-level refinement process (Stage 3/4) to iteratively re-analyze and improve annotations for low-confidence cell clusters.

## Key Features

-   **Automated Parameter Tuning**: Uses Bayesian Optimization to find the best parameters (`n_highly_variable_genes`, `n_pcs`, `n_neighbors`, `resolution`) for clustering and cell type annotation.
-   **Multi-Metric Objective Function**: Optimizes for a balanced score that considers annotation accuracy (CAS), marker gene specificity (MCS), Marker Prior Score (MPS), and cluster separation (Silhouette score).
-   **Marker Prior Score (MPS)**: Integrates external canonical marker databases to calculate F1-based scores validating cluster markers against known cell type signatures.
-   **Single & Multi-Sample Modes**: Natively supports analysis of a single dataset or the integration of two datasets (e.g., control vs. treated) using Harmony.
-   **Automatic Batch Detection**: Intelligently detects batch information from barcode suffixes or existing metadata columns.
-   **Iterative Refinement**: Automatically identifies low-confidence cell clusters and re-runs the entire optimization and analysis pipeline on them to improve annotation granularity and accuracy.
-   **Consistent Cell Export**: Exports high-confidence cells where multiple annotation methods agree, with optional deconvolution reference files for spatial transcriptomics integration.
-   **Comprehensive Outputs**: Generates publication-quality plots, detailed metric reports, annotated data objects (`.h5ad`), and summary tables for easy interpretation.

---

## Repository Structure
```text
ARCADE/
├── .gitignore                      # Specifies files for Git to ignore
├── LICENSE                         # Project license (e.g., MIT)
├── README.md                       # This documentation file
├── requirements.txt                # Exact Python dependencies for reproducibility
└── 01a_ARCADE_ref_optimizer.py     # The main executable Python script
```

---

## Step-by-Step Workflow

### 1. Prerequisites

-   Git installed on your system.
-   Python 3.8 or newer.
-   Access to a Linux-based command line.

### 2. Clone the Repository

```bash
git clone https://github.com/QiangSu/ARCADE.git
cd ARCADE
```

### 3. Set Up a Python Environment (Recommended)

Using a virtual environment prevents conflicts with other Python projects.

```bash
# Create a new conda environment with Python 3.9
conda create -n ARCADE_env python=3.9

# Activate the environment
conda activate ARCADE_env

```

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# To deactivate later, simply run: deactivate
```

### 4. Install Dependencies

The `requirements.txt` file contains the exact library versions for perfect reproducibility.

```bash
pip install -r requirements.txt
```

### 5. Prepare Your Data

-   **scRNA-seq Data**: Ensure your Cell Ranger output (the folder containing barcodes.tsv.gz, features.tsv.gz, and matrix.mtx.gz) is accessible. The pipeline also accepts .h5 or .h5ad files.
-   **CellTypist Model**: Download a pre-trained CellTypist model (.pkl file). You can find available models on the official CellTypist models website.
-   **Marker Prior Database (Optional)**: A CSV file containing canonical marker genes for cell types. Expected columns: species, organ, cell_type, marker_genes (semicolon-separated), gene_count.

### 6. Run the Pipeline

Here is an example command for a single-sample analysis:

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./my_analysis_output/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --cas_aggregation_method leiden \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 
```

Single-sample refinement analysis

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --cas_aggregation_method leiden \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 \
  --cas_refine_threshold 50 \
  --min_cells_refinement 50 \
  --refinement_depth 3
```

Single-sample analysis with Marker Prior Score (MPS)

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --marker_prior_db ./markers/combined_markers_summary.csv \
  --marker_prior_species Human \
  --mps_bonus_weight 0.2 \
  --n_degs_for_mps 50
```

Analysis with Spatial Transcriptomics integration (for deconvolution):

```bash
python scBOA.py \
  --data_dir /path/to/your/cellranger_output/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Healthy_COVID19_PBMC.pkl \
  --output_prefix sample \
  --st_data_dir /path/to/spatial_data/ \
  --min_cells_per_type 50
```

Multiple-sample refinement analysis:

```bash
python scBOA.py \
  --multi_sample ./WT_CellRanger/ ./treated_CellRanger/ \
  --output_dir ./my_analysis_output/ \
  --model_path ./reference/Mouse_Whole_Brain.pkl \
  --output_prefix WTTR \
  --seed 42 \
  --n_calls 50 \
  --target all \
  --model_type biological \
  --cas_aggregation_method leiden \
  --hvg_min_mean 0.0125 \
  --hvg_max_mean 3.0 \
  --hvg_min_disp 0.3 \
  --cas_refine_threshold 50 \
  --min_cells_refinement 50 \
  --refinement_depth 3
```
---

## Command-Line Arguments Explained

### Stage 1 & 2: Main I/O and Mode

| Argument | Description | Explanation/Usage |
|----------|-------------|-------------------|
| `--data_dir <path>` | Path to expression data. | **(Single-Sample Mode)** Path to 10x directory, `.h5` file, or `.h5ad` file. |
| `--multi_sample <path1> <path2>` | Two paths for WT and Treated data. | **(Multi-Sample Mode)** First path for control/WT, second for treated/perturbed. Enables Harmony integration. |
| `--output_dir <path>` | Path for all output files. | Main directory for results. Subdirectories for each stage are created automatically. |
| `--model_path <path>` | Path to CellTypist model (`.pkl`). | **Required.** Pre-trained model for cell type annotation. |
| `--output_prefix <str>` | Base prefix for Stage 1 output files. | Default: `bayesian_opt`. |
| `--st_data_dir <path>` | Path to Spatial Transcriptomics data. | **(Optional)** For gene intersection and deconvolution reference export. |

### Stage 1 & 2: Batch Integration Options

| Argument | Description | Explanation/Usage |
|----------|-------------|-------------------|
| `--batch_key <str>` | Column name for batch information. | If not specified, auto-detects from metadata or barcode suffixes. |
| `--no_integration` | Force single-sample mode. | Skips Harmony integration even if batches are detected. |
| `--integration_method <choice>` | Integration method. | `harmony` (default) or `none`. |

### Stage 1: Optimization Parameters

| Argument | Description | Explanation/Usage |
|----------|-------------|-------------------|
| `--seed <int>` | Global random seed. | Default: `42`. Ensures reproducibility. |
| `--n_calls <int>` | Trials per optimization strategy. | Default: `50`. Three strategies run, so total is 150 trials. |
| `--model_type <choice>` | Objective function type. | `biological` (default): CAS & MCS. `structural`: adds Silhouette. `silhouette`: Silhouette only. |
| `--marker_gene_model <choice>` | Genes for MCS calculation. | `non-mitochondrial` (default) or `all`. |
| `--target <choice>` | Optimization target. | `all` (default): balanced. Or `weighted_cas`, `simple_cas`, `mcs`. |
| `--cas_aggregation_method <choice>` | CAS calculation method. | `leiden` (default): per-cluster. `consensus`: per-cell-type. |

### Stage 1 & 2: HVG Selection Method

| Argument | Description | Explanation/Usage |
|----------|-------------|-------------------|
| `--hvg_min_mean <float>` | Min mean for two-step HVG. | Activates pre-filtering if set with other HVG params. |
| `--hvg_max_mean <float>` | Max mean for two-step HVG. | See above. |
| `--hvg_min_disp <float>` | Min dispersion for two-step HVG. | See above. |

### Stage 1 & 2: QC & Filtering Parameters

| Argument | Description | Explanation/Usage |
|----------|-------------|-------------------|
| `--min_genes <int>` | Min genes per cell. | Default: `200`. Filters low-quality cells. |
| `--max_genes <int>` | Max genes per cell. | Default: `7000`. Filters potential doublets. |
| `--max_pct_mt <float>` | Max mitochondrial percentage. | Default: `10.0`. Filters dying cells. |
| `--min_cells <int>` | Min cells per gene. | Default: `3`. Filters negligible genes. |

### Stage 2 & Optional Refinement: Final Run Parameters

| Argument | Description | Explanation/Usage |
|----------|-------------|-------------------|
| `--final_run_prefix <str>` | Prefix for Stage 2 outputs. | Default: `sc_analysis_repro`. |
| `--fig_dpi <int>` | Figure resolution. | Default: `500`. |
| `--n_pcs_compute <int>` | PCs to compute. | Default: `105`. |
| `--n_top_genes <int>` | Top markers to show. | Default: `5`. |
| `--cellmarker_db <path>` | Cell marker database CSV. | **(Optional)** For manual-style annotation. |
| `--n_degs_for_capture <int>` | DEGs for Marker Capture Score. | Default: `5`. |
| `--cas_refine_threshold <float>` | CAS threshold for refinement. | **(Optional)** Triggers re-analysis of low-confidence clusters. |
| `--refinement_depth <int>` | Max refinement iterations. | Default: `1`. |
| `--min_cells_refinement <int>` | Min cells for refinement. | Default: `100`. |
| `--min_cells_per_type <int>` | Min cells per type for export. | **(Optional)** Filters small populations from consistent cell export. |

### Marker Prior Score (MPS) Options

| Argument | Description | Explanation/Usage |
|----------|-------------|-------------------|
| `--marker_prior_db <path>` | External marker database CSV. | Expected columns: `species`, `organ`, `cell_type`, `marker_genes`, `gene_count`. |
| `--marker_prior_species <str>` | Species filter. | Default: `Human`. |
| `--marker_prior_organ <str>` | Organ filter. | **(Optional)** E.g., `Adipose`, `Brain`. |
| `--mps_bonus_weight <float>` | MPS bonus weight. | Default: `0.2`. Maximum bonus MPS adds to base score. |
| `--n_degs_for_mps <int>` | DEGs for MPS calculation. | Default: `50`. |
| `--protect_canonical_markers` | Protect markers in HVG. | Ensures canonical markers are included in HVG selection. |
| `--penalize_unmatched_clusters` | Penalize unmatched clusters. | Default: `True`. Unmatched clusters receive MPS=0. |
| `--deg_ranking_method <choice>` | DEG ranking method. | `original` (default): log2FC only. `composite`: weighted combination. |
| `--deg_weight_fc <float>` | Weight for log2FC. | Default: `0.4`. For composite ranking. |
| `--deg_weight_expr <float>` | Weight for expression. | Default: `0.3`. For composite ranking. |
| `--deg_weight_pct <float>` | Weight for pct difference. | Default: `0.3`. For composite ranking. |
| `--mps_similarity_threshold <float>` | Fuzzy matching threshold. | Default: `0.6`. Minimum similarity for cell type matching. |
| `--mps_verbose_matching` | Verbose matching output. | Prints detailed matching info for debugging. |
| `--mps_min_cells_per_group <int>` | Min cells for MPS. | Default: `5`. Clusters below this get MPS=0. |

---

## Output Directory Structure

The script generates a structured output directory. Below is an example structure and an explanation of key files.

```
<output_dir>/
├── stage_1_bayesian_optimization/
│   ├── bayesian_opt_biological_balanced_FINAL_annotated.h5ad
│   ├── bayesian_opt_biological_balanced_FINAL_best_params.txt
│   ├── bayesian_opt_biological_balanced_yield_scores_report.csv
│   ├── bayesian_opt_biological_balanced_optimizer_convergence.png
│   ├── bayesian_opt_biological_balanced_BO-EI_opt_result.skopt
│   ├── ... (other plots and strategy files) ...
│   └── refinement_depth_1/
│       └── ... (mirrors structure for refined cells) ...
│
├── stage_2_final_analysis/
│   ├── sc_analysis_repro_final_processed.h5ad
│   ├── sc_analysis_repro_final_processed_with_refinement.h5ad
│   ├── sc_analysis_repro_all_annotations.csv
│   ├── sc_analysis_repro_all_annotations_with_refinement.csv
│   ├── sc_analysis_repro_FINAL_refined_annotations.csv
│   ├── sc_analysis_repro_leiden_cluster_annotation_scores.csv
│   ├── sc_analysis_repro_consensus_group_annotation_scores.csv
│   ├── sc_analysis_repro_combined_cluster_annotation_scores.csv
│   ├── sc_analysis_repro_cell_type_journey_summary.csv
│   ├── sc_analysis_repro_celltype_matching_diagnostics.csv
│   ├── sc_analysis_repro_umap_leiden.png
│   ├── sc_analysis_repro_cluster_celltypist_umap.png
│   ├── sc_analysis_repro_umap_low_confidence_greyed.png
│   ├── sc_analysis_repro_FINAL_refined_annotation_umap.png
│   ├── sc_analysis_repro_refinement_before_after_comparison.png
│   ├── sc_analysis_repro_cells_changed_by_refinement_umap.png
│   ├── celltype_marker_details/
│   │   ├── sc_analysis_repro_celltype_top_markers.csv
│   │   ├── sc_analysis_repro_celltype_matching_summary.csv
│   │   ├── sc_analysis_repro_celltype_canonical_overlap.csv
│   │   └── sc_analysis_repro_celltype_hvg_genes.csv
│   └── refinement_depth_1/
│       └── ... (mirrors Stage 2 for refined subset) ...
│
└── consistent_cells_subset/
    ├── sc_analysis_repro_consistent_cells_*.csv
    ├── sc_analysis_repro_consistent_cells_*_umap.png
    ├── sc_analysis_repro_consistency_context_all_cells_umap.png
    ├── sc_counts.csv                    # For deconvolution
    ├── sc_labels.csv                    # For deconvolution
    ├── st_counts.csv                    # If --st_data_dir provided
    ├── refined_annotation_exports/      # If refinement was run
    │   ├── sc_analysis_repro_all_cells_REFINED_annotation_umap.png
    │   ├── sc_analysis_repro_REFINED_inconsistent_cells_grey_umap.png
    │   ├── sc_analysis_repro_all_cells_REFINED_annotations.csv
    │   ├── sc_analysis_repro_REFINED_cell_type_counts.csv
    │   └── deconvolution_reference_REFINED/
    │       ├── sc_counts.csv
    │       ├── sc_labels.csv
    │       ├── sc_labels_REFINED_with_comparison.csv
    │       └── REFINED_deconvolution_summary.txt
    └── ... (additional filtered exports if --min_cells_per_type) ...
```

## Key File Explanations

### Stage 1: `stage_1_bayesian_optimization/`

| File | Description |
|------|-------------|
| `*_FINAL_best_params.txt` | Summary of optimal parameters and final metrics. **Most important summary file.** |
| `*_FINAL_annotated.h5ad` | AnnData processed with best parameters, containing all annotations. |
| `*_yield_scores_report.csv` | Detailed log of all trials with parameters and scores (CAS, MCS, MPS, Silhouette). |
| `*_optimizer_convergence.png` | Plot showing score improvement over time for each strategy. |
| `*_opt_result.skopt` | Saved optimization state for reloading. |

### Stage 2: `stage_2_final_analysis/`

| File | Description |
|------|-------------|
| `*_final_processed.h5ad` | Final annotated AnnData from initial Stage 2 run. |
| `*_final_processed_with_refinement.h5ad` | Master AnnData with combined annotations after all refinement. |
| `*_FINAL_refined_annotations.csv` | Comprehensive cell-by-cell annotations with refinement status. |
| `*_cluster_annotation_scores.csv` | CAS scores for Leiden clusters and consensus groups. |
| `*_combined_cluster_annotation_scores.csv` | Concatenated CAS from all refinement levels. |
| `*_cell_type_journey_summary.csv` | Cell count and CAS changes across refinement stages. |
| `*_celltype_matching_diagnostics.csv` | Detailed matching between annotations and marker database. |
| `*_FINAL_refined_annotation_umap.png` | Primary output: UMAP with final refined annotations. |
| `*_refinement_before_after_comparison.png` | Side-by-side comparison of before/after refinement. |
| `*_cells_changed_by_refinement_umap.png` | UMAP highlighting cells that changed during refinement. |
| `celltype_marker_details/` | Detailed marker gene information per cell type. |

### Consistent Cells: `consistent_cells_subset/`

| File | Description |
|------|-------------|
| `sc_counts.csv` | Expression matrix for consistent cells (deconvolution input). |
| `sc_labels.csv` | Cell type labels for consistent cells (deconvolution input). |
| `st_counts.csv` | Spatial expression matrix (if `--st_data_dir` provided). |
| `*_consistent_cells_*_umap.png` | UMAP showing only consistent cells. |
| `*_consistency_context_all_cells_umap.png` | UMAP with inconsistent cells in grey. |
| `refined_annotation_exports/` | Additional exports using refined annotations. |

---

## Marker Prior Score (MPS) Details

The MPS feature validates cluster marker genes against a canonical marker database using **F1 Score**:
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Where:

- **Precision** = `|DEGs ∩ Canonical| / |DEGs|` — What fraction of top DEGs are canonical markers?
- **Recall** = `|DEGs ∩ Canonical| / |Canonical|` — What fraction of canonical markers appear in top DEGs?

---

## Cell Type Matching

The pipeline uses multiple matching strategies:

| Strategy | Description |
|----------|-------------|
| **Exact match** | Direct name match |
| **Case-insensitive match** | Ignores capitalization |
| **Normalized match** | Removes prefixes/suffixes, standardizes terms |
| **Substring match** | Detects partial matches |
| **Fuzzy match** | Uses similarity algorithms for approximate matching |

---

## Abbreviation Expansion

Common abbreviations are automatically expanded:

| Abbreviation | Expansion |
|--------------|-----------|
| `OPC` | `oligodendrocyte precursor cell` |
| `Astro` | `astrocyte` |
| `L5-6 Exc` | `layer 5-6 excitatory neuron` |
| ... | |
---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.