#!/usr/bin/env Rscript

# ==============================================================================
# --- Integrated Bayesian Optimization & Visualization Pipeline for scRNA-seq ---
# ==============================================================================
#
# DESCRIPTION:
# This script combines multiple stages into a single, automated workflow:
#
# 1. OPTIMIZATION STAGE (Bayesian Optimization):
#    - Finds the optimal scRNA-seq analysis parameters (HVGs, PCs, neighbors,
#      resolution) by optimizing a multi-objective score (CAS, MCS, Silhouette).
#    - Supports both HUMAN and MOUSE data for correct gene/mito prefix handling.
#    - Compares three optimization strategies ('Exploit', 'Explore', 'BO-EI').
#
# 2. VISUALIZATION STAGE (Fixed-Parameter Run):
#    - Takes the best parameter set found in Stage 1.
#    - Runs a complete Seurat analysis pipeline with optimal parameters.
#    - Generates publication-ready plots and tables.
#
# 3. REFINEMENT STAGE (Optional):
#    - Iteratively refines low-confidence cell populations.
#    - Re-runs optimization and analysis on failing clusters.
#
# 4. CONSISTENT CELLS EXPORT:
#    - Exports cells with consistent annotations across all methods.
#    - Generates deconvolution-ready files (sc_counts, sc_labels, st_counts).

# --- Load Libraries ---
suppressPackageStartupMessages({
    library(Seurat)
    library(dplyr)
    library(ggplot2)
    library(argparse)
    library(rBayesianOptimization)
    library(Rtsne)
    library(cluster)
    library(AnnotationDbi)
    library(patchwork)
    library(tibble)
    library(viridis)
    library(Matrix)
    library(uwot)
    library(scales)
    library(harmony)
})

# ==============================================================================
# --- GLOBAL CONFIGURATION ---
# ==============================================================================

# QC Thresholds
MIN_GENES_PER_CELL <- 200
MAX_GENES_PER_CELL <- 7000
MAX_PCT_COUNTS_MT <- 10
MIN_CELLS_PER_GENE <- 3

# Search Space Bounds for Bayesian Optimization
SEARCH_SPACE_BOUNDS <- list(
    n_hvg = c(800L, 20000L),
    n_pcs = c(10L, 100L),
    n_neighbors = c(10L, 50L),
    resolution = c(0.2, 1.0)
)

# Harmony Integration Parameters (NEW)
HARMONY_CONFIG <- list(
    theta = 2,           # Diversity clustering penalty
    lambda = 1,          # Ridge regression penalty
    sigma = 0.1,         # Width of soft kmeans clusters
    nclust = NULL,       # Number of clusters (NULL = auto)
    max_iter = 20,       # Maximum iterations
    early_stop = TRUE    # Early stopping
)

#' Cell Type Abbreviation to Full Name Mapping
#' Used to expand short annotations (e.g., "OPC") to full names for marker matching
CELLTYPE_ABBREVIATION_MAP <- list(
    # Direct mappings (case-insensitive keys will be handled in function)
    "Astrocyte" = "Astrocyte",
    "Chandelier" = "Chandelier Cell (Pvalb+ GABAergic interneuron)",
    "Endothelial" = "Endothelial Cell",
    "L2/3 IT" = "Layer 2/3 Intratelencephalic-projecting excitatory neuron",
    "L2/3IT" = "Layer 2/3 Intratelencephalic-projecting excitatory neuron",
    "L23 IT" = "Layer 2/3 Intratelencephalic-projecting excitatory neuron",
    "L23IT" = "Layer 2/3 Intratelencephalic-projecting excitatory neuron",
    "L4 IT" = "Layer 4 Intratelencephalic-projecting excitatory neuron",
    "L4IT" = "Layer 4 Intratelencephalic-projecting excitatory neuron",
    "L5 ET" = "Layer 5 Extratelencephalic-projecting excitatory neuron",
    "L5ET" = "Layer 5 Extratelencephalic-projecting excitatory neuron",
    "L5 IT" = "Layer 5 Intratelencephalic-projecting excitatory neuron",
    "L5IT" = "Layer 5 Intratelencephalic-projecting excitatory neuron",
    "L5/6 NP" = "Layer 5/6 Near-projecting excitatory neuron",
    "L56 NP" = "Layer 5/6 Near-projecting excitatory neuron",
    "L5/6NP" = "Layer 5/6 Near-projecting excitatory neuron",
    "L56NP" = "Layer 5/6 Near-projecting excitatory neuron",
    "L6 CT" = "Layer 6 Corticothalamic-projecting excitatory neuron",
    "L6CT" = "Layer 6 Corticothalamic-projecting excitatory neuron",
    "L6 IT" = "Layer 6 Intratelencephalic-projecting excitatory neuron",
    "L6IT" = "Layer 6 Intratelencephalic-projecting excitatory neuron",
    "L6 IT Car3" = "Layer 6 Intratelencephalic Car3+ excitatory neuron",
    "L6IT Car3" = "Layer 6 Intratelencephalic Car3+ excitatory neuron",
    "L6ITCar3" = "Layer 6 Intratelencephalic Car3+ excitatory neuron",
    "L6b" = "Layer 6b excitatory neuron",
    "Lamp5" = "Lamp5+ GABAergic interneuron",
    "Lamp5 Lhx6" = "Lamp5+ Lhx6+ GABAergic interneuron",
    "Lamp5Lhx6" = "Lamp5+ Lhx6+ GABAergic interneuron",
    "Microglia-PVM" = "Microglia and Perivascular Macrophage",
    "Microglia" = "Microglia and Perivascular Macrophage",
    "PVM" = "Microglia and Perivascular Macrophage",
    "Oligodendrocyte" = "Oligodendrocyte",
    "Oligo" = "Oligodendrocyte",
    "OPC" = "Oligodendrocyte Precursor Cell",
    "Pax6" = "Pax6+ GABAergic interneuron",
    "Pvalb" = "Parvalbumin+ GABAergic interneuron",
    "PV" = "Parvalbumin+ GABAergic interneuron",
    "Sncg" = "Synuclein Gamma+ GABAergic interneuron",
    "Sst" = "Somatostatin+ GABAergic interneuron",
    "SST" = "Somatostatin+ GABAergic interneuron",
    "Sst Chodl" = "Somatostatin+ Chodl+ long-projection GABAergic neuron",
    "SstChodl" = "Somatostatin+ Chodl+ long-projection GABAergic neuron",
    "Vip" = "Vasoactive Intestinal Peptide+ GABAergic interneuron",
    "VIP" = "Vasoactive Intestinal Peptide+ GABAergic interneuron",
    "VLMC" = "Vascular and Leptomeningeal Cell"
)

#' Create reverse mapping (full name -> abbreviation) for reference
CELLTYPE_FULLNAME_TO_ABBREV <- setNames(
    names(CELLTYPE_ABBREVIATION_MAP),
    unlist(CELLTYPE_ABBREVIATION_MAP)
)

#' Expand cell type abbreviation to full name
#' @param celltype_abbrev Abbreviated cell type name (e.g., "OPC")
#' @param use_full_for_matching If TRUE, return full name; if FALSE, return original
#' @return Full cell type name if mapping exists, otherwise original name
expand_celltype_abbreviation <- function(celltype_abbrev, use_full_for_matching = TRUE) {
    if (is.null(celltype_abbrev) || is.na(celltype_abbrev) || celltype_abbrev == "") {
        return(celltype_abbrev)
    }
    
    if (!use_full_for_matching) {
        return(celltype_abbrev)
    }
    
    # Clean the input
    celltype_clean <- trimws(celltype_abbrev)
    
    # Try exact match first
    if (celltype_clean %in% names(CELLTYPE_ABBREVIATION_MAP)) {
        return(CELLTYPE_ABBREVIATION_MAP[[celltype_clean]])
    }
    
    # Try case-insensitive match
    for (abbrev in names(CELLTYPE_ABBREVIATION_MAP)) {
        if (tolower(celltype_clean) == tolower(abbrev)) {
            return(CELLTYPE_ABBREVIATION_MAP[[abbrev]])
        }
    }
    
    # Try matching with normalized spaces/separators
    celltype_normalized <- gsub("[_\\s-]+", " ", celltype_clean)
    celltype_normalized <- gsub("\\s+", " ", celltype_normalized)
    celltype_normalized <- trimws(celltype_normalized)
    
    for (abbrev in names(CELLTYPE_ABBREVIATION_MAP)) {
        abbrev_normalized <- gsub("[_\\s-]+", " ", abbrev)
        abbrev_normalized <- gsub("\\s+", " ", abbrev_normalized)
        abbrev_normalized <- trimws(abbrev_normalized)
        
        if (tolower(celltype_normalized) == tolower(abbrev_normalized)) {
            return(CELLTYPE_ABBREVIATION_MAP[[abbrev]])
        }
    }
    
    # No match found, return original
    return(celltype_abbrev)
}

#' Expand all cell type abbreviations in a vector
#' @param celltype_vector Vector of cell type names (may contain abbreviations)
#' @param verbose Print expansion info
#' @return Vector with abbreviations expanded to full names
expand_celltype_abbreviations_vector <- function(celltype_vector, verbose = FALSE) {
    if (is.null(celltype_vector) || length(celltype_vector) == 0) {
        return(celltype_vector)
    }
    
    expanded <- sapply(celltype_vector, expand_celltype_abbreviation, USE.NAMES = FALSE)
    
    if (verbose) {
        # Report expansions
        changed_mask <- celltype_vector != expanded
        n_changed <- sum(changed_mask, na.rm = TRUE)
        
        if (n_changed > 0) {
            cat(sprintf("[CELLTYPE EXPANSION] Expanded %d abbreviations:\n", n_changed))
            unique_changes <- unique(data.frame(
                original = celltype_vector[changed_mask],
                expanded = expanded[changed_mask],
                stringsAsFactors = FALSE
            ))
            for (i in 1:min(10, nrow(unique_changes))) {
                cat(sprintf("   '%s' -> '%s'\n", 
                            unique_changes$original[i], 
                            unique_changes$expanded[i]))
            }
            if (nrow(unique_changes) > 10) {
                cat(sprintf("   ... and %d more\n", nrow(unique_changes) - 10))
            }
        }
    }
    
    return(expanded)
}

#' Get standardized cell type name for matching (with abbreviation expansion)
#' @description Combines abbreviation expansion with standardization for robust matching
#' @param celltype_name Original cell type name
#' @param expand_abbreviations Whether to expand abbreviations first
#' @return Standardized cell type name ready for matching
get_matching_celltype_name <- function(celltype_name, expand_abbreviations = TRUE) {
    if (is.null(celltype_name) || is.na(celltype_name) || celltype_name == "") {
        return("")
    }
    
    # Step 1: Expand abbreviation if applicable
    if (expand_abbreviations) {
        celltype_name <- expand_celltype_abbreviation(celltype_name)
    }
    
    # Step 2: Standardize for matching (lowercase, remove special chars)
    standardized <- standardize_celltype_name(celltype_name)
    
    return(standardized)
}

# MPS Configuration (NEW)
MPS_CONFIG <- list(
    n_top_degs = 100,           # Number of top DEGs to check per cluster
    min_pct = 0.1,             # Minimum fraction of cells expressing gene
    logfc_threshold = 0.25,    # Log fold change threshold for DEGs
    p_adj_threshold = 0.05     # Adjusted p-value threshold
)

# PCA Configuration
N_PCS_FOR_PCA <- 105
N_MCS_TOP_GENES <- 5
MITO_REGEX_PATTERN <- "^(MT|Mt|mt)[-._:]"

# Global variables (will be initialized in main())
GLOBAL_ENV <- new.env()

# Color palette for UMAP plots
UMAP_COLOR_PALETTE <- c(
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#FFFF33", "#A65628", "#F781BF", "#999999", "#66C2A5",
    "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F",
    "#E5C494", "#B3B3B3", "#1B9E77", "#D95F02", "#7570B3",
    "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666",
    "#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3",
    "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD",
    "#CCEBC5", "#FFED6F", "#1F78B4", "#33A02C", "#FB9A99"
)

# ==============================================================================
# --- UTILITY FUNCTIONS ---
# ==============================================================================
#' Get color palette for plotting with automatic extension
#' @param n Number of colors needed
#' @return Vector of colors
get_plot_colors <- function(n) {
    if (n <= length(UMAP_COLOR_PALETTE)) {
        return(UMAP_COLOR_PALETTE[1:n])
    } else {
        # Extend palette if needed
        extended <- colorRampPalette(UMAP_COLOR_PALETTE)(n)
        return(extended)
    }
}

#' Detect if data contains batch information
#' @param seurat_obj Seurat object to check
#' @param batch_col Column name to check for batch info (user-specified)
#' @param barcode_separator Separator in barcodes for batch extraction
#' @return List with is_batched flag, batch_column name, and optionally updated seurat object
detect_batch_info <- function(seurat_obj, batch_col = NULL, barcode_separator = "_") {
    
    cat("\n" %+% paste(rep("-", 60), collapse="") %+% "\n")
    cat("BATCH DETECTION\n")
    cat(paste(rep("-", 60), collapse="") %+% "\n")
    
    # === PRIORITY 1: User-specified batch column ===
    if (!is.null(batch_col) && batch_col %in% colnames(seurat_obj@meta.data)) {
        n_unique <- length(unique(seurat_obj@meta.data[[batch_col]]))
        if (n_unique > 1) {
            batch_sizes <- table(seurat_obj@meta.data[[batch_col]])
            cat(sprintf("   ✓ Using user-specified batch column: '%s'\n", batch_col))
            cat(sprintf("   ✓ Number of batches: %d\n", n_unique))
            cat(sprintf("   ✓ Batch sizes: %s\n", 
                        paste(sprintf("%s=%d", names(batch_sizes), batch_sizes), collapse = ", ")))
            
            return(list(
                is_batched = TRUE,
                batch_column = batch_col,
                n_batches = n_unique,
                batch_sizes = batch_sizes,
                seurat_obj = seurat_obj,
                source = "user_specified"
            ))
        }
    }
    
    # === PRIORITY 2: Auto-detect from common metadata columns ===
    candidate_cols <- c(
        "batch", "Batch", "BATCH",
        "sample", "Sample", "SAMPLE", 
        "orig.ident",
        "dataset", "Dataset",
        "library", "Library",
        "donor", "Donor",
        "patient", "Patient",
        "subject", "Subject"
    )
    
    metadata_cols <- colnames(seurat_obj@meta.data)
    
    for (col in candidate_cols) {
        if (col %in% metadata_cols) {
            n_unique <- length(unique(seurat_obj@meta.data[[col]]))
            if (n_unique > 1) {
                batch_sizes <- table(seurat_obj@meta.data[[col]])
                cat(sprintf("   ✓ Auto-detected batch column: '%s'\n", col))
                cat(sprintf("   ✓ Number of batches: %d\n", n_unique))
                cat(sprintf("   ✓ Batch sizes: %s\n", 
                            paste(sprintf("%s=%d", names(batch_sizes), batch_sizes), collapse = ", ")))
                
                return(list(
                    is_batched = TRUE,
                    batch_column = col,
                    n_batches = n_unique,
                    batch_sizes = batch_sizes,
                    seurat_obj = seurat_obj,
                    source = "auto_metadata"
                ))
            }
        }
    }
    
    # === PRIORITY 3: Extract from barcodes ===
    cat("   → No batch column found in metadata. Checking barcodes...\n")
    
    # Show sample barcodes
    sample_barcodes <- head(colnames(seurat_obj), 3)
    cat(sprintf("   → Sample barcodes: %s\n", paste(sample_barcodes, collapse = ", ")))
    
    # Try to extract batch from barcodes
    extraction_result <- extract_batch_from_barcodes(
        seurat_obj, 
        separator = barcode_separator, 
        position = "last"
    )
    
    if (extraction_result$success) {
        seurat_obj <- extraction_result$seurat
        
        return(list(
            is_batched = TRUE,
            batch_column = "barcode_batch",
            n_batches = extraction_result$n_batches,
            batch_sizes = extraction_result$batch_table,
            seurat_obj = seurat_obj,  # <-- IMPORTANT: Return modified object
            source = "barcode_extracted"
        ))
    }
    
    # === NO BATCH DETECTED ===
    cat("   → No batch information detected. Running single-sample pipeline.\n")
    cat(paste(rep("-", 60), collapse="") %+% "\n")
    
    return(list(
        is_batched = FALSE,
        batch_column = NULL,
        n_batches = 1,
        batch_sizes = NULL,
        seurat_obj = seurat_obj,
        source = "none"
    ))
}

#' Extract batch information from cell barcodes
#' @param seurat_obj Seurat object
#' @param separator Character separating barcode from batch ID (default: "_")
#' @param position Which part contains batch: "last" (default) or "first"
#' @return Seurat object with "barcode_batch" column added to metadata
extract_batch_from_barcodes <- function(seurat_obj, separator = "_", position = "last") {
    
    barcodes <- colnames(seurat_obj)
    
    # Check if barcodes contain the separator
    if (!any(grepl(separator, barcodes, fixed = TRUE))) {
        cat(sprintf("[BATCH EXTRACTION] No '%s' separator found in barcodes.\n", separator))
        return(list(seurat = seurat_obj, success = FALSE, batch_ids = NULL))
    }
    
    # Extract batch IDs
    if (position == "last") {
        # Pattern: AAACAAGTATCTCCCA-1_Br2720 -> Br2720
        batch_ids <- sapply(strsplit(barcodes, separator, fixed = TRUE), function(x) {
            if (length(x) >= 2) {
                return(tail(x, 1))  # Get last element
            } else {
                return("unknown")
            }
        })
    } else {
        # Pattern: Br2720_AAACAAGTATCTCCCA-1 -> Br2720  
        batch_ids <- sapply(strsplit(barcodes, separator, fixed = TRUE), function(x) {
            if (length(x) >= 2) {
                return(x[1])  # Get first element
            } else {
                return("unknown")
            }
        })
    }
    
    # Add to metadata
    seurat_obj$barcode_batch <- batch_ids
    
    # Report
    batch_table <- table(batch_ids)
    n_batches <- length(batch_table)
    
    cat(sprintf("[BATCH EXTRACTION] Extracted %d batches from barcodes:\n", n_batches))
    for (batch_name in names(batch_table)) {
        cat(sprintf("   -> %s: %d cells\n", batch_name, batch_table[batch_name]))
    }
    
    return(list(
        seurat = seurat_obj, 
        success = (n_batches > 1), 
        batch_ids = batch_ids,
        n_batches = n_batches,
        batch_table = batch_table
    ))
}

#' Run Harmony integration on Seurat object
#' @param seurat_obj Seurat object with PCA computed
#' @param batch_col Column containing batch information
#' @param dims_use Dimensions to use for integration
#' @param verbose Print progress messages
#' @return Seurat object with harmony reduction
run_harmony_integration <- function(seurat_obj, batch_col, dims_use = 1:30, verbose = TRUE) {
    
    if (verbose) {
        cat(sprintf("[HARMONY] Integrating across '%s' (%d batches)...\n", 
                    batch_col, length(unique(seurat_obj@meta.data[[batch_col]]))))
    }
    
    # Ensure PCA exists
    if (!"pca" %in% names(seurat_obj@reductions)) {
        stop("PCA must be computed before running Harmony integration")
    }
    
    # Get number of PCs available
    n_pcs_available <- ncol(Embeddings(seurat_obj, "pca"))
    dims_use <- 1:min(max(dims_use), n_pcs_available)
    
    # Run Harmony
    seurat_obj <- RunHarmony(
        seurat_obj,
        group.by.vars = batch_col,
        dims.use = dims_use,
        theta = HARMONY_CONFIG$theta,
        lambda = HARMONY_CONFIG$lambda,
        sigma = HARMONY_CONFIG$sigma,
        nclust = HARMONY_CONFIG$nclust,
        max.iter.harmony = HARMONY_CONFIG$max_iter,
        early_stop = HARMONY_CONFIG$early_stop,
        verbose = verbose,
        reduction.save = "harmony"
    )
    
    if (verbose) {
        cat("[HARMONY] Integration complete.\n")
    }
    
    return(seurat_obj)
}

#' String concatenation operator
`%+%` <- function(a, b) paste0(a, b)

#' Null coalescing operator
`%||%` <- function(a, b) if (is.null(a)) b else a

#' Check and load Bioconductor package
check_and_load_bioc_package <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        if (!requireNamespace("BiocManager", quietly = TRUE)) {
            install.packages("BiocManager")
        }
        BiocManager::install(pkg, ask = FALSE, update = FALSE)
    }
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

#' Safe metric sanitization with multiple validation layers
#' @param x Input value to sanitize
#' @param default Default value if x is invalid
#' @param min_val Minimum allowed value (optional)
#' @param max_val Maximum allowed value (optional)
#' @return Sanitized numeric value
sanitize_metric <- function(x, default = 0.0, min_val = NULL, max_val = NULL) {
    # Validate default first
    if (!is.numeric(default) || length(default) != 1 || !is.finite(default)) {
        default <- 0.0
    }
    
    # Handle NULL, empty, or wrong type
    if (is.null(x) || length(x) == 0) {
        return(as.numeric(default))
    }
    
    # Handle non-numeric types
    if (!is.numeric(x) && !is.logical(x)) {
        return(as.numeric(default))
    }
    
    # Convert to numeric and take first element
    x <- suppressWarnings(as.numeric(x[1]))
    
    # Check for all non-finite cases
    if (is.na(x) || is.nan(x) || is.infinite(x)) {
        return(as.numeric(default))
    }
    
    # Apply bounds if specified
    if (!is.null(min_val) && is.finite(min_val)) {
        x <- max(x, min_val)
    }
    if (!is.null(max_val) && is.finite(max_val)) {
        x <- min(x, max_val)
    }
    
    return(as.numeric(x))
}

#' Safe geometric mean calculation
#' @param values Numeric vector of values
#' @param epsilon Small value to add for numerical stability
#' @return Geometric mean
safe_geometric_mean <- function(values, epsilon = 1e-6) {
    # Sanitize all input values
    values <- sapply(values, function(v) sanitize_metric(v, epsilon, min_val = epsilon))
    
    # Ensure all values are positive
    values <- pmax(values, epsilon)
    
    # Calculate using log transform for numerical stability
    log_values <- log(values)
    
    # Check for non-finite log values
    if (any(!is.finite(log_values))) {
        log_values[!is.finite(log_values)] <- log(epsilon)
    }
    
    result <- exp(mean(log_values))
    
    # Final sanitization
    return(sanitize_metric(result, epsilon, min_val = epsilon))
}

# Check SeuratObject version
if (packageVersion("SeuratObject") >= "5.0.0") {
    cat("[INFO] Using SeuratObject 5.0+ layer syntax\n")
}
#' Ensure Seurat object uses standard Assay (not Assay5)
#' @description Converts Assay5 to standard Assay format safely, preserving all data layers.
#' @param seurat_obj Seurat object to convert
#' @param verbose Print info message (default TRUE)
#' @return Seurat object with standard Assay
ensure_standard_assay <- function(seurat_obj, verbose = TRUE) {
    if (is.null(seurat_obj)) {
        return(seurat_obj)
    }
    
    current_assay <- DefaultAssay(seurat_obj)
    assay_obj <- seurat_obj[[current_assay]]
    
    # Check if it's already a standard Assay (not Assay5)
    if (!inherits(assay_obj, "Assay5")) {
        if (verbose) cat("[INFO] Assay is already in standard format.\n")
        return(seurat_obj)
    }
    
    if (verbose) {
        cat("[INFO] Converting Assay5 to standard Assay format...\n")
    }
    
    # Detect SeuratObject version for proper syntax
    use_layer_syntax <- (packageVersion("SeuratObject") >= "5.0.0")
    
    # Extract all existing layers with robust error handling
    counts_data <- tryCatch({
        if (use_layer_syntax) {
            GetAssayData(seurat_obj, assay = current_assay, layer = "counts")
        } else {
            GetAssayData(seurat_obj, assay = current_assay, slot = "counts")
        }
    }, error = function(e) {
        if (verbose) cat("   -> Note: 'counts' layer not found\n")
        NULL
    })
    
    data_data <- tryCatch({
        if (use_layer_syntax) {
            GetAssayData(seurat_obj, assay = current_assay, layer = "data")
        } else {
            GetAssayData(seurat_obj, assay = current_assay, slot = "data")
        }
    }, error = function(e) {
        if (verbose) cat("   -> Note: 'data' layer not found\n")
        NULL
    })
    
    scale_data_data <- tryCatch({
        layer_data <- if (use_layer_syntax) {
            GetAssayData(seurat_obj, assay = current_assay, layer = "scale.data")
        } else {
            GetAssayData(seurat_obj, assay = current_assay, slot = "scale.data")
        }
        # Check if it's actually populated
        if (is.null(layer_data) || length(layer_data) == 0 || 
            (is.matrix(layer_data) && (nrow(layer_data) == 0 || ncol(layer_data) == 0))) {
            NULL
        } else {
            layer_data
        }
    }, error = function(e) {
        NULL
    })
    
    # We need at least counts to proceed
    if (is.null(counts_data)) {
        warning("Could not extract counts matrix during conversion. Returning original object.")
        return(seurat_obj)
    }
    
    # Attempt standard coercion first
    conversion_success <- FALSE
    tryCatch({
        converted_assay <- as(assay_obj, "Assay")
        seurat_obj[[current_assay]] <- converted_assay
        conversion_success <- TRUE
        if (verbose) cat("   -> Standard coercion successful\n")
    }, error = function(e) {
        if (verbose) cat("   -> Standard coercion failed:", conditionMessage(e), "\n")
    })
    
    # If coercion failed, do manual rebuild
    if (!conversion_success) {
        if (verbose) cat("   -> Attempting manual Assay rebuild...\n")
        
        tryCatch({
            # Create new assay from counts
            new_assay <- CreateAssayObject(counts = counts_data)
            
            # Add data layer if it exists and differs from counts
            if (!is.null(data_data)) {
                # Check if data is different from counts (i.e., normalized)
                if (!identical(counts_data, data_data)) {
                    # Use slot for standard Assay objects (not Assay5)
                    new_assay <- SetAssayData(new_assay, slot = "data", new.data = data_data)
                    if (verbose) cat("      -> Preserved 'data' layer.\n")
                }
            }
            
            # Add scale.data if it exists
            if (!is.null(scale_data_data) && length(rownames(scale_data_data)) > 0) {
                tryCatch({
                    new_assay <- SetAssayData(new_assay, slot = "scale.data", new.data = scale_data_data)
                    if (verbose) cat("      -> Preserved 'scale.data' layer.\n")
                }, error = function(e) {
                    if (verbose) cat("      -> Could not preserve 'scale.data' layer\n")
                })
            }
            
            seurat_obj[[current_assay]] <- new_assay
            if (verbose) cat("   -> Manual rebuild successful.\n")
            
        }, error = function(e) {
            warning(sprintf("Manual Assay rebuild failed: %s. Returning original object.", e$message))
        })
    }
    
    return(seurat_obj)
}

#' Ultra-safe return value for Bayesian Optimization
#' @description Creates a guaranteed valid return value with unique jitter
#' @param score The score to return
#' @param add_jitter Whether to add unique jitter (prevents GP fitting issues)
#' @return List with Score and Pred components
safe_bo_return <- function(score = 0.001, add_jitter = TRUE) {
    
    # Stage 1: Initial validation
    if (is.null(score) || length(score) == 0) {
        score <- 0.001
    }
    
    # Stage 2: Convert to numeric
    score <- tryCatch({
        as.numeric(score[1])
    }, error = function(e) {
        0.001
    }, warning = function(w) {
        suppressWarnings(as.numeric(score[1]))
    })
    
    # Handle conversion failure
    if (is.null(score) || length(score) == 0) {
        score <- 0.001
    }
    
    # Stage 3: Check for non-finite values
    if (is.na(score) || is.nan(score) || is.infinite(score)) {
        score <- 0.001
    }
    
    # Stage 4: Ensure strictly positive (GP requirement)
    if (score <= 0) {
        score <- 1e-6
    }
    
    # Stage 5: Cap extremely large values
    if (score > 1e6) {
        score <- 1e6
    }
    
    # Stage 6: Add unique jitter (CRITICAL for GP)
    if (add_jitter) {
        # Use multiple sources of randomness for uniqueness
        time_component <- as.numeric(Sys.time())
        time_frac <- abs((time_component %% 1)) * 1e-9
        random_component <- runif(1, min = 1e-10, max = 1e-8)
        
        # Combine for unique jitter
        jitter_value <- time_frac + random_component
        
        # Ensure jitter is finite
        if (!is.finite(jitter_value)) {
            jitter_value <- runif(1, 1e-10, 1e-8)
        }
        
        score <- score + jitter_value
    }
    
    # Stage 7: Final validation
    score <- as.numeric(score)
    
    # Triple-check finiteness
    if (!is.finite(score)) {
        warning("safe_bo_return: Score became non-finite after processing. Using fallback.")
        score <- 0.001 + runif(1, 1e-10, 1e-8)
    }
    
    return(list(Score = score, Pred = 0))
}

#' Create clean UMAP theme without grid, with square aspect ratio, keeping legend
#' @return ggplot2 theme object
theme_umap_clean <- function() {
    theme_minimal(base_size = 12) +
    theme(
        # Remove grid lines
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        
        # Square aspect ratio
        aspect.ratio = 1,
        
        # Clean axis appearance
        axis.line = element_line(color = "black", linewidth = 0.5),
        axis.ticks = element_line(color = "black", linewidth = 0.3),
        axis.text = element_text(size = 10, color = "black"),
        axis.title = element_text(size = 12, face = "bold"),
        
        # Title styling
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 10, hjust = 0.5, color = "grey40"),
        
        # Legend styling (KEPT)
        legend.position = "right",
        legend.text = element_text(size = 9),
        legend.title = element_text(size = 10, face = "bold")
    )
}

#' Load Marker Gene Database with Flexible Format Support
#' 
#' Supports two formats:
#' 1. Long format: cell_type, gene (one gene per row)
#' 2. Wide format: cell_type, marker_genes (semicolon-separated genes)
#'
#' @param file_path Path to CSV file
#' @param species Species to filter for (human/mouse)
#' @param organ Optional organ/tissue filter
#' @return Named list: cell_type -> vector of standardized gene names
load_marker_database <- function(file_path, species = "human", organ = NULL) {
    if (is.null(file_path) || !file.exists(file_path)) {
        cat("[MPS WARNING] Marker database file not found:", file_path, "\n")
        return(list())
    }
    
    cat("[MPS] Loading marker database from:", file_path, "\n")
    
    tryCatch({
        # Read the CSV file
        marker_df <- read.csv(file_path, stringsAsFactors = FALSE, 
                              na.strings = c("", "NA", "N/A"))
        
        # Standardize column names (lowercase, remove spaces/underscores)
        orig_colnames <- colnames(marker_df)
        colnames(marker_df) <- tolower(gsub("[_ ]+", "", colnames(marker_df)))
        
        cat("[MPS] Original columns:", paste(orig_colnames, collapse = ", "), "\n")
        cat("[MPS] Standardized columns:", paste(colnames(marker_df), collapse = ", "), "\n")
        
        # Detect format: wide (marker_genes) or long (gene)
        has_marker_genes <- "markergenes" %in% colnames(marker_df)
        has_gene <- "gene" %in% colnames(marker_df)
        has_cell_type <- "celltype" %in% colnames(marker_df)
        
        if (!has_cell_type) {
            cat("[MPS ERROR] 'cell_type' column not found in marker file.\n")
            return(list())
        }
        
        # Filter by species if column exists
        if ("species" %in% colnames(marker_df)) {
            species_std <- tolower(trimws(species))
            marker_df$species <- tolower(trimws(marker_df$species))
            
            before_filter <- nrow(marker_df)
            marker_df <- marker_df[marker_df$species == species_std, ]
            cat("[MPS] Filtered by species '", species, "': ", 
                before_filter, " -> ", nrow(marker_df), " rows\n", sep = "")
        }
        
        # Filter by organ if specified and column exists
        if (!is.null(organ) && "organ" %in% colnames(marker_df)) {
            organ_std <- standardize_celltype_name(organ)
            marker_df$organ_std <- sapply(marker_df$organ, standardize_celltype_name)
            
            before_filter <- nrow(marker_df)
            marker_df <- marker_df[marker_df$organ_std == organ_std, ]
            cat("[MPS] Filtered by organ '", organ, "': ", 
                before_filter, " -> ", nrow(marker_df), " rows\n", sep = "")
        }
        
        if (nrow(marker_df) == 0) {
            cat("[MPS WARNING] No markers remaining after filtering.\n")
            return(list())
        }
        
        # Process based on format
        if (has_marker_genes) {
            # WIDE FORMAT: marker_genes column with semicolon-separated genes
            cat("[MPS] Detected WIDE format (marker_genes column with ';' separator)\n")
            
            marker_dict <- list()
            
            for (i in seq_len(nrow(marker_df))) {
                cell_type_raw <- marker_df$celltype[i]
                genes_raw <- marker_df$markergenes[i]
                
                if (is.na(cell_type_raw) || is.na(genes_raw)) next
                if (trimws(cell_type_raw) == "" || trimws(genes_raw) == "") next
                
                # Standardize cell type name
                cell_type_std <- standardize_celltype_name(cell_type_raw)
                
                # Split genes by semicolon, comma, or pipe
                genes <- unlist(strsplit(genes_raw, "[;,|]+"))
                genes <- trimws(genes)
                genes <- genes[genes != ""]
                
                # Standardize gene names for the species
                genes_std <- sapply(genes, function(g) {
                    standardize_gene_name(g, species = species)
                })
                genes_std <- unique(genes_std[genes_std != ""])
                
                if (length(genes_std) > 0) {
                    if (cell_type_std %in% names(marker_dict)) {
                        # Merge with existing genes
                        marker_dict[[cell_type_std]] <- unique(c(
                            marker_dict[[cell_type_std]], 
                            genes_std
                        ))
                    } else {
                        marker_dict[[cell_type_std]] <- genes_std
                    }
                }
            }
            
        } else if (has_gene) {
            # LONG FORMAT: one gene per row
            cat("[MPS] Detected LONG format (gene column)\n")
            
            marker_df$celltype_std <- sapply(marker_df$celltype, standardize_celltype_name)
            marker_df$gene_std <- sapply(marker_df$gene, function(g) {
                standardize_gene_name(g, species = species)
            })
            
            # Remove empty entries
            marker_df <- marker_df[!is.na(marker_df$gene_std) & marker_df$gene_std != "", ]
            
            # Aggregate by cell type
            marker_dict <- split(marker_df$gene_std, marker_df$celltype_std)
            marker_dict <- lapply(marker_dict, unique)
            
        } else {
            cat("[MPS ERROR] Could not detect format. Need 'gene' or 'marker_genes' column.\n")
            cat("   Available columns:", paste(colnames(marker_df), collapse = ", "), "\n")
            return(list())
        }
        
        # Remove empty entries
        marker_dict <- marker_dict[sapply(marker_dict, length) > 0]
        
        cat("[MPS] Successfully loaded marker database:\n")
        cat("   - Cell types:", length(marker_dict), "\n")
        cat("   - Total unique markers:", length(unique(unlist(marker_dict))), "\n")
        
        # Show sample entries
        if (length(marker_dict) > 0) {
            cat("   - Sample entries:\n")
            sample_types <- head(names(marker_dict), 3)
            for (ct in sample_types) {
                genes_preview <- paste(head(marker_dict[[ct]], 5), collapse = ", ")
                if (length(marker_dict[[ct]]) > 5) {
                    genes_preview <- paste0(genes_preview, ", ... (", 
                                           length(marker_dict[[ct]]), " total)")
                }
                cat("     ", ct, ":", genes_preview, "\n")
            }
        }
        
        return(marker_dict)
        
    }, error = function(e) {
        cat("[MPS ERROR] Failed to load marker database:", conditionMessage(e), "\n")
        return(list())
    })
}

#' Standardize gene names based on species
#' @description Converts gene names to standard format:
#'   - Human: UPPERCASE (e.g., CD4, PTPRC, MT-CO1)
#'   - Mouse: Title case (e.g., Cd4, Ptprc, mt-Co1)
#' @param gene_names Vector of gene names
#' @param species "human" or "mouse"
#' @return Standardized gene names
standardize_gene_names <- function(gene_names, species = "human") {
    if (is.null(gene_names) || length(gene_names) == 0) {
        return(gene_names)
    }
    
    # Remove any leading/trailing whitespace
    gene_names <- trimws(gene_names)
    
    if (species == "human") {
        # Human genes: ALL UPPERCASE
        standardized <- toupper(gene_names)
    } else if (species == "mouse") {
        # Mouse genes: First letter uppercase, rest lowercase
        # Exception: Mitochondrial genes keep "mt-" prefix lowercase
        standardized <- sapply(gene_names, function(g) {
            if (grepl("^(MT|Mt|mt)[-._:]", g, ignore.case = TRUE)) {
                # Mitochondrial gene: mt-Xxx format
                parts <- strsplit(g, "[-._:]")[[1]]
                if (length(parts) >= 2) {
                    paste0("mt-", tools::toTitleCase(tolower(parts[2])))
                } else {
                    paste0("mt-", tools::toTitleCase(tolower(gsub("^(MT|Mt|mt)[-._:]?", "", g))))
                }
            } else {
                # Standard gene: First uppercase, rest lowercase
                tools::toTitleCase(tolower(g))
            }
        }, USE.NAMES = FALSE)
    } else {
        standardized <- gene_names
    }
    
    return(standardized)
}

#' Standardize cell type names for robust matching
#' @description Normalizes cell type names by:
#'   - Trimming whitespace
#'   - Standardizing case (lowercase for comparison)
#'   - Removing special characters
#'   - Standardizing common variations
#' @param celltype_names Vector of cell type names
#' @param for_comparison If TRUE, returns lowercase for comparison; if FALSE, returns cleaned original case
#' @return Standardized cell type names
standardize_celltype_names <- function(celltype_names, for_comparison = FALSE) {
    if (is.null(celltype_names) || length(celltype_names) == 0) {
        return(celltype_names)
    }
    
    standardized <- sapply(celltype_names, function(ct) {
        if (is.na(ct) || ct == "") return(ct)
        
        # Trim whitespace
        ct <- trimws(ct)
        
        # Replace multiple spaces with single space
        ct <- gsub("\\s+", " ", ct)
        
        # Standardize common separators (underscores, dots) to spaces
        ct_clean <- gsub("[_.]", " ", ct)
        
        # Remove special characters except spaces and hyphens
        ct_clean <- gsub("[^a-zA-Z0-9 -]", "", ct_clean)
        
        # Collapse multiple spaces
        ct_clean <- gsub("\\s+", " ", ct_clean)
        ct_clean <- trimws(ct_clean)
        
        if (for_comparison) {
            return(tolower(ct_clean))
        } else {
            return(ct_clean)
        }
    }, USE.NAMES = FALSE)
    
    return(standardized)
}

#' Create cell type name lookup dictionary
#' @description Creates a mapping from standardized names to original names
#' @param original_names Vector of original cell type names
#' @return Named list: standardized_name -> original_name
create_celltype_lookup <- function(original_names) {
    original_names <- unique(original_names[!is.na(original_names)])
    standardized <- standardize_celltype_names(original_names, for_comparison = TRUE)
    
    lookup <- setNames(original_names, standardized)
    return(lookup)
}

#' Diagnose cell type matching between annotations and marker database
#' @param seurat_obj Seurat object with annotations
#' @param marker_db Marker database from load_marker_database()
#' @param annotation_col Column containing cell type annotations
#' @param expand_abbreviations Whether to expand abbreviations before matching
#' @return Dataframe with matching diagnostics
diagnose_celltype_matching <- function(seurat_obj, marker_db, 
                                        annotation_col = "ctpt_consensus_prediction",
                                        expand_abbreviations = TRUE) {
    
    if (is.null(marker_db) || is.null(marker_db$markers)) {
        cat("[DIAGNOSTIC] No marker database available.\n")
        return(NULL)
    }
    
    marker_types <- names(marker_db$markers)
    
    # Get unique annotated types
    if (!(annotation_col %in% colnames(seurat_obj@meta.data))) {
        cat(sprintf("[DIAGNOSTIC] Column '%s' not found.\n", annotation_col))
        return(NULL)
    }
    
    annotated_types <- unique(as.character(seurat_obj@meta.data[[annotation_col]]))
    annotated_types <- annotated_types[!is.na(annotated_types) & annotated_types != ""]
    
    cat("\n" , paste(rep("=", 70), collapse=""), "\n")
    cat("CELL TYPE MATCHING DIAGNOSTICS\n")
    cat(paste(rep("=", 70), collapse=""), "\n\n")
    
    cat(sprintf("Annotated cell types: %d\n", length(annotated_types)))
    cat(sprintf("Marker DB cell types: %d\n", length(marker_types)))
    cat(sprintf("Abbreviation expansion: %s\n", ifelse(expand_abbreviations, "ENABLED", "DISABLED")))
    cat("\n")
    
    # Check each annotated type
    results <- data.frame(
        annotated_type = character(),
        expanded_name = character(),
        standardized = character(),
        matched_to = character(),
        match_type = character(),
        n_markers = integer(),
        stringsAsFactors = FALSE
    )
    
    for (atype in annotated_types) {
        # Expand abbreviation first
        expanded_name <- atype
        if (expand_abbreviations) {
            expanded_name <- expand_celltype_abbreviation(atype)
        }
        
        std_type <- standardize_celltype_name(expanded_name)
        matched <- find_best_celltype_match(atype, marker_types, 
                                             expand_abbreviations = expand_abbreviations)
        
        if (!is.null(matched)) {
            n_markers <- length(marker_db$markers[[matched]])
            
            if (std_type == matched) {
                match_type <- "exact"
            } else if (expanded_name != atype) {
                match_type <- "abbreviation_expanded"
            } else {
                match_type <- "fuzzy"
            }
        } else {
            n_markers <- 0
            match_type <- "none"
            matched <- NA
        }
        
        results <- rbind(results, data.frame(
            annotated_type = atype,
            expanded_name = expanded_name,
            standardized = std_type,
            matched_to = matched,
            match_type = match_type,
            n_markers = n_markers,
            stringsAsFactors = FALSE
        ))
    }
    
    # Print summary
    cat("MATCHING RESULTS:\n")
    cat(paste(rep("-", 70), collapse=""), "\n")
    
    n_exact <- sum(results$match_type == "exact")
    n_abbrev <- sum(results$match_type == "abbreviation_expanded")
    n_fuzzy <- sum(results$match_type == "fuzzy")
    n_none <- sum(results$match_type == "none")
    
    cat(sprintf("  Exact matches:              %d (%.1f%%)\n", n_exact, 100*n_exact/nrow(results)))
    cat(sprintf("  Abbreviation expansions:    %d (%.1f%%)\n", n_abbrev, 100*n_abbrev/nrow(results)))
    cat(sprintf("  Fuzzy matches:              %d (%.1f%%)\n", n_fuzzy, 100*n_fuzzy/nrow(results)))
    cat(sprintf("  No match:                   %d (%.1f%%)\n", n_none, 100*n_none/nrow(results)))
    cat("\n")
    
    # Show abbreviation expansions
    if (n_abbrev > 0) {
        cat("ABBREVIATION EXPANSIONS:\n")
        abbrev_results <- results[results$match_type == "abbreviation_expanded", ]
        for (i in 1:nrow(abbrev_results)) {
            cat(sprintf("  '%s' -> '%s' => matched to '%s' (%d markers)\n",
                        abbrev_results$annotated_type[i],
                        abbrev_results$expanded_name[i],
                        abbrev_results$matched_to[i],
                        abbrev_results$n_markers[i]))
        }
        cat("\n")
    }
    
    # Show unmatched types
    if (n_none > 0) {
        cat("UNMATCHED CELL TYPES:\n")
        unmatched <- results[results$match_type == "none", ]
        for (i in 1:nrow(unmatched)) {
            cat(sprintf("  - '%s'", unmatched$annotated_type[i]))
            if (unmatched$expanded_name[i] != unmatched$annotated_type[i]) {
                cat(sprintf(" (expanded: '%s')", unmatched$expanded_name[i]))
            }
            cat(sprintf(" (std: '%s')\n", unmatched$standardized[i]))
            
            # Suggest closest matches
            suggestions <- find_closest_matches(unmatched$standardized[i], marker_types, n = 3)
            if (length(suggestions) > 0) {
                cat(sprintf("    Suggestions: %s\n", paste(suggestions, collapse = ", ")))
            }
        }
        cat("\n")
    }
    
    # Show fuzzy matches for verification
    if (n_fuzzy > 0) {
        cat("FUZZY MATCHES (verify these are correct):\n")
        fuzzy <- results[results$match_type == "fuzzy", ]
        for (i in 1:min(10, nrow(fuzzy))) {
            cat(sprintf("  '%s' -> '%s' (%d markers)\n",
                        fuzzy$annotated_type[i],
                        fuzzy$matched_to[i],
                        fuzzy$n_markers[i]))
        }
        if (nrow(fuzzy) > 10) {
            cat(sprintf("  ... and %d more\n", nrow(fuzzy) - 10))
        }
    }
    
    cat(paste(rep("=", 70), collapse=""), "\n")
    
    return(results)
}

#' Find closest matches using edit distance
#' @param query Standardized query string
#' @param candidates Vector of candidate strings
#' @param n Number of suggestions to return
#' @return Vector of closest matches
find_closest_matches <- function(query, candidates, n = 3) {
    if (length(candidates) == 0) return(character(0))
    
    distances <- adist(query, candidates)[1, ]
    sorted_idx <- order(distances)
    
    # Return top n with reasonable distance
    max_dist <- nchar(query) * 0.5
    good_matches <- sorted_idx[distances[sorted_idx] <= max_dist]
    
    return(head(candidates[good_matches], n))
}

#' Load Marker Gene Database with Flexible Format Support
#' 
#' Supports two formats:
#' 1. Long format: cell_type, gene (one gene per row)
#' 2. Wide format: cell_type, marker_genes (semicolon-separated genes)
#'
#' @param marker_path Path to CSV file
#' @param species Species to filter for (human/mouse)
#' @param organ Optional organ/tissue filter
#' @return List with $markers (named list) and $species
load_marker_database <- function(marker_path, species = "human", organ = NULL) {
    if (is.null(marker_path) || !file.exists(marker_path)) {
        cat("[MPS] Marker database file not found or not specified.\n")
        return(NULL)
    }
    
    cat("[MPS] Loading marker database from:", marker_path, "\n")
    
    tryCatch({
        # Read the CSV file
        marker_df <- read.csv(marker_path, stringsAsFactors = FALSE, 
                              na.strings = c("", "NA", "N/A"))
        
        # Standardize column names (lowercase, remove spaces/underscores)
        orig_colnames <- colnames(marker_df)
        # FIX: Use proper regex - gsub with perl=FALSE treats \s differently
        # Replace underscores and spaces, then lowercase
        std_colnames <- tolower(orig_colnames)
        std_colnames <- gsub("[_ ]+", "", std_colnames)  # Remove underscores and spaces only
        colnames(marker_df) <- std_colnames
        
        cat("[MPS] Original columns:", paste(orig_colnames, collapse = ", "), "\n")
        cat("[MPS] Standardized columns:", paste(colnames(marker_df), collapse = ", "), "\n")
        
        # Map common column name variations
        # This handles: marker_genes -> markergenes, cell_type -> celltype, etc.
        col_mapping <- list(
            species = c("species", "organism"),
            organ = c("organ", "tissue", "tissuetype"),
            celltype = c("celltype", "cell_type", "celltypes", "annotation"),
            markergenes = c("markergenes", "marker_genes", "markers", "genes"),
            gene = c("gene", "genename", "genesymbol", "symbol"),
            genecount = c("genecount", "gene_count", "count", "ngenes")
        )
        
        # Detect which standard columns are present
        has_marker_genes <- "markergenes" %in% colnames(marker_df)
        has_gene <- "gene" %in% colnames(marker_df)
        has_cell_type <- "celltype" %in% colnames(marker_df)
        
        cat("[MPS] Original columns:", paste(orig_colnames, collapse = ", "), "\n")
        cat("[MPS] Standardized columns:", paste(colnames(marker_df), collapse = ", "), "\n")
        
        # Detect format: wide (markergenes) or long (gene)
        has_marker_genes <- "markergenes" %in% colnames(marker_df)
        has_gene <- "gene" %in% colnames(marker_df)
        has_cell_type <- "celltype" %in% colnames(marker_df)
        has_species <- "species" %in% colnames(marker_df)
        
        cat("[MPS] Column detection:\n")
        cat(sprintf("   - celltype: %s\n", has_cell_type))
        cat(sprintf("   - markergenes: %s\n", has_marker_genes))
        cat(sprintf("   - gene: %s\n", has_gene))
        cat(sprintf("   - species: %s\n", has_species))
        
        if (!has_cell_type) {
            cat("[MPS ERROR] 'cell_type' column not found in marker file.\n")
            cat("   Available columns:", paste(colnames(marker_df), collapse = ", "), "\n")
            return(NULL)
        }
        
        # Filter by species if column exists
        if (has_species) {
            species_std <- tolower(trimws(species))
            marker_df$species <- tolower(trimws(marker_df$species))
            
            before_filter <- nrow(marker_df)
            marker_df <- marker_df[marker_df$species == species_std, ]
            cat(sprintf("[MPS] Filtered by species '%s': %d -> %d rows\n", 
                        species, before_filter, nrow(marker_df)))
        }
        
        # Filter by organ if specified and column exists
        if (!is.null(organ) && "organ" %in% colnames(marker_df)) {
            organ_std <- standardize_celltype_name(organ)
            marker_df$organ_std <- sapply(marker_df$organ, standardize_celltype_name)
            
            before_filter <- nrow(marker_df)
            marker_df <- marker_df[marker_df$organ_std == organ_std, ]
            cat(sprintf("[MPS] Filtered by organ '%s': %d -> %d rows\n", 
                        organ, before_filter, nrow(marker_df)))
        }
        
        if (nrow(marker_df) == 0) {
            cat("[MPS WARNING] No markers remaining after filtering.\n")
            return(NULL)
        }
        
        # Process based on format
        marker_dict <- list()
        
        if (has_marker_genes) {
            # WIDE FORMAT: markergenes column with semicolon-separated genes
            cat("[MPS] Detected WIDE format (marker_genes column with ';' separator)\n")
            
            for (i in seq_len(nrow(marker_df))) {
                cell_type_raw <- marker_df$celltype[i]
                genes_raw <- marker_df$markergenes[i]
                
                if (is.na(cell_type_raw) || is.na(genes_raw)) next
                if (trimws(cell_type_raw) == "" || trimws(genes_raw) == "") next
                
                # Standardize cell type name
                cell_type_std <- standardize_celltype_name(cell_type_raw)
                
                # Split genes by semicolon, comma, or pipe
                genes <- unlist(strsplit(genes_raw, "[;,|]+"))
                genes <- trimws(genes)
                genes <- genes[genes != ""]
                
                # Standardize gene names for the species
                genes_std <- sapply(genes, function(g) {
                    standardize_gene_name(g, species = species)
                })
                genes_std <- unique(genes_std[genes_std != ""])
                
                if (length(genes_std) > 0) {
                    if (cell_type_std %in% names(marker_dict)) {
                        # Merge with existing genes
                        marker_dict[[cell_type_std]] <- unique(c(
                            marker_dict[[cell_type_std]], 
                            genes_std
                        ))
                    } else {
                        marker_dict[[cell_type_std]] <- genes_std
                    }
                }
            }
            
        } else if (has_gene) {
            # LONG FORMAT: one gene per row
            cat("[MPS] Detected LONG format (gene column)\n")
            
            for (i in seq_len(nrow(marker_df))) {
                cell_type_raw <- marker_df$celltype[i]
                gene_raw <- marker_df$gene[i]
                
                if (is.na(cell_type_raw) || is.na(gene_raw)) next
                
                cell_type_std <- standardize_celltype_name(cell_type_raw)
                gene_std <- standardize_gene_name(gene_raw, species = species)
                
                if (gene_std != "") {
                    if (cell_type_std %in% names(marker_dict)) {
                        marker_dict[[cell_type_std]] <- unique(c(
                            marker_dict[[cell_type_std]], gene_std
                        ))
                    } else {
                        marker_dict[[cell_type_std]] <- gene_std
                    }
                }
            }
            
        } else {
            cat("[MPS ERROR] Could not detect format. Need 'gene' or 'marker_genes' column.\n")
            cat("   Available columns:", paste(colnames(marker_df), collapse = ", "), "\n")
            return(NULL)
        }
        
        # Remove empty entries
        marker_dict <- marker_dict[sapply(marker_dict, length) > 0]
        
        cat("[MPS] Successfully loaded marker database:\n")
        cat("   - Cell types:", length(marker_dict), "\n")
        cat("   - Total unique markers:", length(unique(unlist(marker_dict))), "\n")
        
        # Show sample entries
        if (length(marker_dict) > 0) {
            cat("   - Sample entries:\n")
            sample_types <- head(names(marker_dict), 3)
            for (ct in sample_types) {
                genes_preview <- paste(head(marker_dict[[ct]], 5), collapse = ", ")
                if (length(marker_dict[[ct]]) > 5) {
                    genes_preview <- paste0(genes_preview, ", ... (", 
                                           length(marker_dict[[ct]]), " total)")
                }
                cat("      ", ct, ":", genes_preview, "\n")
            }
        }
        
        return(list(
            markers = marker_dict,
            species = species
        ))
        
    }, error = function(e) {
        cat("[MPS ERROR] Failed to load marker database:", conditionMessage(e), "\n")
        return(NULL)
    })
}

#' Calculate MPS (Marker Prior Score) for a single cluster/cell type
#' @description Calculates precision, recall, and F1 score for marker genes
#' @param discovered_genes Vector of discovered DEGs (top N)
#' @param canonical_markers Vector of canonical marker genes for this cell type
#' @param all_genes_in_data Vector of all genes present in the dataset
#' @return List with precision, recall, f1, and overlap details
calculate_mps_single <- function(discovered_genes, canonical_markers, all_genes_in_data = NULL) {
    
    # Handle edge cases
    if (is.null(discovered_genes) || length(discovered_genes) == 0) {
        return(list(
            precision = 0, recall = 0, f1 = 0,
            n_discovered = 0, n_canonical = length(canonical_markers),
            n_overlap = 0, overlapping_genes = character(0)
        ))
    }
    
    if (is.null(canonical_markers) || length(canonical_markers) == 0) {
        return(list(
            precision = NA, recall = NA, f1 = NA,
            n_discovered = length(discovered_genes), n_canonical = 0,
            n_overlap = NA, overlapping_genes = character(0),
            note = "No canonical markers available"
        ))
    }
    
    # Filter canonical markers to those present in the data (if provided)
    if (!is.null(all_genes_in_data)) {
        canonical_markers <- intersect(canonical_markers, all_genes_in_data)
        if (length(canonical_markers) == 0) {
            return(list(
                precision = NA, recall = NA, f1 = NA,
                n_discovered = length(discovered_genes), n_canonical = 0,
                n_overlap = NA, overlapping_genes = character(0),
                note = "No canonical markers found in dataset"
            ))
        }
    }
    
    # Calculate overlap
    overlapping_genes <- intersect(discovered_genes, canonical_markers)
    n_overlap <- length(overlapping_genes)
    
    # Precision: What fraction of discovered genes are canonical markers?
    precision <- n_overlap / length(discovered_genes)
    
    # Recall: What fraction of canonical markers were discovered?
    recall <- n_overlap / length(canonical_markers)
    
    # F1 Score: Harmonic mean of precision and recall
    if (precision + recall > 0) {
        f1 <- 2 * precision * recall / (precision + recall)
    } else {
        f1 <- 0
    }
    
    return(list(
        precision = precision,
        recall = recall,
        f1 = f1,
        n_discovered = length(discovered_genes),
        n_canonical = length(canonical_markers),
        n_overlap = n_overlap,
        overlapping_genes = overlapping_genes
    ))
}

#' Standardize a single gene name based on species
#' @param gene_name Single gene name string
#' @param species "human" or "mouse"
#' @return Standardized gene name
standardize_gene_name <- function(gene_name, species = "human") {
    if (is.na(gene_name) || gene_name == "") return("")
    
    gene_name <- trimws(gene_name)
    
    if (species == "human") {
        # Human genes: ALL UPPERCASE
        return(toupper(gene_name))
    } else if (species == "mouse") {
        # Mouse genes: First letter uppercase, rest lowercase
        # Exception: Mitochondrial genes keep "mt-" prefix
        if (grepl("^(MT|Mt|mt)[-._:]", gene_name, ignore.case = TRUE)) {
            parts <- strsplit(gene_name, "[-._:]")[[1]]
            if (length(parts) >= 2) {
                return(paste0("mt-", tools::toTitleCase(tolower(parts[2]))))
            } else {
                return(paste0("mt-", tools::toTitleCase(tolower(gsub("^(MT|Mt|mt)[-._:]?", "", gene_name)))))
            }
        } else {
            return(tools::toTitleCase(tolower(gene_name)))
        }
    }
    return(gene_name)
}

#' Calculate enhanced DEG ranking with expression and specificity weighting
#' 
#' @description Ranks DEGs using a composite score that considers:
#'   - log2 fold change (differential expression strength)
#'   - Average expression level (gene reliability)
#'   - Detection rate specificity (pct.1 - pct.2)
#'   
#' @param markers_df Output from FindAllMarkers()
#' @param seurat_obj Seurat object (for calculating avg expression if needed)
#' @param ranking_method "original" (log2FC only) or "composite" (weighted score)
#' @param weights Named vector: c(fc=0.4, expr=0.3, pct=0.3)
#' @param n_top_genes Number of top genes to return per cluster
#' @param species Species for gene name standardization ("human" or "mouse")
#' @param verbose Print progress messages
#' @return Enhanced markers dataframe with composite_score and enhanced_rank columns
calculate_enhanced_deg_ranking <- function(markers_df, 
                                            seurat_obj = NULL,
                                            ranking_method = "composite",
                                            weights = c(fc = 0.4, expr = 0.3, pct = 0.3),
                                            n_top_genes = 200,
                                            species = "human",
                                            verbose = FALSE) {
    
    if (is.null(markers_df) || nrow(markers_df) == 0) {
        if (verbose) cat("[DEG RANKING] No markers provided. Returning empty dataframe.\n")
        return(markers_df)
    }
    
    if (verbose) {
        cat(sprintf("[DEG RANKING] Method: %s\n", ranking_method))
        if (ranking_method == "composite") {
            cat(sprintf("[DEG RANKING] Weights: FC=%.2f, Expr=%.2f, Pct=%.2f\n", 
                        weights["fc"], weights["expr"], weights["pct"]))
        }
    }
    
    # === ORIGINAL METHOD: Rank by log2FC only ===
    if (ranking_method == "original") {
        markers_ranked <- markers_df %>%
            dplyr::filter(avg_log2FC > 0) %>%
            dplyr::group_by(cluster) %>%
            dplyr::arrange(cluster, dplyr::desc(avg_log2FC)) %>%
            dplyr::mutate(enhanced_rank = dplyr::row_number()) %>%
            dplyr::filter(enhanced_rank <= n_top_genes) %>%
            dplyr::ungroup()
        
        if (verbose) {
            cat(sprintf("[DEG RANKING] Original ranking: %d genes across %d clusters\n",
                        nrow(markers_ranked), length(unique(markers_ranked$cluster))))
        }
        
        return(markers_ranked)
    }
    
    # === COMPOSITE METHOD: Weighted score ===
    if (ranking_method == "composite") {
        
        # Step 1: Check if avg_expression column exists
        if (!"avg_expression" %in% colnames(markers_df)) {
            
            if (!is.null(seurat_obj)) {
                # Calculate average expression per gene per cluster
                if (verbose) cat("[DEG RANKING] Calculating average expression levels...\n")
                
                expr_matrix <- GetAssayData(seurat_obj, assay = "RNA", layer = "data")
                cluster_ids <- Idents(seurat_obj)
                
                # Pre-compute cluster membership
                cluster_cells <- split(names(cluster_ids), cluster_ids)
                
                # Calculate mean expression for each gene-cluster pair in markers_df
                markers_df$avg_expression <- sapply(seq_len(nrow(markers_df)), function(i) {
                    gene <- markers_df$gene[i]
                    cl <- as.character(markers_df$cluster[i])
                    
                    if (gene %in% rownames(expr_matrix) && cl %in% names(cluster_cells)) {
                        cells <- cluster_cells[[cl]]
                        if (length(cells) > 0) {
                            return(mean(expr_matrix[gene, cells, drop = TRUE], na.rm = TRUE))
                        }
                    }
                    return(0)
                })
                
            } else {
                # Fallback: use pct.1 as proxy for expression level
                if (verbose) cat("[DEG RANKING] No Seurat object provided. Using pct.1 as expression proxy.\n")
                markers_df$avg_expression <- markers_df$pct.1
            }
        }
        
        # Step 2: Calculate composite score per cluster
        markers_ranked <- markers_df %>%
            dplyr::filter(avg_log2FC > 0) %>%
            dplyr::group_by(cluster) %>%
            dplyr::mutate(
                # Normalize log2FC to 0-1 within cluster
                fc_min = min(avg_log2FC, na.rm = TRUE),
                fc_max = max(avg_log2FC, na.rm = TRUE),
                fc_norm = ifelse(fc_max > fc_min, 
                                 (avg_log2FC - fc_min) / (fc_max - fc_min + 1e-10),
                                 0.5),
                
                # Normalize expression to 0-1 within cluster
                expr_min = min(avg_expression, na.rm = TRUE),
                expr_max = max(avg_expression, na.rm = TRUE),
                expr_norm = ifelse(expr_max > expr_min,
                                   (avg_expression - expr_min) / (expr_max - expr_min + 1e-10),
                                   0.5),
                
                # Calculate detection rate specificity (pct.1 - pct.2)
                pct_diff = pct.1 - pct.2,
                pct_min = min(pct_diff, na.rm = TRUE),
                pct_max = max(pct_diff, na.rm = TRUE),
                pct_norm = ifelse(pct_max > pct_min,
                                  (pct_diff - pct_min) / (pct_max - pct_min + 1e-10),
                                  0.5),
                
                # Calculate composite score
                composite_score = (weights["fc"] * fc_norm) + 
                                  (weights["expr"] * expr_norm) + 
                                  (weights["pct"] * pct_norm)
            ) %>%
            dplyr::arrange(cluster, dplyr::desc(composite_score)) %>%
            dplyr::mutate(enhanced_rank = dplyr::row_number()) %>%
            dplyr::filter(enhanced_rank <= n_top_genes) %>%
            # Clean up intermediate columns
            dplyr::select(-fc_min, -fc_max, -expr_min, -expr_max, -pct_min, -pct_max) %>%
            dplyr::ungroup()
        
        if (verbose) {
            cat(sprintf("[DEG RANKING] Composite ranking: %d genes across %d clusters\n",
                        nrow(markers_ranked), length(unique(markers_ranked$cluster))))
            
            # Show example of ranking change
            if (nrow(markers_ranked) > 0) {
                example_cluster <- markers_ranked$cluster[1]
                example_genes <- markers_ranked %>%
                    dplyr::filter(cluster == example_cluster) %>%
                    head(5)
                
                cat(sprintf("[DEG RANKING] Top 5 for cluster '%s':\n", example_cluster))
                for (i in 1:nrow(example_genes)) {
                    cat(sprintf("   %d. %s (FC=%.2f, expr=%.2f, pct_diff=%.2f, score=%.3f)\n",
                                i, 
                                example_genes$gene[i],
                                example_genes$avg_log2FC[i],
                                example_genes$avg_expression[i],
                                example_genes$pct_diff[i],
                                example_genes$composite_score[i]))
                }
            }
        }
        
        return(markers_ranked)
    }
    
    # Fallback: return original
    warning(sprintf("Unknown ranking_method: %s. Using original.", ranking_method))
    return(markers_df)
}

#' Standardize a single cell type name for matching
#' @param celltype_name Single cell type name
#' @return Standardized (lowercase, no special chars) cell type name
standardize_celltype_name <- function(celltype_name) {
    if (is.na(celltype_name) || celltype_name == "") return("")
    
    # Trim whitespace
    ct <- trimws(celltype_name)
    
    # Convert to lowercase
    ct <- tolower(ct)
    
    # Replace separators with nothing (compact form)
    ct <- gsub("[_\\s.,-]+", "", ct)
    
    # Remove special characters
    ct <- gsub("[^a-z0-9]", "", ct)
    
    return(ct)
}

#' Wrapper for backward compatibility - calls enhanced matching
#' @param query_type Query cell type name
#' @param reference_types Vector of reference cell type names
#' @param expand_abbreviations Whether to expand abbreviations
#' @param min_similarity Minimum similarity threshold
#' @param verbose Print matching details
#' @return Matched cell type name (string) or NULL
find_best_celltype_match <- function(query_type, 
                                      reference_types,
                                      expand_abbreviations = TRUE,
                                      min_similarity = 0.4,
                                      verbose = FALSE) {
    
    # Call the enhanced matching function
    result <- find_best_celltype_match_enhanced(
        query_type = query_type,
        reference_types = reference_types,
        min_similarity = min_similarity,
        expand_abbreviations = expand_abbreviations,
        use_hierarchy = TRUE,
        verbose = verbose
    )
    
    # Return just the matched type (for backward compatibility)
    return(result$matched_type)
}

#' Extract meaningful tokens from a cell type name
#' @param celltype_name Original cell type name (not standardized)
#' @return Vector of lowercase tokens
extract_celltype_tokens <- function(celltype_name) {
    if (is.null(celltype_name) || celltype_name == "") return(character(0))
    
    # Convert to lowercase
    ct <- tolower(celltype_name)
    
    # Split on spaces, underscores, hyphens
    tokens <- unlist(strsplit(ct, "[\\s_-]+"))
    
    # Remove very short tokens and common stop words
    stop_words <- c("cell", "cells", "the", "a", "an", "of", "and", "or", "type")
    tokens <- tokens[nchar(tokens) >= 2]
    
    # Keep "cell" as it can be important for matching
    meaningful_tokens <- tokens[!tokens %in% c("the", "a", "an", "of", "and", "or")]
    
    return(unique(meaningful_tokens))
}

#' Extract tokens from standardized (compacted) cell type name
#' @param std_name Standardized cell type name (no spaces, lowercase)
#' @return Vector of likely tokens
extract_celltype_tokens_from_std <- function(std_name) {
    if (is.null(std_name) || std_name == "") return(character(0))
    
    # Common cell type terms to look for
    known_terms <- c(
        "adipocyte", "adipose", "progenitor", "precursor", "stem", "cell",
        "macrophage", "monocyte", "lymphocyte", "tcell", "bcell", "nkcell",
        "dendritic", "neutrophil", "eosinophil", "basophil", "mast",
        "epithelial", "endothelial", "fibroblast", "myocyte", "neuron",
        "astrocyte", "oligodendrocyte", "microglia", "keratinocyte",
        "hepatocyte", "cardiomyocyte", "smooth", "muscle", "stromal",
        "mesenchymal", "derived", "activated", "naive", "memory",
        "regulatory", "helper", "cytotoxic", "natural", "killer",
        "resident", "infiltrating", "circulating", "mature", "immature",
        "cd4", "cd8", "cd14", "cd16", "cd34", "cd45"
    )
    
    found_tokens <- c()
    remaining <- std_name
    
    # Greedy matching of known terms
    for (term in known_terms[order(-nchar(known_terms))]) {  # Longest first
        if (grepl(term, remaining, fixed = TRUE)) {
            found_tokens <- c(found_tokens, term)
            remaining <- gsub(term, "", remaining, fixed = TRUE)
        }
    }
    
    # Add any remaining chunks if long enough
    if (nchar(remaining) >= 3) {
        found_tokens <- c(found_tokens, remaining)
    }
    
    return(unique(found_tokens))
}

#' Tissue/Region Prefixes to Strip Before Matching
#' These are commonly prepended by CellTypist and other annotation tools
TISSUE_PREFIXES <- c(
    # Brain regions
    "Brain ", "Cerebellum ", "Cortex ", "Hippocampus ", "Hypothalamus ",
    "Striatum ", "Thalamus ", "Midbrain ", "Hindbrain ", "Forebrain ",
    "Brainstem ", "Spinal ", "Retina ", "Cerebral ",
    
    # Developmental stages
    "Developing ", "Adult ", "Fetal ", "Mature ", "Immature ",
    "Embryonic ", "Postnatal ", "Neonatal ", "Aged ",
    
    # Other tissues (for general use)
    "Liver ", "Lung ", "Heart ", "Kidney ", "Spleen ", "Pancreas ",
    "Intestine ", "Colon ", "Stomach ", "Skin ", "Bone ", "Blood ",
    "Lymph ", "Thymus ", "Adipose ", "Muscle ", "Testis ", "Ovary ",
    
    # Generic prefixes
    "Primary ", "Secondary ", "Resident ", "Infiltrating ", "Circulating "
)

#' Hierarchical Cell Type Mappings (child -> parent)
#' Used as fallback when specific type not found
CELLTYPE_HIERARCHY <- list(
    # Neurons - specific to general
    "layer 2/3 intratelencephalic-projecting excitatory neuron" = "excitatory neuron",
    "layer 4 intratelencephalic-projecting excitatory neuron" = "excitatory neuron",
    "layer 5 intratelencephalic-projecting excitatory neuron" = "excitatory neuron",
    "layer 5 extratelencephalic-projecting excitatory neuron" = "excitatory neuron",
    "layer 6 intratelencephalic-projecting excitatory neuron" = "excitatory neuron",
    "layer 6 corticothalamic-projecting excitatory neuron" = "excitatory neuron",
    "excitatory neuron" = "neuron",
    
    # Interneurons - specific to general
    "parvalbumin+ gabaergic interneuron" = "gabaergic interneuron",
    "somatostatin+ gabaergic interneuron" = "gabaergic interneuron",
    "vasoactive intestinal peptide+ gabaergic interneuron" = "gabaergic interneuron",
    "lamp5+ gabaergic interneuron" = "gabaergic interneuron",
    "chandelier cell" = "gabaergic interneuron",
    "gabaergic interneuron" = "interneuron",
    "interneuron" = "neuron",
    
    # Glia
    "oligodendrocyte precursor cell" = "oligodendrocyte",
    "mature oligodendrocyte" = "oligodendrocyte",
    "fibrous astrocyte" = "astrocyte",
    "protoplasmic astrocyte" = "astrocyte",
    "reactive astrocyte" = "astrocyte",
    
    # Immune
    "microglia and perivascular macrophage" = "microglia",
    "activated microglia" = "microglia",
    "homeostatic microglia" = "microglia",
    "border-associated macrophage" = "macrophage",
    "perivascular macrophage" = "macrophage",
    
    # Vascular
    "vascular and leptomeningeal cell" = "vascular cell",
    "arterial endothelial cell" = "endothelial cell",
    "venous endothelial cell" = "endothelial cell",
    "capillary endothelial cell" = "endothelial cell",
    "endothelial cell" = "vascular cell",
    "pericyte" = "vascular cell",
    "smooth muscle cell" = "vascular cell"
)

#' Strip tissue/region prefixes from cell type name
#' @param celltype_name Original cell type name
#' @return Cell type name with prefix removed (if any)
strip_tissue_prefix <- function(celltype_name) {
    if (is.null(celltype_name) || is.na(celltype_name) || celltype_name == "") {
        return(celltype_name)
    }
    
    result <- celltype_name
    
    # Try case-insensitive prefix stripping
    for (prefix in TISSUE_PREFIXES) {
        # Case-insensitive check
        if (startsWith(tolower(result), tolower(prefix))) {
            result <- substring(result, nchar(prefix) + 1)
            result <- trimws(result)
            break  # Only strip one prefix
        }
    }
    
    return(result)
}

#' Get parent cell type from hierarchy
#' @param celltype_name Standardized cell type name
#' @return Parent cell type or NULL if no parent
get_parent_celltype <- function(celltype_name) {
    if (is.null(celltype_name) || celltype_name == "") {
        return(NULL)
    }
    
    ct_lower <- tolower(celltype_name)
    
    # Direct lookup
    if (ct_lower %in% names(CELLTYPE_HIERARCHY)) {
        return(CELLTYPE_HIERARCHY[[ct_lower]])
    }
    
    # Partial match (for standardized names)
    for (child in names(CELLTYPE_HIERARCHY)) {
        if (grepl(child, ct_lower, fixed = TRUE) || grepl(ct_lower, child, fixed = TRUE)) {
            return(CELLTYPE_HIERARCHY[[child]])
        }
    }
    
    return(NULL)
}

#' Enhanced cell type matching with full robustness
#' 
#' @description Implements 8-strategy matching pipeline:
#'   1. Exact match (after standardization)
#'   2. Prefix-stripped match
#'   3. Abbreviation expansion match
#'   4. Synonym replacement match
#'   5. Substring/contains match
#'   6. Token-based Jaccard similarity
#'   7. Hierarchical parent fallback
#'   8. Levenshtein distance fallback
#'
#' @param query_type Query cell type name (from annotation)
#' @param reference_types Vector of reference cell type names (from marker DB)
#' @param min_similarity Minimum similarity score to accept (0-1)
#' @param expand_abbreviations Whether to expand abbreviations
#' @param use_hierarchy Whether to try hierarchical parent matching
#' @param verbose Print matching details
#' @return List with: matched_type, match_method, confidence, details
find_best_celltype_match_enhanced <- function(query_type, 
                                               reference_types, 
                                               min_similarity = 0.4,
                                               expand_abbreviations = TRUE,
                                               use_hierarchy = TRUE,
                                               verbose = FALSE) {
    
    # Handle edge cases
    if (length(reference_types) == 0 || is.null(query_type) || 
        is.na(query_type) || trimws(query_type) == "") {
        return(list(
            matched_type = NULL,
            match_method = "none",
            confidence = 0,
            details = "Empty query or no reference types"
        ))
    }
    
    query_original <- trimws(query_type)
    
    # Pre-compute standardized reference types
    ref_types_std <- sapply(reference_types, standardize_celltype_name)
    names(ref_types_std) <- reference_types
    
    # === STRATEGY 1: EXACT MATCH (after standardization) ===
    query_std <- standardize_celltype_name(query_original)
    
    if (query_std %in% ref_types_std) {
        matched <- names(ref_types_std)[ref_types_std == query_std][1]
        if (verbose) cat(sprintf("   [EXACT] '%s' -> '%s'\n", query_original, matched))
        return(list(
            matched_type = matched,
            match_method = "exact",
            confidence = 1.0,
            details = sprintf("Exact match after standardization")
        ))
    }
    
    # === STRATEGY 2: PREFIX-STRIPPED MATCH ===
    query_stripped <- strip_tissue_prefix(query_original)
    
    if (query_stripped != query_original) {
        query_stripped_std <- standardize_celltype_name(query_stripped)
        
        if (query_stripped_std %in% ref_types_std) {
            matched <- names(ref_types_std)[ref_types_std == query_stripped_std][1]
            if (verbose) cat(sprintf("   [PREFIX] '%s' -> '%s' (stripped: '%s')\n", 
                                     query_original, matched, query_stripped))
            return(list(
                matched_type = matched,
                match_method = "prefix_stripped",
                confidence = 0.95,
                details = sprintf("Matched after stripping prefix from '%s'", query_original)
            ))
        }
        
        # Also try matching stripped query against stripped references
        for (i in seq_along(reference_types)) {
            ref_stripped <- strip_tissue_prefix(reference_types[i])
            ref_stripped_std <- standardize_celltype_name(ref_stripped)
            
            if (query_stripped_std == ref_stripped_std) {
                if (verbose) cat(sprintf("   [PREFIX-BOTH] '%s' -> '%s'\n", 
                                         query_original, reference_types[i]))
                return(list(
                    matched_type = reference_types[i],
                    match_method = "prefix_stripped_both",
                    confidence = 0.9,
                    details = sprintf("Both stripped: '%s' == '%s'", query_stripped, ref_stripped)
                ))
            }
        }
    }
    
    # === STRATEGY 3: ABBREVIATION EXPANSION ===
    if (expand_abbreviations) {
        query_expanded <- expand_celltype_abbreviation(query_original)
        
        if (query_expanded != query_original) {
            query_expanded_std <- standardize_celltype_name(query_expanded)
            
            if (query_expanded_std %in% ref_types_std) {
                matched <- names(ref_types_std)[ref_types_std == query_expanded_std][1]
                if (verbose) cat(sprintf("   [ABBREV] '%s' -> '%s' (expanded: '%s')\n", 
                                         query_original, matched, query_expanded))
                return(list(
                    matched_type = matched,
                    match_method = "abbreviation_expanded",
                    confidence = 0.95,
                    details = sprintf("Expanded '%s' to '%s'", query_original, query_expanded)
                ))
            }
            
            # Try prefix stripping on expanded name
            query_expanded_stripped <- strip_tissue_prefix(query_expanded)
            query_expanded_stripped_std <- standardize_celltype_name(query_expanded_stripped)
            
            if (query_expanded_stripped_std %in% ref_types_std) {
                matched <- names(ref_types_std)[ref_types_std == query_expanded_stripped_std][1]
                if (verbose) cat(sprintf("   [ABBREV+PREFIX] '%s' -> '%s'\n", 
                                         query_original, matched))
                return(list(
                    matched_type = matched,
                    match_method = "abbreviation_prefix_stripped",
                    confidence = 0.9,
                    details = sprintf("Expanded then stripped: '%s' -> '%s' -> '%s'", 
                                     query_original, query_expanded, query_expanded_stripped)
                ))
            }
        }
    }
    
    # === STRATEGY 4: SYNONYM REPLACEMENT ===
    synonyms <- list(
        "neuron" = c("neuronal", "nerve", "neural"),
        "astrocyte" = c("astro", "astroglia", "astroglial"),
        "oligodendrocyte" = c("oligo", "olig", "oligodendroglia"),
        "microglia" = c("microglial", "microgliocyte"),
        "macrophage" = c("mf", "mac", "mφ", "histiocyte"),
        "endothelial" = c("endothelium", "endo", "ec"),
        "fibroblast" = c("fibro", "fb"),
        "epithelial" = c("epithelium", "epi"),
        "progenitor" = c("precursor", "prog"),
        "gabaergic" = c("gaba", "inhibitory", "gabanergic"),
        "glutamatergic" = c("glut", "excitatory"),
        "interneuron" = c("inter", "inhibitory neuron"),
        "parvalbumin" = c("pvalb", "pv", "parv"),
        "somatostatin" = c("sst", "som"),
        "vip" = c("vasoactive intestinal peptide", "vasoactiveintestinalpeptide")
    )
    
    # Try replacing synonyms in query
    query_lower <- tolower(query_original)
    query_variants <- c(query_std)
    
    for (canonical in names(synonyms)) {
        for (syn in synonyms[[canonical]]) {
            if (grepl(syn, query_lower, fixed = TRUE)) {
                variant <- gsub(syn, canonical, query_lower, fixed = TRUE)
                variant_std <- standardize_celltype_name(variant)
                query_variants <- c(query_variants, variant_std)
            }
            if (grepl(canonical, query_lower, fixed = TRUE)) {
                variant <- gsub(canonical, syn, query_lower, fixed = TRUE)
                variant_std <- standardize_celltype_name(variant)
                query_variants <- c(query_variants, variant_std)
            }
        }
    }
    
    query_variants <- unique(query_variants)
    
    for (variant in query_variants) {
        if (variant %in% ref_types_std && variant != query_std) {
            matched <- names(ref_types_std)[ref_types_std == variant][1]
            if (verbose) cat(sprintf("   [SYNONYM] '%s' -> '%s'\n", query_original, matched))
            return(list(
                matched_type = matched,
                match_method = "synonym",
                confidence = 0.85,
                details = sprintf("Synonym variant matched: '%s'", variant)
            ))
        }
    }
    
    # === STRATEGY 5: SUBSTRING/CONTAINS MATCH ===
    # Use the most processed version (stripped + expanded if available)
    query_for_substring <- query_std
    if (exists("query_expanded_stripped_std", inherits = FALSE)) {
        query_for_substring <- query_expanded_stripped_std
    } else if (exists("query_stripped_std", inherits = FALSE) && query_stripped_std != query_std) {
        query_for_substring <- query_stripped_std
    }
    
    for (i in seq_along(reference_types)) {
        ref_std <- ref_types_std[i]
        
        # Skip very short strings to avoid false matches
        if (nchar(query_for_substring) < 4 || nchar(ref_std) < 4) next
        
        # Query contained in reference
        if (grepl(query_for_substring, ref_std, fixed = TRUE)) {
            if (verbose) cat(sprintf("   [SUBSTRING-IN] '%s' found in '%s'\n", 
                                     query_for_substring, reference_types[i]))
            return(list(
                matched_type = reference_types[i],
                match_method = "substring_query_in_ref",
                confidence = 0.75,
                details = sprintf("'%s' contained in '%s'", query_for_substring, ref_std)
            ))
        }
        
        # Reference contained in query
        if (grepl(ref_std, query_for_substring, fixed = TRUE)) {
            if (verbose) cat(sprintf("   [SUBSTRING-CONTAINS] '%s' contains '%s'\n", 
                                     query_for_substring, reference_types[i]))
            return(list(
                matched_type = reference_types[i],
                match_method = "substring_ref_in_query",
                confidence = 0.7,
                details = sprintf("'%s' contains '%s'", query_for_substring, ref_std)
            ))
        }
    }
    
    # Also check variants
    for (variant in query_variants) {
        if (nchar(variant) < 4) next
        
        for (i in seq_along(reference_types)) {
            ref_std <- ref_types_std[i]
            if (nchar(ref_std) < 4) next
            
            if (grepl(variant, ref_std, fixed = TRUE) || grepl(ref_std, variant, fixed = TRUE)) {
                if (verbose) cat(sprintf("   [SUBSTRING-VARIANT] variant '%s' <-> '%s'\n", 
                                         variant, reference_types[i]))
                return(list(
                    matched_type = reference_types[i],
                    match_method = "substring_variant",
                    confidence = 0.65,
                    details = sprintf("Variant '%s' substring match with '%s'", variant, ref_std)
                ))
            }
        }
    }
    
    # === STRATEGY 6: TOKEN-BASED JACCARD SIMILARITY ===
    query_tokens <- extract_celltype_tokens(query_original)
    
    # Add tokens from expanded name if available
    if (expand_abbreviations) {
        expanded_tokens <- extract_celltype_tokens(expand_celltype_abbreviation(query_original))
        query_tokens <- unique(c(query_tokens, expanded_tokens))
    }
    
    # Add tokens from stripped name
    stripped_tokens <- extract_celltype_tokens(strip_tissue_prefix(query_original))
    query_tokens <- unique(c(query_tokens, stripped_tokens))
    
    best_jaccard_match <- NULL
    best_jaccard_score <- 0
    
    for (i in seq_along(reference_types)) {
        ref_tokens <- extract_celltype_tokens(reference_types[i])
        
        if (length(query_tokens) == 0 || length(ref_tokens) == 0) next
        
        intersection <- length(intersect(query_tokens, ref_tokens))
        union <- length(union(query_tokens, ref_tokens))
        
        if (union > 0) {
            jaccard <- intersection / union
            
            # Bonus for matching important biological tokens
            important_tokens <- c(
                "neuron", "astrocyte", "oligodendrocyte", "microglia",
                "interneuron", "excitatory", "inhibitory", "gabaergic",
                "glutamatergic", "progenitor", "stem", "precursor",
                "endothelial", "epithelial", "fibroblast", "macrophage"
            )
            important_matches <- sum(intersect(query_tokens, ref_tokens) %in% important_tokens)
            
            # Weighted score
            score <- jaccard + (0.15 * important_matches)
            
            if (score > best_jaccard_score) {
                best_jaccard_score <- score
                best_jaccard_match <- reference_types[i]
            }
        }
    }
    
    if (!is.null(best_jaccard_match) && best_jaccard_score >= min_similarity) {
        if (verbose) cat(sprintf("   [JACCARD] '%s' -> '%s' (score=%.3f)\n", 
                                 query_original, best_jaccard_match, best_jaccard_score))
        return(list(
            matched_type = best_jaccard_match,
            match_method = "token_jaccard",
            confidence = min(0.8, best_jaccard_score),
            details = sprintf("Jaccard similarity: %.3f", best_jaccard_score)
        ))
    }
    
    # === STRATEGY 7: HIERARCHICAL PARENT FALLBACK ===
    if (use_hierarchy) {
        # Try to find parent of query in reference
        query_for_hierarchy <- tolower(query_original)
        if (expand_abbreviations) {
            query_for_hierarchy <- tolower(expand_celltype_abbreviation(query_original))
        }
        query_for_hierarchy <- tolower(strip_tissue_prefix(query_for_hierarchy))
        
        parent <- get_parent_celltype(query_for_hierarchy)
        
        while (!is.null(parent)) {
            parent_std <- standardize_celltype_name(parent)
            
            if (parent_std %in% ref_types_std) {
                matched <- names(ref_types_std)[ref_types_std == parent_std][1]
                if (verbose) cat(sprintf("   [HIERARCHY] '%s' -> parent '%s' -> '%s'\n", 
                                         query_original, parent, matched))
                return(list(
                    matched_type = matched,
                    match_method = "hierarchical_parent",
                    confidence = 0.6,
                    details = sprintf("Matched via parent: '%s' -> '%s'", query_for_hierarchy, parent)
                ))
            }
            
            # Try next level up
            parent <- get_parent_celltype(parent)
        }
    }
    
    # === STRATEGY 8: LEVENSHTEIN DISTANCE FALLBACK ===
    best_lev_match <- NULL
    best_lev_score <- 0
    
    for (i in seq_along(reference_types)) {
        ref_std <- ref_types_std[i]
        max_len <- max(nchar(query_std), nchar(ref_std))
        
        if (max_len > 0) {
            lev_dist <- adist(query_std, ref_std)[1, 1]
            lev_sim <- 1 - (lev_dist / max_len)
            
            if (lev_sim > best_lev_score) {
                best_lev_score <- lev_sim
                best_lev_match <- reference_types[i]
            }
        }
    }
    
    # Only accept Levenshtein matches with high similarity
    if (!is.null(best_lev_match) && best_lev_score >= 0.7) {
        if (verbose) cat(sprintf("   [LEVENSHTEIN] '%s' -> '%s' (sim=%.3f)\n", 
                                 query_original, best_lev_match, best_lev_score))
        return(list(
            matched_type = best_lev_match,
            match_method = "levenshtein",
            confidence = best_lev_score * 0.7,  # Discount confidence
            details = sprintf("Levenshtein similarity: %.3f", best_lev_score)
        ))
    }
    
    # === NO MATCH FOUND ===
    if (verbose) cat(sprintf("   [NO MATCH] '%s' - no suitable match found\n", query_original))
    
    return(list(
        matched_type = NULL,
        match_method = "none",
        confidence = 0,
        details = sprintf("No match found for '%s' (best Levenshtein: %.3f to '%s')", 
                         query_original, best_lev_score, best_lev_match %||% "none")
    ))
}

#' Wrapper function for backward compatibility
#' @description Calls enhanced matching and returns just the matched type
#' @note This REPLACES the old find_best_celltype_match function
find_best_celltype_match <- function(query_type, reference_types, min_similarity = 0.5,
                                      expand_abbreviations = TRUE) {
    result <- find_best_celltype_match_enhanced(
        query_type = query_type,
        reference_types = reference_types,
        min_similarity = min_similarity,
        expand_abbreviations = expand_abbreviations,
        use_hierarchy = TRUE,
        verbose = FALSE
    )
    
    return(result$matched_type)
}

#' Test cell type matching with detailed diagnostics
#' @param seurat_obj Seurat object with annotations
#' @param marker_db Marker database from load_marker_database()
#' @param annotation_col Column containing cell type annotations
#' @return Dataframe with matching diagnostics
test_celltype_matching_enhanced <- function(seurat_obj, marker_db, 
                                             annotation_col = "ctpt_consensus_prediction") {
    
    if (is.null(marker_db) || is.null(marker_db$markers)) {
        cat("[DIAGNOSTIC] No marker database available.\n")
        return(NULL)
    }
    
    marker_types <- names(marker_db$markers)
    
    # Get unique annotated types
    if (!(annotation_col %in% colnames(seurat_obj@meta.data))) {
        cat(sprintf("[DIAGNOSTIC] Column '%s' not found.\n", annotation_col))
        return(NULL)
    }
    
    annotated_types <- unique(as.character(seurat_obj@meta.data[[annotation_col]]))
    annotated_types <- annotated_types[!is.na(annotated_types) & annotated_types != ""]
    
    cat("\n", paste(rep("=", 80), collapse=""), "\n")
    cat("ENHANCED CELL TYPE MATCHING TEST\n")
    cat(paste(rep("=", 80), collapse=""), "\n\n")
    
    cat(sprintf("Annotated cell types: %d\n", length(annotated_types)))
    cat(sprintf("Marker DB cell types: %d\n", length(marker_types)))
    cat("\n")
    
    results <- data.frame(
        annotation = character(),
        matched_to = character(),
        match_method = character(),
        confidence = numeric(),
        n_markers = integer(),
        stringsAsFactors = FALSE
    )
    
    # Test each annotated type
    for (atype in annotated_types) {
        result <- find_best_celltype_match_enhanced(
            query_type = atype,
            reference_types = marker_types,
            min_similarity = 0.4,
            expand_abbreviations = TRUE,
            use_hierarchy = TRUE,
            verbose = TRUE
        )
        
        n_markers <- 0
        if (!is.null(result$matched_type)) {
            n_markers <- length(marker_db$markers[[result$matched_type]])
        }
        
        results <- rbind(results, data.frame(
            annotation = atype,
            matched_to = result$matched_type %||% "NO MATCH",
            match_method = result$match_method,
            confidence = result$confidence,
            n_markers = n_markers,
            stringsAsFactors = FALSE
        ))
    }
    
    # Summary by method
    cat("\n", paste(rep("-", 80), collapse=""), "\n")
    cat("MATCHING SUMMARY BY METHOD:\n")
    cat(paste(rep("-", 80), collapse=""), "\n")
    
    method_summary <- table(results$match_method)
    for (method in names(method_summary)) {
        pct <- 100 * method_summary[method] / nrow(results)
        cat(sprintf("   %-30s: %3d (%.1f%%)\n", method, method_summary[method], pct))
    }
    
    n_matched <- sum(results$match_method != "none")
    cat(sprintf("\n   TOTAL MATCHED: %d / %d (%.1f%%)\n", 
                n_matched, nrow(results), 100 * n_matched / nrow(results)))
    
    cat(paste(rep("=", 80), collapse=""), "\n")
    
    return(results)
}

#' Calculate Marker Prior Score (MPS) for annotated clusters
#' 
#' UNIFIED VERSION: Works with marker_dict from load_marker_database()
#' NOW WITH ABBREVIATION EXPANSION AND ENHANCED DEG RANKING
#'
#' @param seurat_obj Seurat object with annotations and clusters
#' @param marker_db Named list from load_marker_database() OR list with $markers element
#' @param group_by Column containing cell type annotations (default: ctpt_consensus_prediction)
#' @param n_top_genes Number of top DEGs to consider per cluster
#' @param expand_abbreviations Whether to expand cell type abbreviations before matching
#' @param deg_ranking_method DEG ranking method: "original" or "composite"
#' @param deg_weights Named vector of weights for composite ranking
#' @param verbose Print detailed output
#' @return List with mean_mps, per-cluster scores, and detailed results
calculate_mps <- function(seurat_obj, 
                          marker_db, 
                          group_by = "ctpt_consensus_prediction",
                          n_top_genes = 50,
                          expand_abbreviations = TRUE,
                          deg_ranking_method = "composite",
                          deg_weights = c(fc = 0.4, expr = 0.3, pct = 0.3),
                          verbose = FALSE) {
    
    # === HANDLE DIFFERENT INPUT FORMATS ===
    marker_dict <- NULL
    species <- "human"
    
    if (is.null(marker_db)) {
        if (verbose) cat("[MPS] No marker database provided. Returning NA.\n")
        return(list(mean_mps = NA, scores = NULL, note = "No marker database"))
    }
    
    if (!is.null(marker_db$markers)) {
        marker_dict <- marker_db$markers
        species <- marker_db$species %||% "human"
    } else if (is.list(marker_db) && length(marker_db) > 0) {
        marker_dict <- marker_db
    } else {
        if (verbose) cat("[MPS] Invalid marker database format. Returning NA.\n")
        return(list(mean_mps = NA, scores = NULL, note = "Invalid marker database format"))
    }
    
    if (length(marker_dict) == 0) {
        if (verbose) cat("[MPS] Empty marker database. Returning NA.\n")
        return(list(mean_mps = NA, scores = NULL, note = "Empty marker database"))
    }
    
    # === CHECK GROUPING COLUMN ===
    if (!(group_by %in% colnames(seurat_obj@meta.data))) {
        if (verbose) cat(sprintf("[MPS] Column '%s' not found. Skipping MPS calculation.\n", group_by))
        return(list(mean_mps = NA, scores = NULL, note = "Grouping column not found"))
    }
    
    # Get unique groups
    groups <- unique(seurat_obj@meta.data[[group_by]])
    groups <- groups[!is.na(groups) & groups != ""]
    
    if (length(groups) < 2) {
        if (verbose) cat("[MPS] Need at least 2 groups for DEG analysis.\n")
        return(list(mean_mps = NA, scores = NULL, note = "Insufficient groups"))
    }
    
    if (verbose) {
        cat(sprintf("[MPS] Calculating MPS for %d groups using %d cell types from database\n",
                    length(groups), length(marker_dict)))
        cat(sprintf("[MPS] DEG ranking method: %s\n", deg_ranking_method))
    }
    
    # === SET IDENTITY FOR FINDALLMARKERS ===
    Idents(seurat_obj) <- group_by
    
    # === FIND ALL MARKERS (DEGs) ===
    all_markers <- tryCatch({
        FindAllMarkers(
            seurat_obj,
            assay = "RNA",
            only.pos = TRUE,
            min.pct = MPS_CONFIG$min_pct,
            logfc.threshold = MPS_CONFIG$logfc_threshold,
            verbose = FALSE
        )
    }, error = function(e) {
        if (verbose) cat(sprintf("[MPS] FindAllMarkers failed: %s\n", e$message))
        return(NULL)
    })
    
    if (is.null(all_markers) || nrow(all_markers) == 0) {
        if (verbose) cat("[MPS] No DEGs found.\n")
        return(list(mean_mps = NA, scores = NULL, note = "No DEGs found"))
    }
    
    # Filter by adjusted p-value
    all_markers <- all_markers %>%
        dplyr::filter(p_val_adj < MPS_CONFIG$p_adj_threshold)
    
    if (nrow(all_markers) == 0) {
        if (verbose) cat("[MPS] No significant DEGs after p-value filtering.\n")
        return(list(mean_mps = NA, scores = NULL, note = "No significant DEGs"))
    }
    
    # === APPLY ENHANCED DEG RANKING (NEW) ===
    all_markers_ranked <- calculate_enhanced_deg_ranking(
        markers_df = all_markers,
        seurat_obj = seurat_obj,
        ranking_method = deg_ranking_method,
        weights = deg_weights,
        n_top_genes = n_top_genes,
        species = species,
        verbose = verbose
    )
    
    # === STANDARDIZE DEG GENE NAMES ===
    all_markers_ranked$gene_std <- sapply(all_markers_ranked$gene, function(g) {
        standardize_gene_name(g, species = species)
    })
    
    # Get marker dict keys (standardized cell type names)
    marker_types <- names(marker_dict)
    
    # Get all genes in dataset (for filtering)
    all_genes_in_data <- rownames(seurat_obj)
    all_genes_std <- sapply(all_genes_in_data, function(g) {
        standardize_gene_name(g, species = species)
    })
    
    # === CALCULATE MPS FOR EACH GROUP ===
    mps_results <- list()
    f1_scores <- c()
    
    for (grp in groups) {
        grp_str <- as.character(grp)
        
        # Get top N DEGs for this group (now using enhanced ranking)
        grp_markers_df <- all_markers_ranked %>%
            dplyr::filter(cluster == grp) %>%
            head(n_top_genes)
        
        if (nrow(grp_markers_df) == 0) {
            mps_results[[grp_str]] <- list(
                group = grp,
                expanded_name = if (expand_abbreviations) expand_celltype_abbreviation(grp_str) else grp_str,
                matched_type = NA,
                precision = NA, recall = NA, f1 = NA,
                n_degs = 0, n_canonical = 0, n_overlap = NA,
                overlapping_genes = character(0),
                note = "No DEGs for this group"
            )
            next
        }
        
        grp_degs_std <- grp_markers_df$gene_std
        
        # === MATCH TO MARKER DATABASE (WITH ABBREVIATION EXPANSION) ===
        matched_type <- NULL
        expanded_name <- grp_str
        
        if (expand_abbreviations) {
            expanded_name <- expand_celltype_abbreviation(grp_str)
            if (expanded_name != grp_str && verbose) {
                cat(sprintf("   -> Expanded '%s' to '%s' for matching\n", grp_str, expanded_name))
            }
        }
        
        # Try to find matching cell type in database
        matched_type <- find_best_celltype_match(expanded_name, marker_types, 
                                                  expand_abbreviations = expand_abbreviations)
        
        if (is.null(matched_type)) {
            mps_results[[grp_str]] <- list(
                group = grp,
                expanded_name = expanded_name,
                matched_type = NA,
                precision = NA, recall = NA, f1 = NA,
                n_degs = length(grp_degs_std),
                n_canonical = 0, n_overlap = NA,
                overlapping_genes = character(0),
                note = "No matching cell type in marker database"
            )
            next
        }
        
        # Get canonical markers for matched type
        canonical_entry <- marker_dict[[matched_type]]
        
        # Handle both formats: direct vector or list with $genes
        if (is.list(canonical_entry) && !is.null(canonical_entry$genes)) {
            canonical_markers <- canonical_entry$genes
        } else {
            canonical_markers <- canonical_entry
        }
        
        # Standardize canonical markers
        canonical_markers_std <- sapply(canonical_markers, function(g) {
            standardize_gene_name(g, species = species)
        })
        canonical_markers_std <- unique(canonical_markers_std[canonical_markers_std != ""])
        
        # Filter to markers present in data
        canonical_in_data <- intersect(canonical_markers_std, all_genes_std)
        
        if (length(canonical_in_data) == 0) {
            mps_results[[grp_str]] <- list(
                group = grp,
                expanded_name = expanded_name,
                matched_type = matched_type,
                precision = NA, recall = NA, f1 = NA,
                n_degs = length(grp_degs_std),
                n_canonical = length(canonical_markers_std),
                n_canonical_in_data = 0, n_overlap = NA,
                overlapping_genes = character(0),
                note = "No canonical markers found in dataset"
            )
            next
        }
        
        # === CALCULATE OVERLAP METRICS ===
        overlapping_genes <- intersect(grp_degs_std, canonical_in_data)
        n_overlap <- length(overlapping_genes)
        
        # Precision: fraction of discovered DEGs that are canonical markers
        precision <- n_overlap / length(grp_degs_std)
        
        # Recall: fraction of canonical markers that were discovered
        recall <- n_overlap / length(canonical_in_data)
        
        # F1 Score
        if (precision + recall > 0) {
            f1 <- 2 * precision * recall / (precision + recall)
        } else {
            f1 <- 0
        }
        
        mps_results[[grp_str]] <- list(
            group = grp,
            expanded_name = expanded_name,
            matched_type = matched_type,
            precision = precision,
            recall = recall,
            f1 = f1,
            n_degs = length(grp_degs_std),
            n_canonical = length(canonical_markers_std),
            n_canonical_in_data = length(canonical_in_data),
            n_overlap = n_overlap,
            overlapping_genes = overlapping_genes,
            ranking_method = deg_ranking_method  # NEW: Track which method was used
        )
        
        f1_scores <- c(f1_scores, f1)
        
        if (verbose) {
            if (expanded_name != grp_str) {
                cat(sprintf("   -> %s (%s) matched to '%s': F1=%.1f%% (overlap=%d/%d)\n",
                            grp_str, expanded_name, matched_type, f1 * 100, n_overlap, length(canonical_in_data)))
            } else {
                cat(sprintf("   -> %s matched to '%s': F1=%.1f%% (overlap=%d/%d)\n",
                            grp_str, matched_type, f1 * 100, n_overlap, length(canonical_in_data)))
            }
        }
    }
    
    # === CALCULATE MEAN MPS ===
    if (length(f1_scores) > 0) {
        mean_mps <- mean(f1_scores, na.rm = TRUE) * 100  # Convert to percentage
    } else {
        mean_mps <- 0
    }
    
    if (verbose) {
        cat(sprintf("[MPS] Calculated MPS for %d/%d groups: mean=%.2f%%\n", 
                    length(f1_scores), length(groups), mean_mps))
        cat(sprintf("[MPS] DEG ranking method used: %s\n", deg_ranking_method))
    }
    
    return(list(
        mean_mps = mean_mps,
        scores = mps_results,
        n_groups_evaluated = length(f1_scores),
        n_groups_total = length(groups),
        species = species,
        abbreviations_expanded = expand_abbreviations,
        deg_ranking_method = deg_ranking_method  # NEW: Include in return
    ))
}

# ==============================================================================
# --- DATA LOADING FUNCTIONS ---
# ==============================================================================

#' Load expression data from various formats
#' @param data_path Path to data (directory for 10X, or file path for h5/rds/csv)
#' @return Expression matrix or Seurat object
load_expression_data <- function(data_path) {
    cat(sprintf("[INFO] Loading data from: %s\n", data_path))
    
    if (!file.exists(data_path) && !dir.exists(data_path)) {
        stop(sprintf("Data path does not exist: %s", data_path))
    }
    
    # Check if it's a directory (10X format)
    if (dir.exists(data_path)) {
        # Check for 10X files
        if (file.exists(file.path(data_path, "matrix.mtx")) ||
            file.exists(file.path(data_path, "matrix.mtx.gz"))) {
            cat("   -> Detected 10X Genomics format (MTX)\n")
            data <- Read10X(data.dir = data_path)
            return(data)
        }
        
        # Check for filtered_feature_bc_matrix subdirectory
        sub_path <- file.path(data_path, "filtered_feature_bc_matrix")
        if (dir.exists(sub_path)) {
            cat("   -> Detected 10X filtered_feature_bc_matrix\n")
            data <- Read10X(data.dir = sub_path)
            return(data)
        }
        
        # Check for raw_feature_bc_matrix subdirectory
        sub_path <- file.path(data_path, "raw_feature_bc_matrix")
        if (dir.exists(sub_path)) {
            cat("   -> Detected 10X raw_feature_bc_matrix\n")
            data <- Read10X(data.dir = sub_path)
            return(data)
        }
        
        stop("Directory exists but no recognized data format found")
    }
    
    # Handle file formats
    file_ext <- tolower(tools::file_ext(data_path))
    
    if (file_ext == "h5") {
        cat("   -> Detected HDF5 format\n")
        data <- Read10X_h5(filename = data_path)
        return(data)
    }
    
    if (file_ext == "rds") {
        cat("   -> Detected RDS format\n")
        data <- readRDS(data_path)
        if (inherits(data, "Seurat")) {
            cat("   -> Loaded Seurat object directly\n")
        }
        return(data)
    }
    
    if (file_ext == "csv") {
        cat("   -> Detected CSV format\n")
        data <- read.csv(data_path, row.names = 1, check.names = FALSE)
        data <- as(as.matrix(data), "dgCMatrix")
        return(data)
    }
    
    if (file_ext %in% c("tsv", "txt")) {
        cat("   -> Detected TSV/TXT format\n")
        data <- read.delim(data_path, row.names = 1, check.names = FALSE)
        data <- as(as.matrix(data), "dgCMatrix")
        return(data)
    }
    
    stop(sprintf("Unsupported file format: %s", file_ext))
}

# ==============================================================================
# --- BAYESIAN OPTIMIZATION WRAPPER ---
# ==============================================================================

#' Safe Bayesian Optimization Wrapper with Trial Replacement Logic
#' @description Maintains predefined trial count by replacing failed trials
#'              while keeping the same seed for reproducibility
#' @param FUN Objective function
#' @param bounds Parameter bounds
#' @param init_points Number of initial random points
#' @param n_iter Number of optimization iterations
#' @param acq Acquisition function type
#' @param kappa Exploration-exploitation parameter
#' @param eps Noise parameter
#' @param verbose Print verbose output
#' @param max_total_failures Maximum allowed failures before giving up
#' @param seed Random seed for reproducibility
#' @return Optimization result
safe_bayesian_optimization <- function(FUN, bounds, init_points, n_iter, 
                                       acq, kappa, eps, verbose = FALSE,
                                       max_total_failures = NULL,
                                       seed = NULL) {
    
    # Use global seed if not provided - KEEP SAME SEED THROUGHOUT
    if (is.null(seed)) {
        seed <- GLOBAL_ENV$RANDOM_SEED %||% 42
    }
    
    # Calculate total trials needed
    total_trials_needed <- init_points + n_iter
    
    # Default max failures: allow up to 2x total trials for replacement attempts
    if (is.null(max_total_failures)) {
        max_total_failures <- total_trials_needed * 2
    }
    
    # Tracking variables
    successful_trials <- 0
    failed_trials <- 0
    consecutive_failures <- 0
    max_consecutive_failures <- 10
    
    # Storage for all results
    all_params <- list()
    all_scores <- c()
    best_score <- -Inf
    best_params <- NULL
    
    cat(sprintf("\n   [BO] Target trials: %d (init=%d, iter=%d), max_failures=%d, seed=%d\n", 
                total_trials_needed, init_points, n_iter, max_total_failures, seed))
    
    # === PRE-GENERATE ALL PARAMETER SETS FOR REPRODUCIBILITY ===
    # This ensures we can reproduce results even with failures
    set.seed(seed)
    n_param_sets <- total_trials_needed + max_total_failures
    pregenerated_params <- lapply(1:n_param_sets, function(i) {
        params <- lapply(names(bounds), function(param_name) {
            b <- bounds[[param_name]]
            if (is.integer(b) || (is.numeric(b) && all(b == floor(b)))) {
                sample(seq(as.integer(b[1]), as.integer(b[2])), 1)
            } else {
                runif(1, b[1], b[2])
            }
        })
        names(params) <- names(bounds)
        params
    })
    
    # === PHASE 1: Try standard BayesianOptimization with error handling ===
    cat("   [BO] Phase 1: Attempting standard Bayesian optimization...\n")
    
    # Reset seed for BO (same seed as used for pre-generation)
    set.seed(seed)
    
    # Track scores for variance monitoring
    score_history <- c()
    
    bo_result <- tryCatch({
        
        # Create wrapper that handles individual trial failures
        trial_counter <- 0
        
        wrapped_FUN <- function(...) {
            trial_counter <<- trial_counter + 1
            
            # Attempt to evaluate the objective function
            result <- tryCatch({
                FUN(...)
            }, error = function(e) {
                cat(sprintf("     [TRIAL %d ERROR] %s\n", trial_counter, e$message))
                return(list(Score = NA, Pred = 0, failed = TRUE))
            }, warning = function(w) {
                # Suppress warnings but continue
                suppressWarnings(FUN(...))
            })
            
            # Validate result structure
            if (is.null(result) || !is.list(result)) {
                cat(sprintf("     [TRIAL %d] Invalid result structure\n", trial_counter))
                failed_trials <<- failed_trials + 1
                consecutive_failures <<- consecutive_failures + 1
                return(safe_bo_return(0.001, add_jitter = TRUE))
            }
            
            # Check for Score validity
            score <- result$Score
            if (is.null(score) || !is.numeric(score) || length(score) != 1 || !is.finite(score)) {
                cat(sprintf("     [TRIAL %d] Non-finite or invalid score: %s\n", 
                            trial_counter, as.character(score)))
                failed_trials <<- failed_trials + 1
                consecutive_failures <<- consecutive_failures + 1
                return(safe_bo_return(0.001, add_jitter = TRUE))
            }
            
            # Check for very small scores (likely failed trials)
            if (score <= 0.001) {
                cat(sprintf("     [TRIAL %d] Very low score (%.6f), treating as marginal success\n", 
                            trial_counter, score))
            }
            
            # Success - reset consecutive failures
            successful_trials <<- successful_trials + 1
            consecutive_failures <<- 0
            score_history <<- c(score_history, score)
            
            # Track best result
            if (score > best_score) {
                best_score <<- score
                best_params <<- list(...)
            }
            
            # Store in history
            all_params[[successful_trials]] <<- list(...)
            all_scores <<- c(all_scores, score)
            
            # CRITICAL: Ensure minimum variance in scores to prevent GP fitting issues
            # This is the main cause of "non-finite value supplied by optim"
            if (length(score_history) >= 3) {
                score_var <- var(score_history)
                if (!is.na(score_var) && score_var < 1e-10) {
                    # Add small jitter to prevent singular covariance matrix
                    jitter_amount <- runif(1, 1e-6, 1e-4)
                    score <- score + jitter_amount
                    cat(sprintf("     [TRIAL %d] Added jitter (%.2e) for GP stability\n", 
                                trial_counter, jitter_amount))
                }
            }
            
            return(list(Score = score, Pred = 0))
        }
        
        # Run BayesianOptimization with the wrapped function
        BayesianOptimization(
            FUN = wrapped_FUN,
            bounds = bounds,
            init_points = init_points,
            n_iter = n_iter,
            acq = acq,
            kappa = kappa,
            eps = eps,
            verbose = verbose
        )
        
    }, error = function(e) {
        error_msg <- e$message
        
        # Check for specific GP-related errors
        if (grepl("non-finite value supplied by optim", error_msg, ignore.case = TRUE) ||
            grepl("singular", error_msg, ignore.case = TRUE) ||
            grepl("Cholesky", error_msg, ignore.case = TRUE) ||
            grepl("covariance", error_msg, ignore.case = TRUE)) {
            
            cat(sprintf("     [BO PHASE 1] GP fitting error detected: %s\n", error_msg))
            cat("     [BO PHASE 1] Switching to Phase 2 (manual trial execution)...\n")
            
        } else {
            cat(sprintf("     [BO PHASE 1 ERROR] %s\n", error_msg))
        }
        
        return(NULL)
    })
    
    # Check if Phase 1 succeeded
    if (!is.null(bo_result) && !is.null(bo_result$Best_Value) && 
        is.finite(bo_result$Best_Value)) {
        
        n_completed <- if (!is.null(bo_result$History)) nrow(bo_result$History) else 0
        
        cat(sprintf("   [BO] Phase 1 completed successfully: %d trials, best=%.4f\n", 
                    n_completed, bo_result$Best_Value))
        
        # Add tracking info
        bo_result$actual_successes <- successful_trials
        bo_result$total_failures <- failed_trials
        bo_result$target_trials <- total_trials_needed
        bo_result$phase_completed <- 1
        
        return(bo_result)
    }
    
    # === PHASE 2: Manual trial execution to reach target count ===
    cat("\n   [BO] Phase 2: Running manual trials with pre-generated parameters...\n")
    cat(sprintf("   [BO] Current status: %d successful, %d failed\n", 
                successful_trials, failed_trials))
    
    # Continue from where we left off
    param_index <- successful_trials + failed_trials + 1
    
    while (successful_trials < total_trials_needed && 
           failed_trials < max_total_failures &&
           consecutive_failures < max_consecutive_failures) {
        
        # Get pre-generated parameters (maintains reproducibility with same seed)
        if (param_index <= length(pregenerated_params)) {
            current_params <- pregenerated_params[[param_index]]
        } else {
            # Fallback: generate new params deterministically
            set.seed(seed + param_index)
            current_params <- lapply(names(bounds), function(param_name) {
                b <- bounds[[param_name]]
                if (is.integer(b) || (is.numeric(b) && all(b == floor(b)))) {
                    sample(seq(as.integer(b[1]), as.integer(b[2])), 1)
                } else {
                    runif(1, b[1], b[2])
                }
            })
            names(current_params) <- names(bounds)
        }
        param_index <- param_index + 1
        
        # Display progress
        if ((successful_trials + failed_trials) %% 10 == 0) {
            cat(sprintf("     [PHASE 2] Progress: %d/%d successful, %d failed, param_index=%d\n", 
                        successful_trials, total_trials_needed, failed_trials, param_index - 1))
        }
        
        # Evaluate with error handling
        result <- tryCatch({
            do.call(FUN, current_params)
        }, error = function(e) {
            cat(sprintf("     [PHASE 2 TRIAL ERROR] %s\n", e$message))
            list(Score = NA, failed = TRUE)
        }, warning = function(w) {
            suppressWarnings(do.call(FUN, current_params))
        })
        
        # Validate result
        score <- NA
        if (is.list(result) && !is.null(result$Score)) {
            score <- result$Score
        }
        
        # Check for valid score
        if (is.na(score) || !is.finite(score) || score <= 0) {
            failed_trials <- failed_trials + 1
            consecutive_failures <- consecutive_failures + 1
            
            if (consecutive_failures >= 5) {
                cat(sprintf("     [WARNING] %d consecutive failures. Continuing...\n", 
                            consecutive_failures))
            }
            next
        }
        
        # Success
        successful_trials <- successful_trials + 1
        consecutive_failures <- 0
        all_params[[successful_trials]] <- current_params
        all_scores <- c(all_scores, score)
        
        if (score > best_score) {
            best_score <- score
            best_params <- current_params
            cat(sprintf("     [PHASE 2] New best score: %.4f (trial %d)\n", 
                        score, successful_trials))
        }
        
        # Progress update every 5 successful trials
        if (successful_trials %% 5 == 0) {
            cat(sprintf("     [PHASE 2] Milestone: %d/%d successful (best=%.4f)\n", 
                        successful_trials, total_trials_needed, best_score))
        }
    }
    
    # === PHASE 3: Handle complete failure ===
    if (successful_trials == 0 || is.null(best_params)) {
        cat("\n   [BO] Phase 3: All standard approaches failed. Running emergency random search...\n")
        
        emergency_result <- run_random_search_fallback(
            FUN = FUN, 
            bounds = bounds, 
            n_evals = min(20, total_trials_needed),
            seed = seed
        )
        
        if (!is.null(emergency_result) && !is.null(emergency_result$Best_Value) && 
            is.finite(emergency_result$Best_Value)) {
            emergency_result$phase_completed <- 3
            emergency_result$is_emergency_fallback <- TRUE
            return(emergency_result)
        }
        
        # Complete failure - return minimal valid result
        cat("   [BO] CRITICAL: Complete optimization failure. Returning default parameters.\n")
        
        default_params <- lapply(names(bounds), function(p) {
            b <- bounds[[p]]
            mean(b)  # Use midpoint of bounds
        })
        names(default_params) <- names(bounds)
        
        return(list(
            Best_Par = unlist(default_params),
            Best_Value = 0.001,
            History = data.frame(),
            actual_successes = 0,
            total_failures = failed_trials,
            target_trials = total_trials_needed,
            phase_completed = 3,
            is_complete_failure = TRUE
        ))
    }
    
    # === PHASE 4: Compile final results ===
    cat(sprintf("\n   [BO] Final results: %d/%d successful trials, %d total failures\n", 
                successful_trials, total_trials_needed, failed_trials))
    
    # Build history dataframe
    history_df <- tryCatch({
        if (length(all_params) > 0 && length(all_scores) > 0) {
            history_list <- lapply(seq_along(all_params), function(i) {
                row <- as.data.frame(all_params[[i]], stringsAsFactors = FALSE)
                row$Value <- all_scores[i]
                row
            })
            bind_rows(history_list)
        } else {
            data.frame()
        }
    }, error = function(e) {
        cat(sprintf("     [WARNING] Failed to build history: %s\n", e$message))
        data.frame()
    })
    
    # Create result structure compatible with BayesianOptimization output
    result <- list(
        Best_Par = unlist(best_params),
        Best_Value = best_score,
        History = history_df,
        Pred = NULL,
        actual_successes = successful_trials,
        total_failures = failed_trials,
        target_trials = total_trials_needed,
        phase_completed = 2,
        replacement_used = (failed_trials > 0),
        seed_used = seed
    )
    
    # Summary message
    if (successful_trials >= total_trials_needed) {
        cat(sprintf("   ✓ Target trial count reached: %d/%d (best=%.4f)\n", 
                    successful_trials, total_trials_needed, best_score))
    } else if (successful_trials >= total_trials_needed * 0.5) {
        cat(sprintf("   ⚠ Partial completion: %d/%d trials (%.1f%%), best=%.4f\n", 
                    successful_trials, total_trials_needed, 
                    100 * successful_trials / total_trials_needed, best_score))
    } else {
        cat(sprintf("   ⚠ Low completion rate: %d/%d trials (%.1f%%), best=%.4f\n", 
                    successful_trials, total_trials_needed, 
                    100 * successful_trials / total_trials_needed, best_score))
    }
    
    return(result)
}

#' Random Search Fallback with Seed Preservation
#' @description Emergency fallback when Bayesian optimization completely fails
#' @param FUN Objective function
#' @param bounds Parameter bounds
#' @param n_evals Number of evaluations
#' @param seed Random seed for reproducibility
#' @return Best result found
run_random_search_fallback <- function(FUN, bounds, n_evals, seed = NULL) {
    cat(sprintf("   [RANDOM SEARCH] Running %d evaluations as fallback...\n", n_evals))
    
    # Set seed for reproducibility
    if (!is.null(seed)) {
        set.seed(seed)
    }
    
    best_score <- -Inf
    best_params <- NULL
    history_list <- list()
    successful <- 0
    failed <- 0
    
    # Pre-generate all parameters for reproducibility
    all_params <- lapply(1:n_evals, function(i) {
        params <- lapply(names(bounds), function(param_name) {
            b <- bounds[[param_name]]
            if (is.integer(b) || (is.numeric(b) && all(b == floor(b)))) {
                sample(seq(as.integer(b[1]), as.integer(b[2])), 1)
            } else {
                runif(1, b[1], b[2])
            }
        })
        names(params) <- names(bounds)
        params
    })
    
    for (i in 1:n_evals) {
        params <- all_params[[i]]
        
        # Evaluate with full error protection
        result <- tryCatch({
            do.call(FUN, params)
        }, error = function(e) {
            list(Score = NA)
        }, warning = function(w) {
            suppressWarnings(do.call(FUN, params))
        })
        
        # Extract and validate score
        score <- NA
        if (is.list(result) && !is.null(result$Score)) {
            score <- result$Score
        }
        
        if (is.na(score) || !is.finite(score) || score <= 0) {
            failed <- failed + 1
            next
        }
        
        successful <- successful + 1
        
        # Store in history
        history_row <- as.data.frame(params, stringsAsFactors = FALSE)
        history_row$Value <- score
        history_list[[length(history_list) + 1]] <- history_row
        
        # Update best
        if (score > best_score) {
            best_score <- score
            best_params <- params
        }
        
        if (i %% 5 == 0 || i == n_evals) {
            cat(sprintf("     [RANDOM] %d/%d complete, %d successful, best=%.4f\n", 
                        i, n_evals, successful, 
                        if (is.finite(best_score)) best_score else 0))
        }
    }
    
    # Combine history
    history_df <- tryCatch({
        if (length(history_list) > 0) bind_rows(history_list) else data.frame()
    }, error = function(e) {
        data.frame()
    })
    
    cat(sprintf("   [RANDOM SEARCH] Complete: %d/%d successful (%.1f%%)\n", 
                successful, n_evals, 100 * successful / n_evals))
    
    return(list(
        Best_Par = if (!is.null(best_params)) unlist(best_params) else NULL,
        Best_Value = if (is.finite(best_score)) best_score else NA,
        History = history_df,
        is_fallback = TRUE,
        actual_successes = successful,
        total_failures = failed
    ))
}

# ==============================================================================
# --- OBJECTIVE FUNCTION ---
# ==============================================================================

#' Objective function for Bayesian Optimization
#' @param n_hvg Number of highly variable genes
#' @param n_pcs Number of principal components
#' @param n_neighbors Number of neighbors for clustering
#' @param resolution Clustering resolution
#' @return List with Score and Pred
objective_function <- function(n_hvg, n_pcs, n_neighbors, resolution) {
    
    # === INPUT VALIDATION WITH STRICT BOUNDS ===
    n_hvg <- tryCatch(as.integer(round(sanitize_metric(n_hvg, 2000))), error = function(e) 2000L)
    n_pcs <- tryCatch(as.integer(round(sanitize_metric(n_pcs, 30))), error = function(e) 30L)
    n_neighbors <- tryCatch(as.integer(round(sanitize_metric(n_neighbors, 20))), error = function(e) 20L)
    resolution <- tryCatch(sanitize_metric(resolution, 0.8), error = function(e) 0.8)
    
    # Strict bounds enforcement
    n_hvg <- max(100L, min(n_hvg, 25000L))
    n_pcs <- max(5L, min(n_pcs, 150L))
    n_neighbors <- max(5L, min(n_neighbors, 100L))
    resolution <- max(0.01, min(resolution, 5.0))
    
    # Verify all parameters are finite
    if (!all(is.finite(c(n_hvg, n_pcs, n_neighbors, resolution)))) {
        cat("     [WARNING] Non-finite parameters detected after bounds check.\n")
        return(safe_bo_return(0.001))
    }
    
    start_time <- Sys.time()
    
    # === CACHE CHECK ===
    cache_key <- sprintf("hvg%d_pcs%d_nei%d_res%.6f", n_hvg, n_pcs, n_neighbors, resolution)
    
    cache_env <- GLOBAL_ENV$OPTIMIZATION_CACHE
    if (!is.null(cache_env) && is.environment(cache_env) && exists(cache_key, envir = cache_env)) {
        cached_result <- tryCatch({
            get(cache_key, envir = cache_env)
        }, error = function(e) NULL)
        
        if (!is.null(cached_result) && is.list(cached_result)) {
            cached_score <- cached_result$Score
            
            if (!is.null(cached_score) && is.numeric(cached_score) && 
                length(cached_score) == 1 && is.finite(cached_score) && 
                cached_score > 0 && cached_score < 1e6) {
                
                cat(sprintf("     [CACHE HIT] Key: %s, Score: %.6f\n", cache_key, cached_score))
                
                if (!is.null(cached_result$Metrics)) {
                    GLOBAL_ENV$TRIAL_METADATA[[length(GLOBAL_ENV$TRIAL_METADATA) + 1]] <- cached_result$Metrics
                }
                
                return(safe_bo_return(cached_score, add_jitter = TRUE))
            }
        }
    }
    
    # === MAIN PROCESSING ===
    result <- tryCatch({
        
        # Get base Seurat object
        seurat_proc <- GLOBAL_ENV$seurat_base
        if (is.null(seurat_proc)) {
            stop("seurat_base not found in global environment")
        }
        
        # Verify we have enough cells
        if (ncol(seurat_proc) < 10) {
            cat("     [WARNING] Too few cells for analysis.\n")
            return(safe_bo_return(0.001))
        }
        
        # === CHECK FOR BATCHED DATA (NEW) ===
        is_batched <- GLOBAL_ENV$IS_BATCHED_DATA %||% FALSE
        batch_col <- GLOBAL_ENV$BATCH_COLUMN %||% NULL
        
        # --- HVG Selection ---
        hvg_result <- tryCatch({
            args <- GLOBAL_ENV$ARGS
            is_two_step_hvg <- !is.null(args$hvg_min_mean) && 
                !is.null(args$hvg_max_mean) && 
                !is.null(args$hvg_min_disp)
            
            if (is_two_step_hvg) {
                seurat_proc <- FindVariableFeatures(seurat_proc, method = "vst", 
                                                    nfeatures = nrow(seurat_proc), verbose = FALSE)
                hvg_info <- HVFInfo(seurat_proc, method = "vst", assay = "RNA")
                hvg_info_filtered <- subset(hvg_info, 
                                            mean > args$hvg_min_mean & 
                                                mean < args$hvg_max_mean & 
                                                variance.standardized > args$hvg_min_disp)
                
                if (nrow(hvg_info_filtered) < 50) {
                    seurat_proc <- FindVariableFeatures(seurat_proc, method = "vst", 
                                                        nfeatures = n_hvg, verbose = FALSE)
                } else {
                    hvg_info_sorted <- hvg_info_filtered[order(-hvg_info_filtered$variance.standardized), ]
                    n_hvg_safe <- min(n_hvg, nrow(hvg_info_sorted))
                    top_genes <- rownames(hvg_info_sorted)[1:n_hvg_safe]
                    VariableFeatures(seurat_proc) <- top_genes
                }
            } else {
                seurat_proc <- FindVariableFeatures(seurat_proc, method = "vst", 
                                                    nfeatures = n_hvg, verbose = FALSE)
            }
            
            list(success = TRUE, seurat = seurat_proc)
        }, error = function(e) {
            cat(sprintf("     [WARNING] HVG selection failed: %s\n", e$message))
            list(success = FALSE, seurat = seurat_proc)
        })
        
        if (!hvg_result$success || length(VariableFeatures(hvg_result$seurat)) < 50) {
            return(safe_bo_return(0.001))
        }
        seurat_proc <- hvg_result$seurat
        
        # --- Scale & PCA ---
        pca_result <- tryCatch({
            all_genes_sorted <- sort(rownames(seurat_proc))
            seurat_proc <- ScaleData(seurat_proc, features = all_genes_sorted, verbose = FALSE)
            
            n_pcs_to_compute <- min(N_PCS_FOR_PCA, ncol(seurat_proc) - 1, nrow(seurat_proc) - 1)
            n_pcs_use <- min(n_pcs, n_pcs_to_compute)
            
            if (n_pcs_use < 2) {
                return(list(success = FALSE, seurat = seurat_proc, n_pcs = 2))
            }
            
            seurat_proc <- RunPCA(seurat_proc, npcs = n_pcs_to_compute, 
                                  features = VariableFeatures(seurat_proc), 
                                  verbose = FALSE, 
                                  seed.use = GLOBAL_ENV$RANDOM_SEED)
            
            list(success = TRUE, seurat = seurat_proc, n_pcs = n_pcs_use)
        }, error = function(e) {
            cat(sprintf("     [WARNING] PCA failed: %s\n", e$message))
            list(success = FALSE, seurat = seurat_proc, n_pcs = 2)
        })
        
        if (!pca_result$success) {
            return(safe_bo_return(0.001))
        }
        seurat_proc <- pca_result$seurat
        n_pcs_use <- pca_result$n_pcs
        
        # === HARMONY INTEGRATION FOR BATCHED DATA (NEW) ===
        reduction_to_use <- "pca"  # Default for single-sample
        
        if (is_batched && !is.null(batch_col)) {
            harmony_result <- tryCatch({
                seurat_proc <- run_harmony_integration(
                    seurat_proc, 
                    batch_col = batch_col,
                    dims_use = 1:n_pcs_use,
                    verbose = FALSE
                )
                list(success = TRUE, seurat = seurat_proc)
            }, error = function(e) {
                cat(sprintf("     [WARNING] Harmony integration failed: %s\n", e$message))
                cat("     [WARNING] Falling back to PCA-based clustering.\n")
                list(success = FALSE, seurat = seurat_proc)
            })
            
            if (harmony_result$success) {
                seurat_proc <- harmony_result$seurat
                reduction_to_use <- "harmony"
            }
        }
        
        # --- Clustering (USE APPROPRIATE REDUCTION) ---
        cluster_result <- tryCatch({
            k_param_safe <- min(n_neighbors, ncol(seurat_proc) - 1)
            if (k_param_safe < 2) k_param_safe <- 2
            
            # Use harmony if available, otherwise pca
            seurat_proc <- FindNeighbors(seurat_proc, dims = 1:n_pcs_use, 
                                         k.param = k_param_safe, verbose = FALSE,
                                         reduction = reduction_to_use)  # <-- MODIFIED
            seurat_proc <- FindClusters(seurat_proc, resolution = resolution, 
                                        algorithm = 4, 
                                        random.seed = GLOBAL_ENV$RANDOM_SEED, 
                                        verbose = FALSE)
            seurat_proc$leiden <- seurat_proc$seurat_clusters
            
            list(success = TRUE, seurat = seurat_proc, k = k_param_safe)
        }, error = function(e) {
            cat(sprintf("     [WARNING] Clustering failed: %s\n", e$message))
            list(success = FALSE, seurat = seurat_proc, k = 2)
        })
        
        if (!cluster_result$success) {
            return(safe_bo_return(0.001))
        }
        seurat_proc <- cluster_result$seurat
        k_param_safe <- cluster_result$k
        
        # --- Silhouette Score (USE APPROPRIATE REDUCTION) ---
        rescaled_silhouette <- tryCatch({
            n_clusters <- nlevels(seurat_proc$leiden)
            n_cells <- ncol(seurat_proc)
            
            if (n_clusters > 1 && n_clusters < n_cells && n_cells > 10) {
                # Use harmony embeddings if available
                embedding_coords <- Embeddings(seurat_proc, reduction = reduction_to_use)[, 1:n_pcs_use, drop = FALSE]
                dist_matrix <- dist(embedding_coords)
                sil_scores <- silhouette(as.integer(seurat_proc$leiden), dist_matrix)
                silhouette_avg <- mean(sil_scores[, "sil_width"], na.rm = TRUE)
                
                if (is.finite(silhouette_avg)) {
                    (silhouette_avg + 1) / 2
                } else {
                    0.5
                }
            } else {
                0.5
            }
        }, error = function(e) {
            cat(sprintf("     [WARNING] Silhouette failed: %s\n", e$message))
            0.5
        })
        rescaled_silhouette <- sanitize_metric(rescaled_silhouette, 0.5, min_val = 0, max_val = 1)
        
        # --- CAS Calculation ---
        cas_result <- tryCatch({
            seurat_ref <- GLOBAL_ENV$seurat_ref
            ref_labels_col <- GLOBAL_ENV$REF_LABELS_COL
            args <- GLOBAL_ENV$ARGS
            
            k_weight_safe <- min(k_param_safe, ncol(seurat_proc) - 1, 50)
            if (k_weight_safe < 1) k_weight_safe <- 1
            
            transfer_anchors <- FindTransferAnchors(
                reference = seurat_ref, 
                query = seurat_proc, 
                dims = 1:n_pcs_use, 
                reduction = "pcaproject", 
                reference.assay = args$reference_assay, 
                query.assay = "RNA", 
                verbose = FALSE
            )
            
            predictions <- TransferData(
                anchorset = transfer_anchors, 
                refdata = seurat_ref[[ref_labels_col, drop = TRUE]], 
                dims = 1:n_pcs_use, 
                k.weight = k_weight_safe, 
                weight.reduction = "pcaproject", 
                verbose = FALSE
            )
            
            seurat_proc$ctpt_individual_prediction <- predictions$predicted.id
            
            # Consensus prediction
            metadata_df <- seurat_proc@meta.data %>% 
                group_by(leiden) %>% 
                mutate(ctpt_consensus_prediction = names(which.max(table(ctpt_individual_prediction)))) %>% 
                ungroup()
            
            seurat_proc$ctpt_consensus_prediction <- metadata_df$ctpt_consensus_prediction
            
            # Weighted CAS
            total_cells <- nrow(metadata_df)
            total_matching <- sum(metadata_df$ctpt_individual_prediction == 
                                      metadata_df$ctpt_consensus_prediction, na.rm = TRUE)
            w_cas <- if (total_cells > 0) (total_matching / total_cells) * 100 else 0
            
            # Simple CAS
            if (args$cas_aggregation_method == 'leiden') {
                cas_groups <- metadata_df %>% 
                    group_by(leiden) %>% 
                    summarise(cas = mean(ctpt_individual_prediction == 
                                             dplyr::first(ctpt_consensus_prediction), na.rm = TRUE) * 100,
                              .groups = "drop")
                s_cas <- mean(cas_groups$cas, na.rm = TRUE)
            } else {
                cas_groups <- metadata_df %>% 
                    group_by(ctpt_consensus_prediction) %>% 
                    summarise(cas = mean(ctpt_individual_prediction == 
                                             dplyr::first(ctpt_consensus_prediction), na.rm = TRUE) * 100,
                              .groups = "drop")
                s_cas <- mean(cas_groups$cas, na.rm = TRUE)
            }
            
            list(weighted = w_cas, simple = s_cas, seurat = seurat_proc, success = TRUE)
            
        }, error = function(e) {
            cat(sprintf("     [WARNING] CAS calculation failed: %s\n", e$message))
            list(weighted = 0, simple = 0, seurat = seurat_proc, success = FALSE)
        })
        
        weighted_mean_cas <- sanitize_metric(cas_result$weighted, 0, min_val = 0, max_val = 100)
        simple_mean_cas <- sanitize_metric(cas_result$simple, 0, min_val = 0, max_val = 100)
        seurat_proc <- cas_result$seurat
        
        # --- MCS Calculation (unchanged) ---
        mean_mcs <- tryCatch({
            if (!"ctpt_consensus_prediction" %in% colnames(seurat_proc@meta.data)) {
                return(0)
            }
            
            args <- GLOBAL_ENV$ARGS
            
            Idents(seurat_proc) <- "ctpt_consensus_prediction"
            label_counts <- table(seurat_proc$ctpt_consensus_prediction)
            valid_labels <- names(label_counts[label_counts > 1])
            
            if (length(valid_labels) < 2) {
                return(0)
            }
            
            markers <- FindAllMarkers(seurat_proc, assay = "RNA", only.pos = TRUE, 
                                      min.pct = 0.25, logfc.threshold = 0, 
                                      verbose = FALSE, layer = "data")
            
            if (nrow(markers) == 0) {
                return(0)
            }
            
            if (args$marker_gene_model == 'non-mitochondrial') {
                markers <- markers %>% 
                    filter(!grepl(MITO_REGEX_PATTERN, gene, ignore.case = TRUE))
            }
            
            if (nrow(markers) == 0) {
                return(0)
            }
            
            top_genes <- markers %>% 
                group_by(cluster) %>% 
                top_n(n = N_MCS_TOP_GENES, wt = avg_log2FC)
            
            unique_genes <- unique(top_genes$gene)
            unique_genes <- intersect(unique_genes, rownames(seurat_proc))
            
            if (length(unique_genes) == 0) {
                return(0)
            }
            
            expr_data <- FetchData(seurat_proc, 
                                   vars = c("ctpt_consensus_prediction", unique_genes))
            frac_df <- expr_data %>% 
                group_by(ctpt_consensus_prediction) %>% 
                summarise(across(all_of(unique_genes), ~ mean(.x > 0, na.rm = TRUE)),
                          .groups = "drop")
            
            mcs_scores <- sapply(unique(top_genes$cluster), function(ct) {
                m <- top_genes %>% filter(cluster == ct) %>% pull(gene)
                m <- intersect(m, colnames(frac_df))
                if (length(m) > 0 && ct %in% frac_df$ctpt_consensus_prediction) {
                    vals <- as.numeric(frac_df[frac_df$ctpt_consensus_prediction == ct, m])
                    mean(vals, na.rm = TRUE)
                } else {
                    NA_real_
                }
            })
            
            result <- mean(mcs_scores, na.rm = TRUE) * 100
            if (is.na(result) || !is.finite(result)) 0 else result
            
        }, error = function(e) {
            cat(sprintf("     [WARNING] MCS calculation failed: %s\n", e$message))
            0
        })
        
        mean_mcs <- sanitize_metric(mean_mcs, 0, min_val = 0, max_val = 100)
        
        # === MPS CALCULATION (NEW) ===
        mps_result <- tryCatch({
            marker_db <- GLOBAL_ENV$MARKER_DB
            args <- GLOBAL_ENV$ARGS  # Get args for DEG ranking parameters
            
            if (!is.null(marker_db) && length(marker_db) > 0) {
                
                # Build DEG weights from arguments (with defaults)
                deg_weights <- c(
                    fc   = args$deg_weight_fc   %||% 0.4,
                    expr = args$deg_weight_expr %||% 0.3,
                    pct  = args$deg_weight_pct  %||% 0.3
                )
                
                # Get DEG ranking method from arguments (with default)
                deg_method <- args$deg_ranking_method %||% "weighted_composite"
                
                # Call calculate_mps with ALL parameters
                mps_calc <- calculate_mps(
                    seurat_obj = seurat_proc,
                    marker_db = marker_db,
                    group_by = "ctpt_consensus_prediction",
                    n_top_genes = MPS_CONFIG$n_top_degs,
                    expand_abbreviations = TRUE,
                    deg_ranking_method = deg_method,
                    deg_weights = deg_weights,
                    verbose = FALSE
                )
                mps_calc
            } else {
                list(mean_mps = NA, scores = NULL, note = "No marker database")
            }
        }, error = function(e) {
            cat(sprintf("     [WARNING] MPS calculation failed: %s\n", e$message))
            list(mean_mps = NA, scores = NULL, note = e$message)
        })
        
        mean_mps_score <- sanitize_metric(mps_result$mean_mps, NA)
        
        args <- GLOBAL_ENV$ARGS
        target <- GLOBAL_ENV$CURRENT_OPTIMIZATION_TARGET
        model_type <- args$model_type %||% "annotation"
        
        epsilon <- 1e-4
        
        score <- tryCatch({
            if (target == 'weighted_cas') {
                sanitize_metric(weighted_mean_cas, 0.001, min_val = 0.001)
                
            } else if (target == 'simple_cas') {
                sanitize_metric(simple_mean_cas, 0.001, min_val = 0.001)
                
            } else if (target == 'mcs') {
                sanitize_metric(mean_mcs, 0.001, min_val = 0.001)
                
            } else if (target == 'mps') {
                # MPS-only target
                if (is.na(mean_mps_score)) {
                    cat("     [WARNING] MPS not available. Falling back to CAS.\n")
                    sanitize_metric(weighted_mean_cas, 0.001, min_val = 0.001)
                } else {
                    sanitize_metric(mean_mps_score, 0.001, min_val = 0.001)
                }
                
            } else if (target == 'balanced') {
                # ============================================================
                # BALANCED SCORING - Model type determines components
                # ============================================================
                
                if (model_type == 'silhouette') {
                    # ----------------------------------------------------------
                    # SILHOUETTE: Silhouette only (Pure clustering quality)
                    # Formula: Silhouette (already normalized to 0-1)
                    # ----------------------------------------------------------
                    final_val <- sanitize_metric(rescaled_silhouette * 100, 0.001, min_val = 0.001)
                    
                    cat(sprintf("     [SILHOUETTE MODEL] Sil=%.4f -> Score=%.4f\n",
                                rescaled_silhouette, final_val))
                    final_val
                    
                } else if (model_type == 'structural') {
                    # ----------------------------------------------------------
                    # STRUCTURAL: CAS + MCS + Silhouette (Cluster structure)
                    # Formula: (wCAS × sCAS × MCS × Silhouette)^(1/4)
                    # ----------------------------------------------------------
                    vals <- c(
                        weighted_mean_cas / 100 + epsilon,
                        simple_mean_cas / 100 + epsilon,
                        mean_mcs / 100 + epsilon,
                        rescaled_silhouette + epsilon
                    )
                    final_val <- safe_geometric_mean(vals) * 100
                    
                    cat(sprintf("     [STRUCTURAL MODEL] wCAS=%.2f%% sCAS=%.2f%% MCS=%.2f%% Sil=%.4f -> Score=%.4f\n",
                                weighted_mean_cas, simple_mean_cas, mean_mcs, rescaled_silhouette, final_val))
                    final_val
                    
                } else if (model_type == 'mps_integrated') {
                    # ----------------------------------------------------------
                    # MPS-INTEGRATED: CAS + MCS + MPS (multiplicative)
                    # WARNING: MPS=0 will zero out the entire score
                    # Formula: (wCAS × sCAS × MCS × MPS)^(1/4) or (wCAS × sCAS × MCS)^(1/3)
                    # ----------------------------------------------------------
                    cas_val <- (weighted_mean_cas / 100 + epsilon)
                    mcs_val <- (mean_mcs / 100 + epsilon)
                    
                    if (!is.na(mean_mps_score) && is.finite(mean_mps_score) && mean_mps_score > 0) {
                        mps_val <- (mean_mps_score / 100 + epsilon)
                        vals <- c(cas_val, mcs_val, mps_val)
                        final_val <- safe_geometric_mean(vals) * 100
                        
                        cat(sprintf("     [MPS_INTEGRATED] wCAS=%.2f%% MCS=%.2f%% MPS=%.2f%% -> Score=%.4f\n",
                                    weighted_mean_cas, mean_mcs, mean_mps_score, final_val))
                    } else {
                        # Fallback when MPS unavailable or zero
                        vals <- c(cas_val, mcs_val)
                        final_val <- safe_geometric_mean(vals) * 100
                        
                        cat(sprintf("     [MPS_INTEGRATED] wCAS=%.2f%% MCS=%.2f%% MPS=N/A -> Score=%.4f (fallback)\n",
                                    weighted_mean_cas, mean_mcs, final_val))
                    }
                    final_val
                    
                } else if (model_type == 'mps_bonus') {
                    # ----------------------------------------------------------
                    # MPS-BONUS: CAS + MCS base with additive MPS bonus
                    # Formula: base_score + (mps_bonus_weight × MPS)
                    # Advantage: MPS=0 or NA does NOT penalize the score
                    # ----------------------------------------------------------
                    cas_w_val <- (weighted_mean_cas / 100 + epsilon)
                    cas_s_val <- (simple_mean_cas / 100 + epsilon)
                    mcs_val <- (mean_mcs / 100 + epsilon)
                    base_score <- safe_geometric_mean(c(cas_w_val, cas_s_val, mcs_val))
                    
                    # MPS bonus: scales from 0 to mps_bonus_weight
                    mps_bonus_weight <- args$mps_bonus_weight %||% 0.2
                    
                    if (!is.na(mean_mps_score) && is.finite(mean_mps_score) && mean_mps_score > 0) {
                        # Normalize MPS to [0, 1] range (MPS is in percentage 0-100)
                        mps_normalized <- max(0, min(1, mean_mps_score / 100))
                        mps_bonus <- mps_bonus_weight * mps_normalized
                    } else {
                        # No penalty when MPS unavailable
                        mps_bonus <- 0
                    }
                    
                    final_val <- (base_score + mps_bonus) * 100
                    
                    # === MODIFIED: Updated log message ===
                    cat(sprintf("     [MPS_BONUS] wCAS=%.2f%% sCAS=%.2f%% MCS=%.2f%% -> base=%.4f\n",
                                weighted_mean_cas, simple_mean_cas, mean_mcs, base_score * 100))
                    cat(sprintf("     [MPS_BONUS] MPS=%.2f%%, bonus=%.4f (weight=%.2f) -> Score=%.4f\n",
                                mean_mps_score %||% 0, mps_bonus * 100, mps_bonus_weight, final_val))
                    final_val
                    
                } else {
                    # ----------------------------------------------------------
                    # ANNOTATION (default): CAS + MCS only
                    # Formula: (wCAS × sCAS × MCS)^(1/3)
                    # NOTE: Silhouette is NOT included
                    # ----------------------------------------------------------
                    vals <- c(
                        weighted_mean_cas / 100 + epsilon,
                        simple_mean_cas / 100 + epsilon,
                        mean_mcs / 100 + epsilon
                    )
                    final_val <- safe_geometric_mean(vals) * 100
                    
                    cat(sprintf("     [ANNOTATION MODEL] wCAS=%.2f%% sCAS=%.2f%% MCS=%.2f%% -> Score=%.4f\n",
                                weighted_mean_cas, simple_mean_cas, mean_mcs, final_val))
                    final_val
                }
                
            } else {
                # Default fallback
                sanitize_metric(weighted_mean_cas, 0.001, min_val = 0.001)
            }
        }, error = function(e) {
            cat(sprintf("     [WARNING] Score calculation error: %s\n", e$message))
            0.001
        })
        
        # Final score sanitization
        score <- sanitize_metric(score, 0.001, min_val = 0.001, max_val = 1e5)
        
        # === STORE TRIAL METADATA (MODIFIED to include MPS) ===
        trial_data <- list(
            n_individual_labels = tryCatch(
                n_distinct(seurat_proc$ctpt_individual_prediction), 
                error = function(e) 0
            ),
            n_consensus_labels = tryCatch(
                n_distinct(seurat_proc$ctpt_consensus_prediction), 
                error = function(e) 0
            ),
            weighted_mean_cas = weighted_mean_cas, 
            simple_mean_cas = simple_mean_cas, 
            mean_mcs = mean_mcs,
            mean_mps = mean_mps_score,  # NEW
            silhouette_score = rescaled_silhouette,
            is_batched = is_batched,
            reduction_used = reduction_to_use
        )
        
        GLOBAL_ENV$TRIAL_METADATA[[length(GLOBAL_ENV$TRIAL_METADATA) + 1]] <- trial_data
        
        # === LOG RESULTS (MODIFIED to include MPS) ===
        end_time <- Sys.time()
        time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))
        strategy_name <- GLOBAL_ENV$CURRENT_STRATEGY_NAME %||% "Unknown"
        
        cat("\n")
        cat("     +--------------------- TRIAL SUMMARY ---------------------+\n")
        cat(sprintf("     | Strategy: %-12s | Time: %.1f sec\n", strategy_name, time_taken))
        if (is_batched) {
            cat(sprintf("     | Mode: BATCHED (Harmony) | Reduction: %s\n", reduction_to_use))
        } else {
            cat(sprintf("     | Mode: SINGLE-SAMPLE     | Reduction: %s\n", reduction_to_use))
        }
        cat("     +---------------------------------------------------------+\n")
        cat(sprintf("     | Params: HVG=%d, PCs=%d, k=%d, res=%.3f\n", 
                    n_hvg, n_pcs_use, k_param_safe, resolution))
        cat("     |---------------------------------------------------------+\n")
        cat(sprintf("     | Weighted CAS: %6.2f%% | Simple CAS: %6.2f%%\n", 
                    weighted_mean_cas, simple_mean_cas))
        cat(sprintf("     | Mean MCS:     %6.2f%% | Silhouette: %.4f\n", 
                    mean_mcs, rescaled_silhouette))
        # NEW: MPS line
        if (!is.na(mean_mps_score)) {
            cat(sprintf("     | Mean MPS:     %6.2f%% | (F1 of marker overlap)\n", mean_mps_score))
        } else {
            cat(sprintf("     | Mean MPS:     %6s | (No marker database)\n", "N/A"))
        }
        cat("     +---------------------------------------------------------+\n")
        cat(sprintf("     | ==> SCORE (%s): %.6f\n", target, score))
        cat("     +---------------------------------------------------------+\n\n")
        
        # === CACHE RESULT ===
        if (!is.null(cache_env) && is.environment(cache_env)) {
            assign(cache_key, list(Score = score, Metrics = trial_data), envir = cache_env)
        }
        
        return(safe_bo_return(score, add_jitter = TRUE))
        
    }, error = function(e) {
        cat(sprintf("     [CRITICAL ERROR] %s\n", e$message))
        cat("     Returning safe default score.\n")
        return(safe_bo_return(0.001, add_jitter = TRUE))
    })
    
    # Final safety check on result
    if (is.null(result) || !is.list(result) || is.null(result$Score) || 
        !is.finite(result$Score)) {
        return(safe_bo_return(0.001, add_jitter = TRUE))
    }
    
    return(result)
}

# ==============================================================================
# --- REPORTING FUNCTIONS ---
# ==============================================================================

#' Evaluate final metrics with optimal parameters
evaluate_final_metrics <- function(params, seurat_input = NULL) {
    cat("\n--- Re-running analysis with overall best parameters for final report ---\n")
    
    if (!is.null(seurat_input)) {
        seurat_final <- seurat_input
    } else {
        seurat_final <- GLOBAL_ENV$seurat_full_data
    }
    
    cat(sprintf("[INFO] Using dataset of %d cells for the final run.\n", ncol(seurat_final)))
    
    params$n_hvg <- as.integer(params$n_hvg)
    params$n_pcs <- as.integer(params$n_pcs)
    params$n_neighbors <- as.integer(params$n_neighbors)
    
    args <- GLOBAL_ENV$ARGS
    
    # HVG Selection
    is_two_step_hvg <- !is.null(args$hvg_min_mean) && !is.null(args$hvg_max_mean) && !is.null(args$hvg_min_disp)
    if (is_two_step_hvg) {
        cat("[INFO] Using two-step sequential HVG selection for final object.\n")
        seurat_final <- FindVariableFeatures(seurat_final, method = "vst", nfeatures = nrow(seurat_final), verbose = FALSE)
        hvg_info <- HVFInfo(seurat_final, method = "vst", assay = "RNA")
        hvg_info_filtered <- subset(hvg_info, mean > args$hvg_min_mean & mean < args$hvg_max_mean & variance.standardized > args$hvg_min_disp)
        hvg_info_sorted <- hvg_info_filtered[order(-hvg_info_filtered$variance.standardized), ]
        n_hvg_safe <- min(params$n_hvg, nrow(hvg_info_sorted))
        top_genes <- rownames(hvg_info_sorted)[1:n_hvg_safe]
        VariableFeatures(seurat_final) <- top_genes
    } else {
        cat(sprintf("[INFO] Using standard rank-based HVG selection with nfeatures = %d for final object.\n", params$n_hvg))
        seurat_final <- FindVariableFeatures(seurat_final, method = "vst", nfeatures = params$n_hvg, verbose = FALSE)
    }
    
    seurat_final <- NormalizeData(seurat_final, normalization.method = "LogNormalize",
                                  scale.factor = 10000, verbose = FALSE)
    seurat_final <- ensure_standard_assay(seurat_final)

    all.genes <- rownames(seurat_final)
    seurat_final <- ScaleData(seurat_final, features = all.genes, verbose = FALSE)
    
    n_pcs_to_compute <- min(N_PCS_FOR_PCA, ncol(seurat_final) - 1, nrow(seurat_final) - 1)
    n_pcs <- min(params$n_pcs, n_pcs_to_compute)
    
    seurat_final <- RunPCA(seurat_final, npcs = n_pcs_to_compute, features = VariableFeatures(object = seurat_final), verbose = FALSE, seed.use = GLOBAL_ENV$RANDOM_SEED)
    seurat_final <- FindNeighbors(seurat_final, dims = 1:n_pcs, k.param = params$n_neighbors, verbose = FALSE)
    seurat_final <- FindClusters(seurat_final, resolution = params$resolution, algorithm = 4, random.seed = GLOBAL_ENV$RANDOM_SEED, verbose = FALSE)
    seurat_final$leiden <- seurat_final$seurat_clusters
    seurat_final <- RunUMAP(seurat_final, dims = 1:n_pcs, seed.use = GLOBAL_ENV$RANDOM_SEED, verbose = FALSE)
    
    # Silhouette
    sil_result <- tryCatch({
        n_clusters <- nlevels(seurat_final$leiden)
        if (n_clusters > 1 && n_clusters < ncol(seurat_final)) {
            sil <- silhouette(as.integer(seurat_final$leiden), dist(Embeddings(seurat_final, "pca")[, 1:n_pcs]))
            sil_avg <- mean(sil[, 3], na.rm = TRUE)
            if (!is.finite(sil_avg)) sil_avg <- 0.0
            list(avg = sil_avg, rescaled = (sil_avg + 1) / 2)
        } else {
            list(avg = 0.0, rescaled = 0.0)
        }
    }, error = function(e) {
        cat(sprintf("     [WARNING] Silhouette calculation failed: %s\n", e$message))
        list(avg = 0.0, rescaled = 0.0)
    })
    
    silhouette_avg <- sil_result$avg
    rescaled_silhouette <- sil_result$rescaled
    
    # CAS
    cas_result <- tryCatch({
        seurat_ref <- GLOBAL_ENV$seurat_ref
        ref_labels_col <- GLOBAL_ENV$REF_LABELS_COL
        
        transfer_anchors_final <- FindTransferAnchors(
            reference = seurat_ref, query = seurat_final, 
            dims = 1:n_pcs, reduction = "pcaproject", 
            reference.assay = args$reference_assay, query.assay = "RNA", 
            verbose = FALSE
        )
        
        k_weight_safe <- min(params$n_neighbors, ncol(seurat_final) - 1, 50)
        
        predictions_final <- TransferData(
            anchorset = transfer_anchors_final, 
            refdata = seurat_ref[[ref_labels_col, drop = TRUE]], 
            dims = 1:n_pcs, k.weight = k_weight_safe, 
            weight.reduction = "pcaproject", verbose = FALSE
        )
        seurat_final$ctpt_individual_prediction <- predictions_final$predicted.id
        seurat_final$ctpt_confidence <- predictions_final$prediction.score.max
        
        metadata_df <- seurat_final@meta.data %>% 
            group_by(leiden) %>% 
            mutate(ctpt_consensus_prediction = names(which.max(table(ctpt_individual_prediction)))) %>% 
            ungroup()
        seurat_final$ctpt_consensus_prediction <- metadata_df$ctpt_consensus_prediction
        
        w_cas <- mean(metadata_df$ctpt_individual_prediction == metadata_df$ctpt_consensus_prediction, na.rm=TRUE) * 100
        
        if (args$cas_aggregation_method == 'leiden') {
            cas_groups <- metadata_df %>% 
                group_by(leiden) %>% 
                summarise(cas = mean(ctpt_individual_prediction == dplyr::first(ctpt_consensus_prediction), na.rm=TRUE), .groups="drop") %>% 
                pull(cas)
            s_cas <- mean(cas_groups, na.rm=TRUE) * 100
        } else {
            cas_groups <- metadata_df %>% 
                group_by(ctpt_consensus_prediction) %>% 
                summarise(cas = mean(ctpt_individual_prediction == dplyr::first(ctpt_consensus_prediction), na.rm=TRUE), .groups="drop") %>% 
                pull(cas)
            s_cas <- mean(cas_groups, na.rm=TRUE) * 100
        }
        
        list(weighted = w_cas, simple = s_cas, seurat = seurat_final)
        
    }, error = function(e) {
        cat(sprintf("     [WARNING] CAS calculation failed: %s\n", e$message))
        list(weighted = 0.0, simple = 0.0, seurat = seurat_final)
    })
    
    weighted_cas <- cas_result$weighted
    simple_cas <- cas_result$simple
    seurat_final <- cas_result$seurat
    
    # MCS
    mean_mcs <- tryCatch({
        Idents(seurat_final) <- "ctpt_consensus_prediction"
        label_counts <- table(seurat_final$ctpt_consensus_prediction)
        valid_labels <- names(label_counts[label_counts > 1])
        
        if (length(valid_labels) >= 2) {
            markers <- FindAllMarkers(seurat_final, assay = "RNA", only.pos = TRUE, 
                                      min.pct = 0.25, logfc.threshold = 0, 
                                      verbose = FALSE, layer = "data")
            if (nrow(markers) > 0) {
                if (args$marker_gene_model == 'non-mitochondrial') {
                    markers <- markers %>% filter(!grepl(MITO_REGEX_PATTERN, gene, ignore.case = TRUE))
                }
                top_genes <- markers %>% group_by(cluster) %>% top_n(n = N_MCS_TOP_GENES, wt = avg_log2FC)
                unique_genes <- unique(top_genes$gene)
                unique_genes <- intersect(unique_genes, rownames(seurat_final))
                
                if (length(unique_genes) > 0) {
                    expr_data <- FetchData(seurat_final, vars = c("ctpt_consensus_prediction", unique_genes))
                    frac_df <- expr_data %>% 
                        group_by(ctpt_consensus_prediction) %>% 
                        summarise(across(all_of(unique_genes), ~ mean(.x > 0, na.rm = TRUE)), .groups = "drop")
                    
                    mcs_scores <- sapply(unique(top_genes$cluster), function(ct) {
                        m <- top_genes %>% filter(cluster == ct) %>% pull(gene)
                        m <- intersect(m, colnames(frac_df))
                        if (length(m) > 0 && ct %in% frac_df$ctpt_consensus_prediction) {
                            mean(as.numeric(frac_df[frac_df$ctpt_consensus_prediction == ct, m]), na.rm = TRUE)
                        } else { NA_real_ }
                    })
                    mean(mcs_scores, na.rm=TRUE) * 100
                } else { 0.0 }
            } else { 0.0 }
        } else { 0.0 }
    }, error = function(e) {
        cat(sprintf("     [WARNING] MCS calculation failed: %s\n", e$message))
        0.0
    })
    
    # === MPS CALCULATION (NEW) ===
    mps_result <- tryCatch({
        marker_db <- GLOBAL_ENV$MARKER_DB
        
        if (!is.null(marker_db) && length(marker_db) > 0) {
            calculate_mps(
                seurat_obj = seurat_final,
                marker_db = marker_db,
                group_by = "ctpt_consensus_prediction",
                n_top_genes = MPS_CONFIG$n_top_degs,
                verbose = TRUE
            )
        } else {
            list(mean_mps = NA, scores = NULL)
        }
    }, error = function(e) {
        cat(sprintf("     [WARNING] MPS calculation failed: %s\n", e$message))
        list(mean_mps = NA, scores = NULL)
    })
    
    mean_mps <- sanitize_metric(mps_result$mean_mps, NA)
    
    # Sanitize all metrics (MODIFIED)
    weighted_cas <- sanitize_metric(weighted_cas, 0.0)
    simple_cas <- sanitize_metric(simple_cas, 0.0)
    mean_mcs <- sanitize_metric(mean_mcs, 0.0)
    silhouette_avg <- sanitize_metric(silhouette_avg, 0.0)
    rescaled_silhouette <- sanitize_metric(rescaled_silhouette, 0.0)
    
    # Balanced score (MODIFIED to optionally include MPS)
    epsilon <- 1e-6
    balanced_score <- tryCatch({
        if (args$model_type == 'structural') {
            vals <- c(
                sanitize_metric(weighted_cas/100, epsilon),
                sanitize_metric(simple_cas/100, epsilon),
                sanitize_metric(mean_mcs/100, epsilon),
                sanitize_metric(rescaled_silhouette, epsilon)
            )
            vals <- pmax(vals, epsilon)
            (prod(vals) ^ (1/4)) * 100
        } else if (args$model_type == 'silhouette') {
            silhouette_avg * 100
        } else if (args$model_type == 'mps_integrated' && !is.na(mean_mps)) {
            vals <- c(
                sanitize_metric(weighted_cas/100, epsilon),
                sanitize_metric(simple_cas/100, epsilon),
                sanitize_metric(mean_mcs/100, epsilon),
                sanitize_metric(mean_mps/100, epsilon)
            )
            vals <- pmax(vals, epsilon)
            (prod(vals) ^ (1/4)) * 100
        } else {
            vals <- c(
                sanitize_metric(weighted_cas/100, epsilon),
                sanitize_metric(simple_cas/100, epsilon),
                sanitize_metric(mean_mcs/100, epsilon)
            )
            vals <- pmax(vals, epsilon)
            (prod(vals) ^ (1/3)) * 100
        }
    }, error = function(e) { 
        0.0 
    })
    
    balanced_score <- sanitize_metric(balanced_score, 0.0)
    
    # MODIFIED: Include MPS in metrics
    metrics <- list(
        weighted_mean_cas = weighted_cas, 
        simple_mean_cas = simple_cas, 
        mean_mcs = mean_mcs,
        mean_mps = mean_mps,  # NEW
        mps_details = mps_result$scores,  # NEW: per-cluster details
        silhouette_score_original = silhouette_avg,
        rescaled_silhouette_score = rescaled_silhouette, 
        balanced_score = balanced_score,
        n_individual_labels = tryCatch(n_distinct(seurat_final$ctpt_individual_prediction), error = function(e) 0),
        n_consensus_labels = tryCatch(n_distinct(seurat_final$ctpt_consensus_prediction), error = function(e) 0)
    )
    
    return(list(metrics = metrics, seurat_final = seurat_final))
}

#' Print final report
print_final_report <- function(target_name, params, metrics, winning_strategy) {
    args <- GLOBAL_ENV$ARGS
    
    target_title_map <- list(
        'weighted_cas' = "Weighted Mean CAS", 
        'simple_cas' = "Simple Mean CAS", 
        'mcs' = "Mean MCS", 
        'balanced' = ifelse(args$model_type == 'structural', "Balanced Score (CAS, MCS & Silhouette)", 
                            ifelse(args$model_type == 'silhouette', "Silhouette Score", "Balanced Score (CAS & MCS)"))
    )
    target_title <- target_title_map[[target_name]]
    
    cat("\n" %+% paste(rep("=", 60), collapse="") %+% "\n")
    cat(sprintf("--- Final Report for %s Optimization ---\n", target_title))
    cat(sprintf("--- (Best result found by '%s' strategy) ---\n", winning_strategy))
    cat("\n--- Optimal Parameters Found ---\n")
    
    params$n_hvg <- as.integer(params$n_hvg)
    params$n_pcs <- as.integer(params$n_pcs)
    params$n_neighbors <- as.integer(params$n_neighbors)
    
    for (key in names(params)) { 
        cat(sprintf("  - Best %s: %s\n", key, format(params[[key]], digits=3))) 
    }
    
    cat("\n--- Final Metrics for Optimal Parameters ---\n")
    cat(sprintf("  - Balanced Score: %.2f\n", metrics$balanced_score))
    cat(sprintf("  - Corresponding Weighted Mean CAS: %.2f%%\n", metrics$weighted_mean_cas))
    cat(sprintf("  - Corresponding Simple Mean CAS: %.2f%%\n", metrics$simple_mean_cas))
    cat(sprintf("  - Corresponding Mean MCS: %.2f%%\n", metrics$mean_mcs))
    if (!is.na(metrics$mean_mps)) {
        cat(sprintf("  - Corresponding Mean MPS (F1): %.2f%%\n", metrics$mean_mps))
    } else {
        cat(sprintf("  - Corresponding Mean MPS: N/A (no marker database)\n"))
    }
    
    cat(sprintf("  - Corresponding Silhouette Score: %.3f\n", metrics$silhouette_score_original))
    cat(sprintf("  - Final # of individual cell labels: %d\n", metrics$n_individual_labels))
    cat(sprintf("  - Final # of consensus cluster labels: %d\n", metrics$n_consensus_labels))
    cat(paste(rep("=", 60), collapse="") %+% "\n")
}

#' Save results to file
save_results_to_file <- function(output_path, target_name, params, metrics, winning_strategy) {
    args <- GLOBAL_ENV$ARGS
    
    target_title_map <- list(
        'weighted_cas' = "Weighted Mean CAS", 
        'simple_cas' = "Simple Mean CAS", 
        'mcs' = "Mean MCS", 
        'balanced' = ifelse(args$model_type == 'structural', 
                            "Balanced Score (Geometric Mean of CAS, MCS & Silhouette)", 
                            ifelse(args$model_type == 'silhouette', "Silhouette Score",
                                   "Balanced Score (Geometric Mean of CAS & MCS)"))
    )
    target_title <- target_title_map[[target_name]]
    
    lines <- c(
        "--- Bayesian Optimization Results ---", 
        paste("Annotation Method: Seurat Cross-Dataset Anchoring"), 
        paste("Species:", args$species), 
        paste("Optimization Model Type:", args$model_type), 
        paste("Marker Gene Model:", args$marker_gene_model),
        paste("CAS Aggregation Method:", args$cas_aggregation_method),
        paste("Optimization Target:", target_title), 
        paste("Winning Strategy:", winning_strategy), 
        paste("Random Seed Used:", GLOBAL_ENV$RANDOM_SEED), 
        "", 
        sapply(names(params), function(key) sprintf("Best %s: %s", key, format(params[[key]], digits=4))), 
        "", 
        sprintf("Highest_balanced_score: %.4f", metrics$balanced_score), 
        sprintf("Corresponding_weighted_mean_cas_pct: %.2f", metrics$weighted_mean_cas), 
        sprintf("Corresponding_simple_mean_cas_pct: %.2f", metrics$simple_mean_cas), 
        sprintf("Corresponding_mean_mcs_pct: %.2f", metrics$mean_mcs),
        # NEW: MPS line
        sprintf("Corresponding_mean_mps_pct: %s", 
                if (!is.na(metrics$mean_mps)) sprintf("%.2f", metrics$mean_mps) else "N/A"),
        sprintf("Corresponding_silhouette_score: %.4f", metrics$silhouette_score_original),
        sprintf("Corresponding_rescaled_silhouette_score: %.4f", metrics$rescaled_silhouette_score), 
        sprintf("Final_n_individual_labels: %d", metrics$n_individual_labels), 
        sprintf("Final_n_consensus_labels: %d", metrics$n_consensus_labels)
    )
    writeLines(lines, output_path)
}

#' Generate yield CSV report
generate_yield_csv <- function(results_dict, target_metric, output_dir, output_prefix) {
    cat("\n--- Generating consolidated yield CSV report ---\n")
    
    args <- GLOBAL_ENV$ARGS
    
    all_dfs <- list()
    for (name in names(results_dict)) {
        result <- results_dict[[name]]
        history_df <- as.data.frame(result$History)
        params_df <- history_df %>% dplyr::select(n_hvg, n_pcs, n_neighbors, resolution)
        
        if (!is.null(result$trial_metadata) && length(result$trial_metadata) == nrow(params_df)) {
            metadata_df <- bind_rows(result$trial_metadata)
            base_df <- bind_cols(params_df, metadata_df)
        } else {
            cat(sprintf("  [WARNING] Per-trial metadata mismatch for strategy '%s'. Metric columns will be empty.\n", name))
            base_df <- params_df
        }
        base_df$yield_score_target <- history_df$Value
        base_df$call_number <- 1:nrow(base_df)
        base_df$strategy <- name
        all_dfs[[name]] <- base_df
    }
    
    if (length(all_dfs) == 0) { 
        cat("  [ERROR] No results found to generate CSV. Skipping.\n")
        return() 
    }
    
    final_df <- bind_rows(all_dfs)
    epsilon <- 1e-6
    
    # MODIFIED: Include MPS in balanced score calculation
    required_cols <- c('weighted_mean_cas', 'simple_mean_cas', 'mean_mcs', 'silhouette_score')
    has_mps <- 'mean_mps' %in% names(final_df)
    
    if (all(required_cols %in% names(final_df))) {
        if (args$model_type == 'structural') {
            final_df$balanced_score_gmean <- with(final_df, 
                (((weighted_mean_cas/100+epsilon)*(simple_mean_cas/100+epsilon)*
                  (mean_mcs/100+epsilon)*(silhouette_score+epsilon))^(1/4)) * 100)
        } else if (args$model_type == 'silhouette') {
            final_df$balanced_score_gmean <- final_df$silhouette_score * 100
        } else if (args$model_type == 'mps_integrated' && has_mps) {
            # NEW: MPS-integrated balanced score
            final_df$balanced_score_gmean <- with(final_df, {
                mps_safe <- ifelse(is.na(mean_mps), 50, mean_mps)  # Use 50% as neutral if NA
                (((weighted_mean_cas/100+epsilon)*(simple_mean_cas/100+epsilon)*
                  (mean_mcs/100+epsilon)*(mps_safe/100+epsilon))^(1/4)) * 100
            })
        } else {
            final_df$balanced_score_gmean <- with(final_df, 
                (((weighted_mean_cas/100+epsilon)*(simple_mean_cas/100+epsilon)*
                  (mean_mcs/100+epsilon))^(1/3)) * 100)
        }
    } else {
        final_df$balanced_score_gmean <- NA
    }
    
    # MODIFIED: Updated column order to include MPS
    final_column_order <- c('call_number', 'strategy', 'n_hvg', 'n_pcs', 'n_neighbors', 'resolution', 
                            'yield_score_target', 'balanced_score_gmean', 'weighted_mean_cas', 
                            'simple_mean_cas', 'mean_mcs', 'mean_mps',  # NEW
                            'silhouette_score', 'n_individual_labels', 'n_consensus_labels')
    final_column_order <- intersect(final_column_order, names(final_df))
    final_df <- final_df[, final_column_order]
    
    output_path <- file.path(output_dir, paste0(output_prefix, "_", target_metric, "_yield_scores_report.csv"))
    write.csv(final_df, output_path, row.names = FALSE)
    cat(sprintf("✅ Success! Saved consolidated CSV report to: %s\n", output_path))
}

#' Plot optimizer convergence
plot_optimizer_convergence <- function(results, target_metric, output_dir, output_prefix) {
    cat("\n--- Generating convergence plot ---\n")
    
    args <- GLOBAL_ENV$ARGS
    
    convergence_data <- bind_rows(lapply(names(results), function(name) { 
        data.frame(call_number = 1:nrow(results[[name]]$History), 
                   score = results[[name]]$History$Value, 
                   strategy = name,
                   stringsAsFactors = FALSE) 
    }))
    
    best_so_far <- convergence_data %>% 
        group_by(strategy) %>% 
        arrange(call_number) %>% 
        mutate(best_score = cummax(score)) %>%
        ungroup()
    
    colors <- c('Exploit' = '#d62728', 'BO-EI' = "#fcbe06", 'Explore' = "#9015d2")
    
    title_map <- list(
        'weighted_cas' = 'Weighted Mean CAS', 
        'simple_cas' = 'Simple Mean CAS', 
        'mcs' = 'Mean MCS', 
        'balanced' = ifelse(args$model_type == 'structural', 
                            'Balanced Score (CAS, MCS & Silhouette)', 
                            ifelse(args$model_type == 'silhouette', 
                                   'Silhouette Score', 
                                   'Balanced Score (CAS & MCS)'))
    )
    
    p <- ggplot(best_so_far, aes(x = call_number, y = best_score, 
                                  color = strategy, group = strategy)) + 
        geom_line(linewidth = 1.5) + 
        geom_point(size = 3) + 
        scale_color_manual(values = colors, name = "Strategy") + 
        labs(title = "Bayesian Optimization Convergence", 
             subtitle = paste("Target:", title_map[[target_metric]]), 
             x = "Call Number (Experiment Iteration)", 
             y = "Best Score Found") + 
        theme_minimal(base_size = 14) + 
        theme(
            # Remove grid lines
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_blank(),
            
            # Square aspect ratio
            aspect.ratio = 1,
            
            # Axis styling
            axis.line = element_line(color = "black", linewidth = 0.5),
            axis.ticks = element_line(color = "black", linewidth = 0.3),
            
            # Title styling
            plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
            plot.subtitle = element_text(size = 12, hjust = 0.5),
            
            # Legend (kept)
            legend.position = "right"
        )
    
    output_path <- file.path(output_dir, paste0(output_prefix, "_", target_metric, 
                                                 "_optimizer_convergence.png"))
    ggsave(output_path, p, width = 10, height = 10, dpi = 300)
    cat(sprintf("✅ Success! Saved convergence plot to: %s\n", output_path))
}

# ==============================================================================
# --- VISUALIZATION PIPELINE ---
# ==============================================================================

#' Run visualization pipeline with optimal parameters
run_visualization_pipeline <- function(optimal_params, output_dir, seurat_input = NULL, data_dir = NULL) {
    
    cat("\n\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
    cat("### STAGE 2: RUNNING FIXED-PARAMETER VISUALIZATION PIPELINE ###\n")
    cat(paste(rep("=", 70), collapse="") %+% "\n")
    
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    
    args <- GLOBAL_ENV$ARGS
    random_seed <- GLOBAL_ENV$RANDOM_SEED
    
    # === CHECK FOR BATCHED DATA (NEW) ===
    is_batched <- GLOBAL_ENV$IS_BATCHED_DATA %||% FALSE
    batch_col <- GLOBAL_ENV$BATCH_COLUMN %||% NULL
    
    if (is_batched) {
        cat(sprintf("[Viz Pipeline] BATCHED MODE: Using Harmony integration (batch_col='%s')\n", batch_col))
    } else {
        cat("[Viz Pipeline] SINGLE-SAMPLE MODE: Standard PCA-based analysis\n")
    }
    
    params <- list(
        n_hvg = as.integer(optimal_params$n_hvg),
        n_pcs = as.integer(optimal_params$n_pcs),
        k = as.integer(optimal_params$n_neighbors),
        res = optimal_params$resolution,
        out_dir = output_dir,
        fig_dpi = args$fig_dpi
    )
    
    cat("Running Seurat Pipeline with Optimal Parameters\n")
    cat("--------------------------------------------------\n")
    cat("HVGs:", params$n_hvg, " PCs:", params$n_pcs, " k:", params$k,
        " Resolution:", params$res, " Output:", params$out_dir, "\n")
    cat("Random Seed:", random_seed, "\n")
    
    options(future.globals.maxSize = 8000 * 1024^2)
    
    # ==========================================================================
    # === DATA LOADING (unchanged) ===
    # ==========================================================================
    
    skip_qc_and_norm <- FALSE
    
    if (!is.null(seurat_input)) {
        cat("\n[Viz Pipeline] Using provided Seurat object (pre-processed from Stage 1)...\n")
        seurat_obj <- seurat_input
        seurat_obj <- ensure_standard_assay(seurat_obj)
        skip_qc_and_norm <- TRUE
        
        cat(sprintf("[Viz Pipeline] Pre-processed data: %d genes x %d cells\n", 
                    nrow(seurat_obj), ncol(seurat_obj)))
        cat("[Viz Pipeline] Skipping QC/normalization (already done, same as Stage 1)\n")
        
        if (!"percent.mt" %in% colnames(seurat_obj@meta.data)) {
            seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = MITO_REGEX_PATTERN)
        }
        
    } else if (!is.null(data_dir)) {
        # ... (data loading from scratch - unchanged) ...
        cat("\n[Viz Pipeline] Loading and preprocessing data from scratch...\n")
        seurat_obj.data <- load_expression_data(data_dir)
        
        if (inherits(seurat_obj.data, "Seurat")) {
            seurat_obj <- seurat_obj.data
            if (any(grepl("^ENSG|^ENSMUS", rownames(seurat_obj)))) {
                cat("[Viz Pipeline] Mapping gene names (Seurat Input)...\n")
                seurat_obj <- ensure_standard_assay(seurat_obj)
                counts_matrix <- GetAssayData(seurat_obj, layer = "counts")
                ensembl_ids <- gsub("\\..*$", "", rownames(counts_matrix))
                gene_symbols <- mapIds(GLOBAL_ENV$SPECIES_DB, keys = ensembl_ids, 
                                   column = "SYMBOL", keytype = "ENSEMBL", multiVals = "first")
                unmapped <- which(is.na(gene_symbols))
                gene_symbols[unmapped] <- rownames(counts_matrix)[unmapped]
                new_gene_names <- make.unique(as.character(gene_symbols))
                rownames(counts_matrix) <- new_gene_names
                seurat_obj <- CreateSeuratObject(counts = counts_matrix, 
                                         meta.data = seurat_obj@meta.data,
                                         project = "scRNA_viz", 
                                         min.cells = 0, min.features = 0)
            }
        } else {
            if (any(grepl("^ENSG|^ENSMUS", rownames(seurat_obj.data)))) {
                cat("[Viz Pipeline] Mapping gene names from ENSEMBL to SYMBOL (Matrix level)...\n")
                ensembl_ids_full <- rownames(seurat_obj.data)
                ensembl_ids <- gsub("\\..*$", "", ensembl_ids_full)
                gene_symbols <- mapIds(GLOBAL_ENV$SPECIES_DB, keys = ensembl_ids, 
                                   column = "SYMBOL", keytype = "ENSEMBL", multiVals = "first")
                unmapped_indices <- which(is.na(gene_symbols))
                gene_symbols[unmapped_indices] <- ensembl_ids_full[unmapped_indices]
                unique_gene_symbols <- make.unique(as.character(gene_symbols))
                rownames(seurat_obj.data) <- unique_gene_symbols
            }
            seurat_obj <- CreateSeuratObject(counts = seurat_obj.data, project = "scRNA_viz", 
                                             min.cells = MIN_CELLS_PER_GENE)
        }
        
        skip_qc_and_norm <- FALSE
        
    } else {
        stop("Must provide either seurat_input or data_dir")
    }
    
    # === STANDARDIZE GENE NAMES IN QUERY DATA ===
    cat("\n--- Standardizing Gene Names ---\n")
    
    original_gene_names <- rownames(seurat_obj)
    standardized_gene_names <- standardize_gene_names(original_gene_names, species = args$species)
    
    # Check for duplicates after standardization
    if (any(duplicated(standardized_gene_names))) {
        cat("[INFO] Resolving duplicate gene names after standardization...\n")
        standardized_gene_names <- make.unique(standardized_gene_names)
    }
    
    # Rename genes in Seurat object
    if (!identical(original_gene_names, standardized_gene_names)) {
        counts_data <- GetAssayData(seurat_obj, assay = "RNA", layer = "counts")
        rownames(counts_data) <- standardized_gene_names
        
        # Recreate assay with standardized names
        seurat_obj[["RNA"]] <- CreateAssayObject(counts = counts_data)
        DefaultAssay(seurat_obj) <- "RNA"
        
        n_changed <- sum(original_gene_names != standardized_gene_names)
        cat(sprintf("   -> Standardized %d gene names to %s format\n", 
                    n_changed, ifelse(args$species == "human", "UPPERCASE", "Title Case")))
    } else {
        cat("   -> Gene names already in standard format\n")
    }
    
    # === ALSO STANDARDIZE REFERENCE GENE NAMES ===
    cat("\n--- Standardizing Reference Gene Names ---\n")
    
    ref_gene_names <- rownames(GLOBAL_ENV$seurat_ref)
    ref_standardized <- standardize_gene_names(ref_gene_names, species = args$species)
    
    if (!identical(ref_gene_names, ref_standardized)) {
        if (any(duplicated(ref_standardized))) {
            ref_standardized <- make.unique(ref_standardized)
        }
        
        ref_counts <- GetAssayData(GLOBAL_ENV$seurat_ref, assay = "RNA", layer = "counts")
        rownames(ref_counts) <- ref_standardized
        GLOBAL_ENV$seurat_ref[["RNA"]] <- CreateAssayObject(counts = ref_counts)
        DefaultAssay(GLOBAL_ENV$seurat_ref) <- "RNA"
        
        # Re-run preprocessing on reference
        GLOBAL_ENV$seurat_ref <- NormalizeData(GLOBAL_ENV$seurat_ref, verbose = FALSE)
        GLOBAL_ENV$seurat_ref <- FindVariableFeatures(GLOBAL_ENV$seurat_ref, method = "vst", 
                                                       nfeatures = 2000, verbose = FALSE)
        GLOBAL_ENV$seurat_ref <- ScaleData(GLOBAL_ENV$seurat_ref, verbose = FALSE)
        GLOBAL_ENV$seurat_ref <- RunPCA(GLOBAL_ENV$seurat_ref, npcs = 105, verbose = FALSE)
        
        n_ref_changed <- sum(ref_gene_names != ref_standardized)
        cat(sprintf("   -> Standardized %d reference gene names\n", n_ref_changed))
    } else {
        cat("   -> Reference gene names already in standard format\n")
    }
    
    # Verify gene overlap after standardization
    query_genes <- rownames(seurat_obj)
    ref_genes <- rownames(GLOBAL_ENV$seurat_ref)
    common_genes <- intersect(query_genes, ref_genes)
    cat(sprintf("   -> Common genes after standardization: %d\n", length(common_genes)))
    
    if (length(common_genes) < 100) {
        warning("Very few common genes between query and reference. Check gene name formats.")
    }

    # ==========================================================================
    # === QC AND NORMALIZATION (unchanged) ===
    # ==========================================================================
    
    if (!skip_qc_and_norm) {
        seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize",
                              scale.factor = 10000, verbose = FALSE)
        seurat_obj <- ensure_standard_assay(seurat_obj)

        if (!"percent.mt" %in% colnames(seurat_obj@meta.data)) {
            seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = MITO_REGEX_PATTERN)
        }

        tryCatch({
            qc_plot <- VlnPlot(seurat_obj,
                                features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
                                ncol = 3)
            ggsave(file.path(params$out_dir,
                             paste0(args$final_run_prefix, "_qc_plots_before_filtering.png")),
                   plot = qc_plot, width = 14, height = 8, dpi = params$fig_dpi)
        }, error = function(e) {
            cat(sprintf("[WARNING] QC VlnPlot failed: %s\n", e$message))
        })

        seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > MIN_GENES_PER_CELL &
                           nFeature_RNA < MAX_GENES_PER_CELL &
                           percent.mt < MAX_PCT_COUNTS_MT)
        cat(sprintf("[Viz Pipeline] After QC: %d cells remain.\n", ncol(seurat_obj)))

        seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize",
                              scale.factor = 10000, verbose = FALSE)
        seurat_obj <- ensure_standard_assay(seurat_obj)
    }
    
    # ==========================================================================
    # === HVG SELECTION (unchanged) ===
    # ==========================================================================
    
    is_two_step_hvg <- !is.null(args$hvg_min_mean) && !is.null(args$hvg_max_mean) && !is.null(args$hvg_min_disp)

    if (is_two_step_hvg) {
        cat("[Viz Pipeline] Using two-step HVG selection (matching Stage 1)...\n")
        seurat_obj <- FindVariableFeatures(seurat_obj, method = "vst", 
                                           nfeatures = nrow(seurat_obj), verbose = FALSE)
        hvg_info <- HVFInfo(seurat_obj, method = "vst", assay = "RNA")
        hvg_info_filtered <- subset(hvg_info, 
                                    mean > args$hvg_min_mean & 
                                    mean < args$hvg_max_mean & 
                                    variance.standardized > args$hvg_min_disp)
        
        if (nrow(hvg_info_filtered) < 50) {
            cat("[Viz Pipeline] WARNING: Too few genes passed two-step filter. Falling back to standard.\n")
            seurat_obj <- FindVariableFeatures(seurat_obj, method = "vst", 
                                               nfeatures = params$n_hvg, verbose = FALSE)
        } else {
            hvg_info_sorted <- hvg_info_filtered[order(-hvg_info_filtered$variance.standardized), ]
            n_hvg_safe <- min(params$n_hvg, nrow(hvg_info_sorted))
            top_genes <- rownames(hvg_info_sorted)[1:n_hvg_safe]
            VariableFeatures(seurat_obj) <- top_genes
        }
    } else {
        cat(sprintf("[Viz Pipeline] Using standard HVG selection: nfeatures=%d\n", params$n_hvg))
        seurat_obj <- FindVariableFeatures(seurat_obj, method = "vst", 
                                           nfeatures = params$n_hvg, verbose = FALSE)
    }
    
    cat(sprintf("[Viz Pipeline] Selected %d HVGs\n", length(VariableFeatures(seurat_obj))))
    
    # Scale data
    all_genes_sorted <- sort(rownames(seurat_obj))
    seurat_obj <- ScaleData(seurat_obj, features = all_genes_sorted, verbose = FALSE)
    
    # === MPS CALCULATION AND EXPORT (NEW) ===
    cat("[Viz Pipeline] Calculating MPS (Marker Prior Score)...\n")
    
    mps_result <- tryCatch({
        marker_db <- GLOBAL_ENV$MARKER_DB
        
        if (!is.null(marker_db) && length(marker_db) > 0) {
            mps_calc <- calculate_mps(
                seurat_obj = seurat_obj,
                marker_db = marker_db,
                group_by = "ctpt_consensus_prediction",
                n_top_genes = MPS_CONFIG$n_top_degs,
                verbose = TRUE
            )
            
            # Export MPS details to CSV
            if (!is.null(mps_calc$scores) && length(mps_calc$scores) > 0) {
                mps_df <- bind_rows(lapply(names(mps_calc$scores), function(grp) {
                    s <- mps_calc$scores[[grp]]
                    data.frame(
                        cell_type = grp,
                        matched_marker_type = s$matched_type %||% NA,
                        n_top_degs = s$n_degs %||% 0,
                        n_canonical_markers = s$n_canonical %||% 0,
                        n_overlap = s$n_overlap %||% 0,
                        precision = s$precision %||% NA,
                        recall = s$recall %||% NA,
                        f1_score = s$f1 %||% NA,
                        overlapping_genes = paste(s$overlapping_genes, collapse = "; "),
                        stringsAsFactors = FALSE
                    )
                }))
                
                mps_csv_path <- file.path(params$out_dir, 
                    paste0(args$final_run_prefix, "_mps_per_celltype.csv"))
                write.csv(mps_df, mps_csv_path, row.names = FALSE)
                cat(sprintf("       -> Saved MPS details to: %s\n", mps_csv_path))
                cat(sprintf("       -> Mean MPS (F1): %.2f%%\n", mps_calc$mean_mps))
            }
            
            mps_calc
        } else {
            cat("       -> No marker database provided. MPS not calculated.\n")
            list(mean_mps = NA, scores = NULL)
        }
    }, error = function(e) {
        cat(sprintf("       -> [WARNING] MPS calculation failed: %s\n", e$message))
        list(mean_mps = NA, scores = NULL)
    })
    
    # Store MPS in Seurat object misc
    seurat_obj@misc$mps_result <- mps_result
    
    # ==========================================================================
    # === PCA + HARMONY INTEGRATION (MODIFIED FOR BATCHED DATA) ===
    # ==========================================================================
    
    n_pcs_to_compute <- min(N_PCS_FOR_PCA, ncol(seurat_obj) - 1, nrow(seurat_obj) - 1)
    n_pcs_to_use <- min(params$n_pcs, n_pcs_to_compute)
    
    cat(sprintf("[Viz Pipeline] Running PCA: computing %d PCs, using %d\n", n_pcs_to_compute, n_pcs_to_use))
    
    seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(seurat_obj), 
                         npcs = n_pcs_to_compute, verbose = FALSE,
                         seed.use = random_seed)
    
    # === HARMONY INTEGRATION FOR BATCHED DATA (NEW) ===
    reduction_to_use <- "pca"
    
    if (is_batched && !is.null(batch_col)) {
        cat(sprintf("[Viz Pipeline] Running Harmony integration (batch_col='%s')...\n", batch_col))
        
        harmony_result <- tryCatch({
            seurat_obj <- run_harmony_integration(
                seurat_obj, 
                batch_col = batch_col,
                dims_use = 1:n_pcs_to_use,
                verbose = TRUE
            )
            list(success = TRUE)
        }, error = function(e) {
            cat(sprintf("[Viz Pipeline] WARNING: Harmony failed: %s\n", e$message))
            cat("[Viz Pipeline] Falling back to PCA-based analysis.\n")
            list(success = FALSE)
        })
        
        if (harmony_result$success) {
            reduction_to_use <- "harmony"
        }
    }
    
    dims_to_use <- 1:n_pcs_to_use
    
    # ==========================================================================
    # === CLUSTERING (USE APPROPRIATE REDUCTION) ===
    # ==========================================================================
    
    k_param_safe <- min(params$k, ncol(seurat_obj) - 1)
    if (k_param_safe < 2) k_param_safe <- 2
    
    cat(sprintf("[Viz Pipeline] FindNeighbors: k.param=%d, reduction=%s\n", k_param_safe, reduction_to_use))
    
    seurat_obj <- FindNeighbors(seurat_obj, dims = dims_to_use, 
                                k.param = k_param_safe, verbose = FALSE,
                                reduction = reduction_to_use)  # <-- USE APPROPRIATE REDUCTION
    
    cat(sprintf("[Viz Pipeline] FindClusters: resolution=%.4f, algorithm=4 (Leiden), seed=%d\n", 
                params$res, random_seed))
    
    seurat_obj <- FindClusters(seurat_obj, resolution = params$res, 
                               algorithm = 4,
                               random.seed = random_seed,
                               verbose = FALSE)
    seurat_obj$leiden <- seurat_obj$seurat_clusters
    
    # UMAP with appropriate reduction
    seurat_obj <- RunUMAP(seurat_obj, dims = dims_to_use, 
                          seed.use = random_seed,
                          reduction = reduction_to_use,  # <-- USE APPROPRIATE REDUCTION
                          verbose = FALSE)
    
    n_clusters <- nlevels(seurat_obj$leiden)
    cat(sprintf("[Viz Pipeline] Clustering complete: %d clusters found\n", n_clusters))
    
    # Silhouette calculation with appropriate reduction
    silhouette_avg <- 0.0
    n_cells <- ncol(seurat_obj)
    if (1 < n_clusters && n_clusters < n_cells) {
        tryCatch({
            embedding_coords <- Embeddings(seurat_obj, reduction_to_use)[, dims_to_use]
            silhouette_avg <- mean(silhouette(as.integer(seurat_obj$leiden), 
                                              dist(embedding_coords))[, 3])
            cat(sprintf("       -> Average Silhouette Score: %.3f\n", silhouette_avg))
        }, error = function(e) {
            cat(sprintf("       -> [WARNING] Silhouette calculation error: %s\n", e$message))
        })
    }
    
    # === BATCH-SPECIFIC UMAP PLOT (NEW) ===
    if (is_batched && !is.null(batch_col)) {
        tryCatch({
            n_batches <- length(unique(seurat_obj@meta.data[[batch_col]]))
            batch_colors <- get_plot_colors(n_batches)
            
            umap_batch <- DimPlot(seurat_obj, reduction = "umap", 
                                  group.by = batch_col, 
                                  pt.size = 1,
                                  cols = batch_colors) +
                ggtitle(sprintf("UMAP by Batch (Harmony Integrated)\n%d batches", n_batches)) +
                labs(color = "Batch") +
                theme_umap_clean()
            
            ggsave(file.path(params$out_dir, paste0(args$final_run_prefix, "_umap_by_batch.png")), 
                   plot = umap_batch, width = 10, height = 10, dpi = params$fig_dpi)
            cat("       -> Saved batch UMAP plot\n")
        }, error = function(e) {
            cat(sprintf("[WARNING] Batch UMAP plot failed: %s\n", e$message))
        })
    }
    
    # Basic UMAP (unannotated) - unchanged
    tryCatch({
        cluster_colors <- get_plot_colors(n_clusters)
        
        umap_clusters <- DimPlot(seurat_obj, reduction = "umap", label = FALSE, 
                                 pt.size = 1, group.by = "leiden",
                                 cols = cluster_colors) +
            ggtitle(sprintf("Leiden Clusters (%d clusters)\nSilhouette: %.3f", 
                            n_clusters, silhouette_avg)) +
            labs(color = "Cluster") +
            theme_umap_clean()
        
        ggsave(file.path(params$out_dir, paste0(args$final_run_prefix, "_umap_leiden.png")), 
               plot = umap_clusters, width = 10, height = 10, dpi = params$fig_dpi)
    }, error = function(e) {
        cat(sprintf("[WARNING] Leiden UMAP plot failed: %s. Skipping.\n", e$message))
    })
    
    # ==========================================================================
    # === REST OF VISUALIZATION PIPELINE (unchanged from here) ===
    # ==========================================================================
    
    # Automated Annotation
    cat("[Viz Pipeline] Performing annotation via reference anchoring...\n")
    seurat_ref <- GLOBAL_ENV$seurat_ref
    ref_labels_col <- GLOBAL_ENV$REF_LABELS_COL
    
    k_weight_safe <- min(k_param_safe, ncol(seurat_obj) - 1, 50)
    if (k_weight_safe < 1) k_weight_safe <- 1
    
    cat(sprintf("[Viz Pipeline] FindTransferAnchors: dims=1:%d, k.weight=%d\n", 
                n_pcs_to_use, k_weight_safe))
    
    anchors <- FindTransferAnchors(reference = seurat_ref, query = seurat_obj, 
                                   dims = dims_to_use, 
                                   reduction = "pcaproject", 
                                   reference.assay = args$reference_assay,
                                   query.assay = "RNA",
                                   verbose = FALSE)
    
    predictions <- TransferData(anchorset = anchors, 
                                refdata = seurat_ref[[ref_labels_col, drop=TRUE]], 
                                dims = dims_to_use, 
                                k.weight = k_weight_safe,
                                weight.reduction = "pcaproject",
                                verbose = FALSE)
    
    seurat_obj <- AddMetaData(seurat_obj, metadata=predictions)
    seurat_obj$ctpt_individual_prediction <- predictions$predicted.id
    seurat_obj$ctpt_confidence <- predictions$prediction.score.max
    
    # ... (rest of visualization pipeline remains identical) ...
    # UMAP by predicted type
    tryCatch({
        n_types <- length(unique(seurat_obj$predicted.id))
        type_colors <- get_plot_colors(n_types)
        
        umap_pred <- DimPlot(seurat_obj, reduction = "umap", group.by = "predicted.id", 
                             label = FALSE, pt.size = 1,
                             cols = type_colors) +
            ggtitle(sprintf("Per-Cell Annotation (%d types)", n_types)) +
            labs(color = "Cell Type") +
            theme_umap_clean()
        
        ggsave(file.path(params$out_dir, paste0(args$final_run_prefix, "_umap_per_cell_seurat.png")), 
               plot = umap_pred, width = 12, height = 10, dpi = params$fig_dpi)
    }, error = function(e) {
        cat(sprintf("[WARNING] Per-cell UMAP plot failed: %s. Skipping.\n", e$message))
    })
    
    # Consensus Annotation + CAS Scores
    metadata_df <- seurat_obj@meta.data

    if (!"ctpt_individual_prediction" %in% colnames(metadata_df)) {
        cat("[WARNING] 'ctpt_individual_prediction' column not found. Skipping CAS calculation.\n")
        seurat_obj$ctpt_consensus_prediction <- NA
        cas_leiden_output_path <- NULL
        cas_consensus_output_path <- NULL
    } else {
        metadata_df <- seurat_obj@meta.data %>%
            group_by(leiden) %>%
            mutate(ctpt_consensus_prediction = names(which.max(table(ctpt_individual_prediction)))) %>%
            ungroup()
        
        seurat_obj$ctpt_consensus_prediction <- metadata_df$ctpt_consensus_prediction
        
        n_consensus_labels <- n_distinct(seurat_obj$ctpt_consensus_prediction)
        cat(sprintf("[Viz Pipeline] Consensus annotation: %d unique cell types\n", n_consensus_labels))
        
        # Leiden-based CAS
        leiden_cas_df <- tryCatch({
            metadata_df %>%
                group_by(leiden) %>%
                summarise(
                    `Cluster_ID (Leiden)` = dplyr::first(as.character(leiden)),
                    Consensus_Cell_Type = dplyr::first(ctpt_consensus_prediction),
                    Total_Cells_in_Group = n(),
                    Matching_Individual_Predictions = sum(
                        ctpt_individual_prediction == ctpt_consensus_prediction, 
                        na.rm = TRUE
                    ),
                    `Cluster_Annotation_Score_CAS (%)` = 100 * Matching_Individual_Predictions / Total_Cells_in_Group,
                    .groups = "drop"
                ) %>%
                arrange(desc(`Cluster_Annotation_Score_CAS (%)`))
        }, error = function(e) {
            cat(sprintf("[WARNING] Leiden CAS calculation failed: %s\n", e$message))
            NULL
        })
        
        if (!is.null(leiden_cas_df)) {
            cas_leiden_output_path <- file.path(params$out_dir, 
                paste0(args$final_run_prefix, "_leiden_cluster_annotation_scores.csv"))
            write.csv(leiden_cas_df, cas_leiden_output_path, row.names = FALSE, quote = TRUE)
            cat(sprintf("       -> Saved Leiden-based CAS to: %s\n", cas_leiden_output_path))
        } else {
            cas_leiden_output_path <- NULL
        }
        
        # Consensus-based CAS
        consensus_cas_df <- tryCatch({
            metadata_df %>%
                group_by(ctpt_consensus_prediction) %>%
                summarise(
                    Consensus_Cell_Type = dplyr::first(ctpt_consensus_prediction),
                    Total_Cells_in_Group = n(),
                    Matching_Individual_Predictions = sum(
                        ctpt_individual_prediction == ctpt_consensus_prediction, 
                        na.rm = TRUE
                    ),
                    `Cluster_Annotation_Score_CAS (%)` = 100 * Matching_Individual_Predictions / Total_Cells_in_Group,
                    .groups = "drop"
                ) %>%
                arrange(desc(`Cluster_Annotation_Score_CAS (%)`))
        }, error = function(e) {
            cat(sprintf("[WARNING] Consensus CAS calculation failed: %s\n", e$message))
            NULL
        })
        
        if (!is.null(consensus_cas_df)) {
            cas_consensus_output_path <- file.path(params$out_dir, 
                paste0(args$final_run_prefix, "_consensus_group_annotation_scores.csv"))
            write.csv(consensus_cas_df, cas_consensus_output_path, row.names = FALSE, quote = TRUE)
            cat(sprintf("       -> Saved Consensus-based CAS to: %s\n", cas_consensus_output_path))
            cat(sprintf("       -> Consensus cell types in CAS table: %d\n", nrow(consensus_cas_df)))
        } else {
            cas_consensus_output_path <- NULL
        }
    }

    if (args$cas_aggregation_method == 'leiden') {
        cas_path_for_refinement <- cas_leiden_output_path
    } else {
        cas_path_for_refinement <- cas_consensus_output_path
    }
    
    # Save all annotations (ADD BATCH INFO TO EXPORT)
    annotation_cols <- c("seurat_clusters", "leiden", "predicted.id", "prediction.score.max", 
                         "ctpt_individual_prediction", "ctpt_confidence", "ctpt_consensus_prediction")
    
    # Add batch column if present
    if (is_batched && !is.null(batch_col) && batch_col %in% colnames(seurat_obj@meta.data)) {
        annotation_cols <- c(batch_col, annotation_cols)
    }
    
    available_cols <- intersect(annotation_cols, colnames(seurat_obj@meta.data))
    write.csv(seurat_obj@meta.data[, available_cols, drop=FALSE], 
              file.path(params$out_dir, paste0(args$final_run_prefix, "_all_annotations.csv")), 
              row.names=TRUE, quote=TRUE)
    
    # UMAP by consensus type
    tryCatch({
        n_consensus_types <- length(unique(seurat_obj$ctpt_consensus_prediction))
        consensus_colors <- get_plot_colors(n_consensus_types)
        
        umap_consensus <- DimPlot(seurat_obj, reduction = "umap", 
                                  group.by = "ctpt_consensus_prediction", 
                                  label = FALSE, pt.size = 1,
                                  cols = consensus_colors) +
            ggtitle(sprintf("Cluster-Consensus Annotation (%d types)", n_consensus_types)) +
            labs(color = "Cell Type") +
            theme_umap_clean()
        
        ggsave(file.path(params$out_dir, paste0(args$final_run_prefix, "_cluster_seurat_umap.png")), 
               plot = umap_consensus, width = 12, height = 10, dpi = params$fig_dpi)
    }, error = function(e) {
        cat(sprintf("[WARNING] Consensus UMAP plot failed: %s. Skipping.\n", e$message))
    })
    
    # Save Final Object (STORE BATCH INFO IN MISC)
    seurat_obj@misc$integration_info <- list(
        is_batched = is_batched,
        batch_column = batch_col,
        reduction_used = reduction_to_use,
        harmony_applied = (reduction_to_use == "harmony")
    )
    
    saveRDS(seurat_obj, file=file.path(params$out_dir, paste0(args$final_run_prefix, "_final_processed.rds")))
    cat("\n✅ Visualization pipeline complete. All results saved to:", params$out_dir, "\n")
    
    return(list(seurat = seurat_obj, cas_path = cas_path_for_refinement))
}

# ==============================================================================
# --- REFINEMENT PIPELINE ---
# ==============================================================================

#' Run iterative refinement pipeline with graceful early stopping
#' @description Matches Python script's refinement logic exactly:
#'   - Identifies failing clusters/cell types below CAS threshold
#'   - Subsets and re-optimizes those populations
#'   - Updates combined_annotation with refined predictions
#'   - Stores depth-specific columns for tri-consensus filtering
#' @param args Pipeline arguments
#' @param seurat_s2 Seurat object from Stage 2
#' @param cas_csv_path_s2 Path to CAS CSV from Stage 2
#' @return Updated Seurat object with refinement annotations
run_iterative_refinement_pipeline <- function(args, seurat_s2, cas_csv_path_s2, output_base_dir = NULL) {
    cat("\n" %+% paste(rep("=", 80), collapse="") %+% "\n")
    cat("### ITERATIVE REFINEMENT PIPELINE ###\n")
    cat(paste(rep("=", 80), collapse="") %+% "\n")
    
    # === DETERMINE OUTPUT DIRECTORIES ===
    # If output_base_dir is provided, use it for both optimization and final outputs
    # Otherwise, use default Stage 1 and Stage 2 directories
    if (!is.null(output_base_dir)) {
        # Unified output mode (e.g., Stage 1 full analysis)
        main_stage1_dir <- file.path(output_base_dir, "optimization")
        stage2_output_dir <- output_base_dir
        cat(sprintf("   -> Output mode: Unified (base: %s)\n", output_base_dir))
    } else {
        # Default split output mode (Stage 1 optimization, Stage 2 final)
        main_stage1_dir <- file.path(args$output_dir, "stage_1_bayesian_optimization")
        stage2_output_dir <- file.path(args$output_dir, "stage_2_final_analysis")
        cat(sprintf("   -> Output mode: Split (Stage 1: %s, Stage 2: %s)\n", 
                    basename(main_stage1_dir), basename(stage2_output_dir)))
    }
    
    # Ensure directories exist
    dir.create(main_stage1_dir, showWarnings = FALSE, recursive = TRUE)
    dir.create(stage2_output_dir, showWarnings = FALSE, recursive = TRUE)
    
    current_cas_csv_path <- cas_csv_path_s2
    seurat_to_check <- seurat_s2
    
    all_refinement_cas_paths <- list()
    refinement_results <- list()
    
    # === CRITICAL: Initialize combined_annotation with consensus predictions ===
    # This matches Python: adata.obs['combined_annotation'] = adata.obs['ctpt_consensus_prediction'].copy()
    seurat_s2$combined_annotation <- as.character(seurat_s2$ctpt_consensus_prediction)
    
    # Track which depths completed successfully
    max_completed_depth <- 0
    early_stop_reason <- NULL
    
    # Store original prefix for restoration
    original_final_run_prefix <- args$final_run_prefix
    
    # === MINIMUM CELLS FOR RELIABLE ANNOTATION ===
    # FindTransferAnchors requires sufficient cells for anchor finding
    MIN_CELLS_FOR_ANNOTATION <- 100  # Minimum cells for reliable transfer learning
    
    # === MAIN REFINEMENT LOOP ===
    # Matches Python: for depth in range(1, refinement_depth + 1):
    for (depth in 1:args$refinement_depth) {
        cat(sprintf("\n--- Refinement Depth %d of %d ---\n", depth, args$refinement_depth))
        
        # === CHECK 1: Read and validate CAS file ===
        # Matches Python: cas_df_prev_level = pd.read_csv(current_cas_csv_path)
        cas_df_prev_level <- tryCatch({
            read.csv(current_cas_csv_path, stringsAsFactors = FALSE, check.names = FALSE)
        }, error = function(e) {
            cat(sprintf("[ERROR] Cannot read CAS file: %s\n", e$message))
            NULL
        })
        
        if (is.null(cas_df_prev_level) || nrow(cas_df_prev_level) == 0) {
            cat(sprintf("[STOP] Empty or invalid CAS file at depth %d. Stopping refinement.\n", depth))
            early_stop_reason <- sprintf("Invalid CAS file at depth %d", depth)
            break
        }
        
        # === FIND CAS COLUMN ===
        # Matches Python logic for finding the CAS score column
        cas_col <- NULL
        cas_col_candidates <- c(
            "Cluster_Annotation_Score_CAS....", 
            "Cluster_Annotation_Score_CAS (%)", 
            "Cluster_Annotation_Score_CAS_pct", 
            "CAS",
            "Cluster_Annotation_Score_CAS"
        )
        
        for (candidate in cas_col_candidates) {
            if (candidate %in% colnames(cas_df_prev_level)) {
                cas_col <- candidate
                break
            }
        }
        
        if (is.null(cas_col)) {
            # Fallback: search by pattern
            cas_col_match <- grep("CAS|Score", colnames(cas_df_prev_level), 
                                  value = TRUE, ignore.case = TRUE)
            if (length(cas_col_match) > 0) cas_col <- cas_col_match[1]
        }
        
        if (is.null(cas_col)) {
            cat(sprintf("[STOP] Cannot find CAS column at depth %d. Stopping refinement.\n", depth))
            early_stop_reason <- sprintf("CAS column not found at depth %d", depth)
            break
        }
        
        cat(sprintf("   -> Using CAS column: '%s'\n", cas_col))
        
        # === IDENTIFY FAILING CLUSTERS/GROUPS ===
        # Matches Python logic based on cas_aggregation_method
        if (args$cas_aggregation_method == 'leiden') {
            # Find Leiden cluster column
            leiden_col <- NULL
            leiden_col_candidates <- c(
                "Cluster_ID..Leiden.", 
                "Cluster_ID (Leiden)", 
                "Cluster_ID_Leiden",
                "Cluster_ID"
            )
            
            for (candidate in leiden_col_candidates) {
                if (candidate %in% colnames(cas_df_prev_level)) {
                    leiden_col <- candidate
                    break
                }
            }
            
            if (is.null(leiden_col)) {
                leiden_col_match <- grep("Leiden|Cluster_ID", colnames(cas_df_prev_level), value = TRUE)
                if (length(leiden_col_match) > 0) leiden_col <- leiden_col_match[1]
            }
            
            if (is.null(leiden_col)) {
                cat(sprintf("[STOP] Cannot find Leiden cluster column at depth %d.\n", depth))
                early_stop_reason <- sprintf("Leiden column not found at depth %d", depth)
                break
            }
            
            # Get failing cluster IDs
            # Matches Python: failing_ids = cas_df[cas_df[cas_col] < threshold][id_col].tolist()
            failing_mask <- cas_df_prev_level[[cas_col]] < args$cas_refine_threshold
            failing_ids <- as.character(cas_df_prev_level[[leiden_col]][failing_mask])
            
            cat(sprintf("   -> Found %d failing clusters (CAS < %.0f%%)\n", 
                        length(failing_ids), args$cas_refine_threshold))
            
            if (length(failing_ids) == 0) {
                cat(sprintf("✅ All clusters meet %.0f%% CAS threshold. Refinement complete.\n", 
                            args$cas_refine_threshold))
                max_completed_depth <- depth
                break
            }
            
            # Get cell barcodes from failing clusters
            # Matches Python: failing_barcodes = adata.obs[adata.obs['leiden'].isin(failing_ids)].index.tolist()
            failing_cell_barcodes <- rownames(seurat_to_check@meta.data)[
                as.character(seurat_to_check$leiden) %in% failing_ids
            ]
            
        } else {
            # Consensus-based aggregation
            # Get failing cell types
            failing_types <- cas_df_prev_level$Consensus_Cell_Type[
                cas_df_prev_level[[cas_col]] < args$cas_refine_threshold
            ]
            
            cat(sprintf("   -> Found %d failing cell types (CAS < %.0f%%)\n", 
                        length(failing_types), args$cas_refine_threshold))
            
            if (length(failing_types) == 0) {
                cat(sprintf("✅ All cell types meet %.0f%% CAS threshold. Refinement complete.\n", 
                            args$cas_refine_threshold))
                max_completed_depth <- depth
                break
            }
            
            # Get cell barcodes from failing cell types
            failing_cell_barcodes <- rownames(seurat_to_check@meta.data)[
                seurat_to_check$ctpt_consensus_prediction %in% failing_types
            ]
        }
        
        # === CHECK 2: Minimum cells for refinement (user-specified threshold) ===
        # Matches Python: if len(failing_barcodes) < min_cells_refinement:
        if (length(failing_cell_barcodes) < args$min_cells_refinement) {
            cat(sprintf("[STOP] Only %d failing cells (< minimum %d). Stopping refinement at depth %d.\n", 
                        length(failing_cell_barcodes), args$min_cells_refinement, depth))
            early_stop_reason <- sprintf("Insufficient cells (%d < %d) at depth %d", 
                                         length(failing_cell_barcodes), args$min_cells_refinement, depth)
            break
        }
        
        # === CHECK 3: Minimum cells for reliable annotation (hard threshold) ===
        if (length(failing_cell_barcodes) < MIN_CELLS_FOR_ANNOTATION) {
            cat(sprintf("[STOP] Only %d failing cells (< minimum %d for reliable annotation).\n", 
                        length(failing_cell_barcodes), MIN_CELLS_FOR_ANNOTATION))
            cat(sprintf("   -> FindTransferAnchors requires sufficient cells for anchor finding.\n"))
            cat(sprintf("   -> Stopping refinement at depth %d to avoid annotation failures.\n", depth))
            early_stop_reason <- sprintf("Subset too small for reliable annotation (%d < %d) at depth %d", 
                                         length(failing_cell_barcodes), MIN_CELLS_FOR_ANNOTATION, depth)
            break
        }
        
        cat(sprintf("   -> Isolated %d cells for refinement at depth %d\n", 
                    length(failing_cell_barcodes), depth))
        
        # === SUBSET FOR REFINEMENT ===
        # Matches Python: adata_refine = adata[failing_barcodes].copy()
        # IMPORTANT: Subset from ORIGINAL Stage 2 object, not iteratively refined object
        seurat_refine <- subset(seurat_s2, cells = failing_cell_barcodes)
        
        # === CHECK 4: Verify actual cell count after subset ===
        if (ncol(seurat_refine) < args$min_cells_refinement) {
            cat(sprintf("[STOP] Only %d cells in subset (< minimum %d). Stopping refinement at depth %d.\n", 
                        ncol(seurat_refine), args$min_cells_refinement, depth))
            early_stop_reason <- sprintf("Subset too small (%d cells) at depth %d", 
                                         ncol(seurat_refine), depth)
            break
        }
        
        # === CHECK 5: Hard threshold for annotation reliability ===
        if (ncol(seurat_refine) < MIN_CELLS_FOR_ANNOTATION) {
            cat(sprintf("[STOP] Subset has only %d cells (< %d minimum for annotation).\n", 
                        ncol(seurat_refine), MIN_CELLS_FOR_ANNOTATION))
            cat(sprintf("   -> Stopping refinement at depth %d.\n", depth))
            early_stop_reason <- sprintf("Subset too small for annotation (%d cells) at depth %d", 
                                         ncol(seurat_refine), depth)
            break
        }
        
        # === NORMALIZE AND CONVERT ASSAY ===
        # Matches Python: sc.pp.normalize_total(adata_refine); sc.pp.log1p(adata_refine)
        seurat_refine <- NormalizeData(seurat_refine, normalization.method = "LogNormalize",
                                       scale.factor = 10000, verbose = FALSE)
        seurat_refine <- ensure_standard_assay(seurat_refine)
        
        # === CHECK 6: Filter genes with sufficient expression BEFORE HVG selection ===
        counts_mat <- GetAssayData(seurat_refine, assay = "RNA", layer = "counts")
        
        gene_detection_rate <- Matrix::rowSums(counts_mat > 0) / ncol(counts_mat)
        gene_cell_counts <- Matrix::rowSums(counts_mat > 0)
        
        genes_to_keep <- names(gene_detection_rate[gene_detection_rate >= 0.01])
        genes_with_min_cells <- names(gene_cell_counts[gene_cell_counts >= 3])
        genes_to_keep <- intersect(genes_to_keep, genes_with_min_cells)
        
        MIN_HVG_LIMIT <- 200
        if (length(genes_to_keep) < MIN_HVG_LIMIT) {
            cat(sprintf("[STOP] Only %d expressed genes (< minimum %d). Stopping refinement at depth %d.\n", 
                        length(genes_to_keep), MIN_HVG_LIMIT, depth))
            early_stop_reason <- sprintf("Insufficient genes (%d < %d) at depth %d", 
                                         length(genes_to_keep), MIN_HVG_LIMIT, depth)
            break
        }
        
        seurat_refine <- subset(seurat_refine, features = genes_to_keep)
        cat(sprintf("   -> Refinement subset: %d cells x %d genes\n", 
                    ncol(seurat_refine), nrow(seurat_refine)))
        
        # === CHECK 7: Test HVG selection before proceeding ===
        hvg_test_passed <- FALSE
        hvg_error_msg <- NULL
        
        tryCatch({
            test_n_hvg <- min(500, nrow(seurat_refine) - 1)
            seurat_refine <- FindVariableFeatures(seurat_refine, method = "vst", 
                                                   nfeatures = test_n_hvg, verbose = FALSE)
            n_hvg_found <- length(VariableFeatures(seurat_refine))
            
            if (n_hvg_found >= 100) {
                hvg_test_passed <- TRUE
                cat(sprintf("   -> HVG test passed: %d variable features found\n", n_hvg_found))
            } else {
                cat(sprintf("[STOP] Only %d HVGs found (< minimum 100). Stopping refinement at depth %d.\n", 
                            n_hvg_found, depth))
                hvg_error_msg <- sprintf("Insufficient HVGs (%d) at depth %d", n_hvg_found, depth)
            }
        }, error = function(e) {
            cat(sprintf("[STOP] HVG selection failed: %s. Stopping refinement at depth %d.\n", 
                        e$message, depth))
            hvg_error_msg <<- sprintf("HVG selection error at depth %d: %s", depth, e$message)
        })
        
        if (!hvg_test_passed) {
            if (!is.null(hvg_error_msg)) {
                early_stop_reason <- hvg_error_msg
            }
            break
        }
        
        # === CHECK 8: Pre-test FindTransferAnchors viability ===
        anchors_viable <- FALSE
        anchor_error_msg <- NULL
        
        cat("   -> Testing annotation viability (FindTransferAnchors)...\n")
        tryCatch({
            seurat_ref <- GLOBAL_ENV$seurat_ref
            
            # Check gene overlap with reference
            common_genes <- intersect(rownames(seurat_refine), rownames(seurat_ref))
            
            if (length(common_genes) < 100) {
                cat(sprintf("[STOP] Only %d genes overlap with reference (< 100 minimum).\n", 
                            length(common_genes)))
                anchor_error_msg <- sprintf("Insufficient gene overlap (%d) at depth %d", 
                                            length(common_genes), depth)
            } else {
                # Try a quick anchor test with reduced dimensions
                test_dims <- min(10, ncol(seurat_refine) - 1, 30)
                
                if (test_dims < 5) {
                    cat(sprintf("[STOP] Insufficient dimensions for PCA (%d < 5).\n", test_dims))
                    anchor_error_msg <- sprintf("Insufficient PCA dimensions at depth %d", depth)
                } else {
                    # Perform quick PCA to verify it works
                    seurat_refine <- ScaleData(seurat_refine, 
                                               features = VariableFeatures(seurat_refine), 
                                               verbose = FALSE)
                    seurat_refine <- RunPCA(seurat_refine, 
                                            features = VariableFeatures(seurat_refine),
                                            npcs = test_dims, verbose = FALSE,
                                            seed.use = GLOBAL_ENV$RANDOM_SEED)
                    
                    # Test anchor finding
                    test_anchors <- FindTransferAnchors(
                        reference = seurat_ref, 
                        query = seurat_refine, 
                        dims = 1:test_dims, 
                        reduction = "pcaproject",
                        reference.assay = args$reference_assay, 
                        query.assay = "RNA", 
                        verbose = FALSE
                    )
                    
                    if (is.null(test_anchors) || length(test_anchors@anchors) == 0) {
                        cat("[STOP] FindTransferAnchors produced no anchors.\n")
                        anchor_error_msg <- sprintf("No anchors found at depth %d", depth)
                    } else {
                        n_anchors <- nrow(test_anchors@anchors)
                        cat(sprintf("   -> Anchor test passed: %d anchors found\n", n_anchors))
                        
                        if (n_anchors < 10) {
                            cat(sprintf("[WARNING] Very few anchors (%d). Annotation may be unreliable.\n", n_anchors))
                        }
                        anchors_viable <- TRUE
                    }
                }
            }
        }, error = function(e) {
            cat(sprintf("[STOP] FindTransferAnchors test failed: %s\n", e$message))
            anchor_error_msg <<- sprintf("Anchor finding failed at depth %d: %s", depth, e$message)
        })
        
        if (!anchors_viable) {
            if (!is.null(anchor_error_msg)) {
                early_stop_reason <- anchor_error_msg
            } else {
                early_stop_reason <- sprintf("Annotation viability check failed at depth %d", depth)
            }
            cat(sprintf("   -> Stopping refinement at depth %d to avoid downstream failures.\n", depth))
            break
        }
        
        # === RUN STAGE 1 (BO) ON SUBSET ===
        # Matches Python: run optimization on subset
        stage1_refinement_dir <- file.path(main_stage1_dir, sprintf("refinement_depth_%d", depth))
        dir.create(stage1_refinement_dir, showWarnings = FALSE, recursive = TRUE)
        
        # Update global objects for optimization
        GLOBAL_ENV$seurat_base <- seurat_refine
        GLOBAL_ENV$seurat_full_data <- seurat_refine
        GLOBAL_ENV$OPTIMIZATION_CACHE <- new.env(hash = TRUE)
        GLOBAL_ENV$TRIAL_METADATA <- list()
        
        cat(sprintf("   -> Refinement optimization will use %d cells\n", ncol(seurat_refine)))
        
        cat(sprintf("\n--- [Depth %d] Running Bayesian Optimization on subset ---\n", depth))
        
        # Run optimization strategies
        target <- if (args$target == 'all') 'balanced' else args$target
        GLOBAL_ENV$CURRENT_OPTIMIZATION_TARGET <- target
        
        strategies <- list(
            "Exploit" = list(acq = 'poi', kappa = 2.576, eps = 0.0), 
            "BO-EI"   = list(acq = 'ei',  kappa = 2.576, eps = 0.0), 
            "Explore" = list(acq = 'ei',  kappa = 2.576, eps = 0.1)
        )
        
        results <- list()
        for (name in names(strategies)) {
            cat(sprintf("   -> Running Strategy: %s\n", name))
            GLOBAL_ENV$CURRENT_STRATEGY_NAME <- name
            GLOBAL_ENV$TRIAL_METADATA <- list()
            
            opt_result <- tryCatch({
                safe_bayesian_optimization(
                    FUN = objective_function, 
                    bounds = SEARCH_SPACE_BOUNDS, 
                    init_points = max(3, args$n_init_points %/% 2),
                    n_iter = max(5, (args$n_calls - args$n_init_points) %/% 2), 
                    acq = strategies[[name]]$acq, 
                    kappa = strategies[[name]]$kappa, 
                    eps = strategies[[name]]$eps, 
                    verbose = FALSE,
                    max_total_failures = args$n_calls,
                    seed = GLOBAL_ENV$RANDOM_SEED
                )
            }, error = function(e) {
                cat(sprintf("     [ERROR] Refinement strategy '%s' failed: %s\n", name, e$message))
                NULL
            })
            
            if (!is.null(opt_result) && !is.null(opt_result$Best_Value) && 
                is.finite(opt_result$Best_Value)) {
                opt_result$trial_metadata <- GLOBAL_ENV$TRIAL_METADATA
                results[[name]] <- opt_result
            }
        }
        
        # Find best result
        if (length(results) == 0) {
            cat(sprintf("[STOP] All optimization strategies failed at depth %d. Stopping refinement.\n", depth))
            early_stop_reason <- sprintf("All optimization strategies failed at depth %d", depth)
            break
        }
        
        best_overall_score <- -Inf
        best_params <- NULL
        for (name in names(results)) {
            if (!is.null(results[[name]]) && results[[name]]$Best_Value > best_overall_score) {
                best_overall_score <- results[[name]]$Best_Value
                best_params <- as.list(results[[name]]$Best_Par)
            }
        }
        
        # === RUN STAGE 2 ON SUBSET ===
        stage2_refinement_dir <- file.path(stage2_output_dir, sprintf("refinement_depth_%d", depth))
        dir.create(stage2_refinement_dir, showWarnings = FALSE, recursive = TRUE)
        
        cat(sprintf("\n--- [Depth %d] Running Final Analysis on subset ---\n", depth))
        
        args$final_run_prefix <- sprintf("%s_refinement_depth_%d", original_final_run_prefix, depth)
        
        viz_result <- tryCatch({
            run_visualization_pipeline(
                optimal_params = best_params,
                output_dir = stage2_refinement_dir,
                seurat_input = seurat_refine
            )
        }, error = function(e) {
            cat(sprintf("[ERROR] Visualization pipeline failed at depth %d: %s\n", depth, e$message))
            NULL
        })
        
        args$final_run_prefix <- original_final_run_prefix
        
        if (is.null(viz_result)) {
            cat(sprintf("[STOP] Visualization failed at depth %d. Stopping refinement.\n", depth))
            early_stop_reason <- sprintf("Visualization pipeline failed at depth %d", depth)
            break
        }
        
        seurat_refinement_processed <- viz_result$seurat
        cas_csv_path_refinement <- viz_result$cas_path
        
        # === CHECK 9: Verify annotation columns exist after visualization ===
        required_annotation_cols <- c("ctpt_individual_prediction", "ctpt_consensus_prediction")
        missing_annotation_cols <- setdiff(required_annotation_cols, 
                                           colnames(seurat_refinement_processed@meta.data))
        
        if (length(missing_annotation_cols) > 0) {
            cat(sprintf("[STOP] Refinement annotation incomplete at depth %d.\n", depth))
            cat(sprintf("   -> Missing columns: %s\n", paste(missing_annotation_cols, collapse = ", ")))
            early_stop_reason <- sprintf("Annotation columns missing at depth %d: %s", 
                                          depth, paste(missing_annotation_cols, collapse = ", "))
            break
        }
        
        # Store refinement CAS path and results
        all_refinement_cas_paths[[depth]] <- cas_csv_path_refinement
        refinement_results[[depth]] <- list(
            seurat = seurat_refinement_processed,
            cas_path = cas_csv_path_refinement,
            best_params = best_params,
            failing_barcodes = failing_cell_barcodes
        )
        
        # === UPDATE MASTER ANNOTATION WITH REFINED PREDICTIONS ===
        # Matches Python logic:
        #   refined_annotations = adata_refined.obs['ctpt_consensus_prediction']
        #   for barcode in refined_barcodes:
        #       adata.obs.loc[barcode, 'combined_annotation'] = refined_annotations[barcode]
        
        refinement_annotations <- as.character(seurat_refinement_processed$ctpt_consensus_prediction)
        names(refinement_annotations) <- colnames(seurat_refinement_processed)
        
        # Store refined annotations in a depth-specific column for tri-consensus
        depth_col_name <- sprintf("refined_depth_%d", depth)
        
        # Initialize column properly in metadata
        seurat_s2@meta.data[[depth_col_name]] <- NA_character_
        
        # Get barcodes and find matching indices
        refined_barcodes <- names(refinement_annotations)
        all_barcodes <- rownames(seurat_s2@meta.data)
        
        # Update annotations - CRITICAL: This updates combined_annotation
        # Matches Python: adata.obs.loc[barcode, 'combined_annotation'] = refined_annotations[barcode]
        n_updated <- 0
        for (i in seq_along(refined_barcodes)) {
            barcode <- refined_barcodes[i]
            idx <- which(all_barcodes == barcode)
            
            if (length(idx) == 1) {
                # Update combined_annotation with refined prediction
                seurat_s2@meta.data$combined_annotation[idx] <- refinement_annotations[barcode]
                # Also store in depth-specific column
                seurat_s2@meta.data[[depth_col_name]][idx] <- refinement_annotations[barcode]
                n_updated <- n_updated + 1
            }
        }
        
        cat(sprintf("--- [Depth %d] Updated %d cell annotations ---\n", depth, n_updated))
        
        # Mark this depth as completed
        max_completed_depth <- depth
        
        # Update state for next iteration
        current_cas_csv_path <- cas_csv_path_refinement
        seurat_to_check <- seurat_refinement_processed
        
    }  # End of refinement loop
    
    # === HANDLE EARLY STOPPING AND CONSENSUS MODE DETERMINATION ===
    # Matches Python logic for determining consensus mode
    cat("\n" %+% paste(rep("-", 60), collapse="") %+% "\n")
    cat("REFINEMENT SUMMARY\n")
    cat(paste(rep("-", 60), collapse="") %+% "\n")
    cat(sprintf("   Requested depth:    %d\n", args$refinement_depth))
    cat(sprintf("   Completed depth:    %d\n", max_completed_depth))
    if (!is.null(early_stop_reason)) {
        cat(sprintf("   Early stop reason:  %s\n", early_stop_reason))
    }
    
    # Determine consensus mode based on completed refinements
    # Matches Python: if max_completed_depth == 0: use biconsensus else use tri-consensus
    if (max_completed_depth == 0) {
        consensus_mode <- "biconsensus"
        cat("\n   -> Using BI-CONSENSUS filtering (individual vs consensus)\n")
        cat("      No refinement completed successfully.\n")
        # Reset combined_annotation to consensus (no refinement applied)
        seurat_s2$combined_annotation <- as.character(seurat_s2$ctpt_consensus_prediction)
    } else {
        consensus_mode <- "refinement_threeway"
        cat(sprintf("\n   -> Using TRI-CONSENSUS filtering (individual vs consensus vs refined)\n"))
        cat(sprintf("      Using refinement results from depth %d\n", max_completed_depth))
    }
    
    # Store consensus mode in metadata for downstream use
    # Matches Python: adata.uns['refinement_info'] = {...}
    seurat_s2@misc$refinement_info <- list(
        max_completed_depth = max_completed_depth,
        early_stop_reason = early_stop_reason,
        consensus_mode = consensus_mode,
        refinement_results = refinement_results,
        all_cas_paths = all_refinement_cas_paths
    )
    
    cat(paste(rep("-", 60), collapse="") %+% "\n")
    
    # Save intermediate results
    intermediate_rds_path <- file.path(stage2_output_dir, 
                                        sprintf("%s_after_refinement_depth_%d.rds", 
                                                args$final_run_prefix, max_completed_depth))
    saveRDS(seurat_s2, intermediate_rds_path)
    cat(sprintf("✅ Saved intermediate Seurat object: %s\n", basename(intermediate_rds_path)))
    
    # Save final objects
    final_rds_path <- file.path(stage2_output_dir, 
                                sprintf("%s_final_processed_with_refinement.rds", args$final_run_prefix))
    saveRDS(seurat_s2, final_rds_path)
    cat(sprintf("✅ Saved final Seurat object: %s\n", basename(final_rds_path)))
    
    # Store output paths in misc for reference
    seurat_s2@misc$refinement_info$output_paths <- list(
        main_stage1_dir = main_stage1_dir,
        stage2_output_dir = stage2_output_dir,
        intermediate_rds = intermediate_rds_path,
        final_rds = final_rds_path
    )
    
    return(seurat_s2)
}

# ==============================================================================
# --- EXPORT DECONVOLUTION FILES ---
# ==============================================================================

#' Export deconvolution-ready files (sc_counts.csv, sc_labels.csv, st_counts.csv)
#' Files are exported with gene intersection applied
#' @param args Pipeline arguments
#' @param seurat_sc Seurat object with scRNA-seq data (consistent cells)
#' @param final_label_col Column name containing final cell type labels
#' @param output_dir Output directory for deconvolution files
#' @return List with paths to exported files and gene counts
export_deconvolution_files <- function(args, seurat_sc, final_label_col, output_dir) {
    cat("\n" %+% paste(rep("-", 60), collapse="") %+% "\n")
    cat("EXPORTING DECONVOLUTION-READY FILES\n")
    cat(paste(rep("-", 60), collapse="") %+% "\n")
    
    # Check if ST data path is provided
    if (is.null(args$st_data_dir) || !file.exists(args$st_data_dir)) {
        cat("[INFO] No spatial data provided (--st_data_dir). Skipping deconvolution file export.\n")
        cat("       To export deconvolution files, provide --st_data_dir parameter.\n")
        return(NULL)
    }
    
    deconv_dir <- file.path(output_dir, "deconvolution_files")
    dir.create(deconv_dir, showWarnings = FALSE, recursive = TRUE)
    
    # === LOAD SPATIAL DATA ===
    cat("\n[1/5] Loading spatial transcriptomics data...\n")
    st_data <- tryCatch({
        load_expression_data(args$st_data_dir)
    }, error = function(e) {
        cat(sprintf("[ERROR] Failed to load ST data: %s\n", e$message))
        return(NULL)
    })
    
    if (is.null(st_data)) {
        return(NULL)
    }
    
    # Extract ST counts matrix
    if (inherits(st_data, "Seurat")) {
        st_counts_matrix <- GetAssayData(st_data, assay = "RNA", layer = "counts")
        st_genes <- rownames(st_counts_matrix)
        st_spots <- colnames(st_counts_matrix)
    } else {
        st_counts_matrix <- st_data
        st_genes <- rownames(st_counts_matrix)
        st_spots <- colnames(st_counts_matrix)
    }
    
    cat(sprintf("   ST data: %d genes x %d spots\n", length(st_genes), length(st_spots)))
    
    # === EXTRACT SC DATA ===
    cat("\n[2/5] Extracting scRNA-seq counts...\n")
    sc_counts_matrix <- GetAssayData(seurat_sc, assay = "RNA", layer = "counts")
    sc_genes <- rownames(sc_counts_matrix)
    sc_cells <- colnames(sc_counts_matrix)
    
    cat(sprintf("   SC data: %d genes x %d cells\n", length(sc_genes), length(sc_cells)))
    
    # === COMPUTE GENE INTERSECTION ===
    cat("\n[3/5] Computing gene intersection...\n")
    common_genes <- sort(intersect(sc_genes, st_genes))
    n_common <- length(common_genes)
    
    if (n_common == 0) {
        cat("[ERROR] No common genes found between SC and ST data!\n")
        cat("        SC gene examples: ", paste(head(sc_genes, 5), collapse=", "), "\n")
        cat("        ST gene examples: ", paste(head(st_genes, 5), collapse=", "), "\n")
        return(NULL)
    }
    
    cat(sprintf("   Common genes: %d\n", n_common))
    cat(sprintf("   SC-only genes: %d (excluded)\n", length(sc_genes) - n_common))
    cat(sprintf("   ST-only genes: %d (excluded)\n", length(st_genes) - n_common))
    
    # === SUBSET TO COMMON GENES ===
    cat("\n[4/5] Subsetting matrices to common genes...\n")
    
    # Subset SC counts
    sc_counts_intersected <- sc_counts_matrix[common_genes, , drop = FALSE]
    
    # Subset ST counts
    st_counts_intersected <- st_counts_matrix[common_genes, , drop = FALSE]
    
    # Verify dimensions match
    stopifnot(nrow(sc_counts_intersected) == n_common)
    stopifnot(nrow(st_counts_intersected) == n_common)
    stopifnot(identical(rownames(sc_counts_intersected), rownames(st_counts_intersected)))
    
    cat(sprintf("   SC counts (intersected): %d genes x %d cells\n", 
                nrow(sc_counts_intersected), ncol(sc_counts_intersected)))
    cat(sprintf("   ST counts (intersected): %d genes x %d spots\n", 
                nrow(st_counts_intersected), ncol(st_counts_intersected)))
    
    # === PREPARE LABELS ===
    sc_labels <- data.frame(
        cell_barcode = sc_cells,
        cell_type = as.character(seurat_sc@meta.data[[final_label_col]]),
        stringsAsFactors = FALSE
    )
    
    # Ensure sc_labels is ordered to match sc_cells exactly
    sc_labels <- sc_labels[match(sc_cells, sc_labels$cell_barcode), ]
    
    # Validate order consistency
    if (!identical(sc_cells, sc_labels$cell_barcode)) {
        stop("ERROR: Cell barcode order mismatch between counts and labels!")
    }
    
    # Verify no NA labels
    n_na_labels <- sum(is.na(sc_labels$cell_type))
    if (n_na_labels > 0) {
        cat(sprintf("[WARNING] %d cells have NA labels. These will be labeled as 'Unknown'.\n", n_na_labels))
        sc_labels$cell_type[is.na(sc_labels$cell_type)] <- "Unknown"
    }
    
    # === EXPORT FILES ===
    cat("\n[5/5] Writing deconvolution files (pandas-compatible format)...\n")
    
    # File paths
    sc_counts_path <- file.path(deconv_dir, "sc_counts.csv")
    sc_labels_path <- file.path(deconv_dir, "sc_labels.csv")
    st_counts_path <- file.path(deconv_dir, "st_counts.csv")
    gene_list_path <- file.path(deconv_dir, "common_genes.txt")
    
    # --- Write sc_counts.csv (cells x genes) ---
    # Python pandas format: 
    #   Header: ,Gene1,Gene2,Gene3,...
    #   Data:   barcode,val1,val2,val3,...
    cat("   Writing sc_counts.csv (cells x genes)...\n")
    
    # Transpose: genes x cells -> cells x genes
    sc_counts_transposed <- t(as.matrix(sc_counts_intersected))
    
    # Write header line manually (empty first cell + gene names)
    header_line <- paste0(",", paste(common_genes, collapse = ","))
    writeLines(header_line, sc_counts_path)
    
    # Write data rows (barcode + values)
    sc_data_lines <- sapply(seq_len(nrow(sc_counts_transposed)), function(i) {
        paste0(sc_cells[i], ",", paste(sc_counts_transposed[i, ], collapse = ","))
    })
    write(sc_data_lines, sc_counts_path, append = TRUE)
    
    # --- Write sc_labels.csv (cells x CellType) ---
    # Python pandas format:
    #   Header: ,CellType
    #   Data:   barcode,cell_type_name
    cat("   Writing sc_labels.csv (cells x CellType)...\n")
    
    # Write header line
    writeLines(",CellType", sc_labels_path)
    
    # Write data rows (barcode + cell type)
    label_data_lines <- paste0(sc_cells, ",", sc_labels$cell_type)
    write(label_data_lines, sc_labels_path, append = TRUE)
    
    # --- Write st_counts.csv (spots x genes) ---
    # Python pandas format:
    #   Header: ,Gene1,Gene2,Gene3,...
    #   Data:   spot_barcode,val1,val2,val3,...
    cat("   Writing st_counts.csv (spots x genes)...\n")
    
    # Transpose: genes x spots -> spots x genes
    st_counts_transposed <- t(as.matrix(st_counts_intersected))
    
    # Write header line manually (empty first cell + gene names)
    writeLines(header_line, st_counts_path)  # Same header as sc_counts (same genes)
    
    # Write data rows (barcode + values)
    st_data_lines <- sapply(seq_len(nrow(st_counts_transposed)), function(i) {
        paste0(st_spots[i], ",", paste(st_counts_transposed[i, ], collapse = ","))
    })
    write(st_data_lines, st_counts_path, append = TRUE)
    
    # --- Write gene list for reference ---
    cat("   Writing common_genes.txt...\n")
    writeLines(common_genes, gene_list_path)
    
    # === SUMMARY STATISTICS ===
    summary_path <- file.path(deconv_dir, "deconvolution_summary.txt")
    summary_lines <- c(
        "DECONVOLUTION FILES SUMMARY",
        paste(rep("=", 50), collapse=""),
        "",
        "Gene Intersection:",
        sprintf("  Total SC genes: %d", length(sc_genes)),
        sprintf("  Total ST genes: %d", length(st_genes)),
        sprintf("  Common genes (intersection): %d", n_common),
        "",
        "sc_counts.csv:",
        sprintf("  Dimensions: %d genes x %d cells", nrow(sc_counts_intersected), ncol(sc_counts_intersected)),
        sprintf("  File size: %.2f MB", file.info(sc_counts_path)$size / 1024^2),
        "",
        "sc_labels.csv:",
        sprintf("  Total cells: %d", nrow(sc_labels)),
        sprintf("  Cell types: %d", length(unique(sc_labels$cell_type))),
        "",
        "Cell type distribution:",
        paste(capture.output(print(table(sc_labels$cell_type))), collapse = "\n"),
        "",
        "st_counts.csv:",
        sprintf("  Dimensions: %d genes x %d spots", nrow(st_counts_intersected), ncol(st_counts_intersected)),
        sprintf("  File size: %.2f MB", file.info(st_counts_path)$size / 1024^2),
        "",
        sprintf("Generated: %s", Sys.time())
    )
    writeLines(summary_lines, summary_path)
    
    # === PRINT SUMMARY ===
    cat("\n" %+% paste(rep("-", 50), collapse="") %+% "\n")
    cat("DECONVOLUTION FILES EXPORTED\n")
    cat(paste(rep("-", 50), collapse="") %+% "\n")
    cat(sprintf("   Output directory: %s\n", deconv_dir))
    cat(sprintf("   ✓ sc_counts.csv: %d genes x %d cells\n", n_common, ncol(sc_counts_intersected)))
    cat(sprintf("   ✓ sc_labels.csv: %d cells, %d cell types\n", 
                nrow(sc_labels), length(unique(sc_labels$cell_type))))
    cat(sprintf("   ✓ st_counts.csv: %d genes x %d spots\n", n_common, ncol(st_counts_intersected)))
    cat(sprintf("   ✓ common_genes.txt: %d genes\n", n_common))
    cat(paste(rep("-", 50), collapse="") %+% "\n")
    
    return(list(
        sc_counts_path = sc_counts_path,
        sc_labels_path = sc_labels_path,
        st_counts_path = st_counts_path,
        gene_list_path = gene_list_path,
        n_common_genes = n_common,
        n_cells = ncol(sc_counts_intersected),
        n_spots = ncol(st_counts_intersected),
        n_cell_types = length(unique(sc_labels$cell_type)),
        common_genes = common_genes
    ))
}

# ==============================================================================
# --- EXPORT CONSISTENT CELLS ---
# ==============================================================================

#' Export consistent cells for downstream analysis
#' @description Implements exact Python logic for consensus filtering:
#'   - Bi-consensus: individual == consensus
#'   - Tri-consensus: individual == consensus == combined (after refinement)
#' Automatically determines consensus mode based on refinement results
#' @param args Pipeline arguments
#' @param seurat_obj Seurat object with annotations
#' @param consensus_mode Mode for consistency check: "auto", "biconsensus", or "refinement_threeway"
#' @return List with consistent cell data and export paths
export_consistent_cells <- function(args, seurat_obj, consensus_mode = "auto") {
    cat("\n" %+% paste(rep("=", 80), collapse="") %+% "\n")
    cat("### EXPORTING CONSISTENT CELL POPULATIONS ###\n")
    cat(paste(rep("=", 80), collapse="") %+% "\n")
    
    out_dir <- file.path(args$output_dir, "consistent_cells_subset")
    dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    
    n_cells <- ncol(seurat_obj)
    
    # === AUTO-DETECT CONSENSUS MODE FROM REFINEMENT INFO ===
    # Matches Python: Automatically determine based on refinement completion
    if (consensus_mode == "auto") {
        # Check if refinement info is stored in misc
        if (!is.null(seurat_obj@misc$refinement_info)) {
            consensus_mode <- seurat_obj@misc$refinement_info$consensus_mode
            max_depth <- seurat_obj@misc$refinement_info$max_completed_depth
            cat(sprintf("\n   Auto-detected consensus mode: %s (max refinement depth: %d)\n", 
                        consensus_mode, max_depth))
        } else {
            # Fallback detection based on available columns
            has_combined <- "combined_annotation" %in% colnames(seurat_obj@meta.data)
            has_biconsensus <- all(c("ctpt_individual_prediction", "ctpt_consensus_prediction") %in% 
                                      colnames(seurat_obj@meta.data))
            
            # Check if combined_annotation differs from consensus (indicates refinement)
            has_refinement <- FALSE
            if (has_combined && has_biconsensus) {
                n_different <- sum(seurat_obj$combined_annotation != seurat_obj$ctpt_consensus_prediction, 
                                   na.rm = TRUE)
                has_refinement <- (n_different > 0)
            }
            
            if (has_refinement) {
                consensus_mode <- "refinement_threeway"
                cat("\n   Auto-detected: TRI-CONSENSUS (refinement detected)\n")
            } else if (has_biconsensus) {
                consensus_mode <- "biconsensus"
                cat("\n   Auto-detected: BI-CONSENSUS (no refinement)\n")
            } else {
                cat("[ERROR] No annotation columns found for consistency filtering.\n")
                return(NULL)
            }
        }
    }
    
    cat(sprintf("\n   Consensus Mode: %s\n", toupper(consensus_mode)))
    
    # === CALCULATE AGREEMENT MASK BASED ON MODE ===
    # This section matches the Python logic exactly
    
    if (consensus_mode == "refinement_threeway") {
        # Check required columns for tri-consensus
        required_cols <- c("ctpt_individual_prediction", "ctpt_consensus_prediction", "combined_annotation")
        missing_cols <- setdiff(required_cols, colnames(seurat_obj@meta.data))
        if (length(missing_cols) > 0) {
            cat(sprintf("[WARNING] Missing columns for tri-consensus: %s\n", 
                        paste(missing_cols, collapse = ", ")))
            cat("   Falling back to bi-consensus mode.\n")
            consensus_mode <- "biconsensus"
        }
    }
    
    if (consensus_mode == "refinement_threeway") {
        # === TRI-CONSENSUS FILTERING ===
        # Matches Python logic exactly:
        #   mask = (ind_labels == cons_labels) & (cons_labels == comb_labels)
        #
        # For cells that were NOT refined: combined_annotation == ctpt_consensus_prediction
        # For cells that WERE refined: combined_annotation == refined prediction
        #
        # Tri-consensus requires ALL THREE to match:
        #   1. ctpt_individual_prediction (per-cell transfer learning)
        #   2. ctpt_consensus_prediction (cluster majority vote)
        #   3. combined_annotation (refined prediction, or consensus if not refined)
        
        ind_labels <- as.character(seurat_obj$ctpt_individual_prediction)
        cons_labels <- as.character(seurat_obj$ctpt_consensus_prediction)
        comb_labels <- as.character(seurat_obj$combined_annotation)
        
        # Three-way agreement check
        # Matches Python: mask = (ind == cons) & (cons == comb)
        mask_ind_cons <- (ind_labels == cons_labels)
        mask_cons_comb <- (cons_labels == comb_labels)
        mask <- mask_ind_cons & mask_cons_comb
        mask[is.na(mask)] <- FALSE
        
        final_label_col <- "combined_annotation"
        mode_label <- "Refinement_ThreeWay"
        
        # Report breakdown for debugging/verification
        n_ind_cons_agree <- sum(mask_ind_cons, na.rm = TRUE)
        n_cons_comb_agree <- sum(mask_cons_comb, na.rm = TRUE)
        n_all_agree <- sum(mask, na.rm = TRUE)
        
        cat(sprintf("\n   Tri-consensus breakdown:\n"))
        cat(sprintf("      Individual == Consensus: %d cells (%.1f%%)\n", 
                    n_ind_cons_agree, 100*n_ind_cons_agree/n_cells))
        cat(sprintf("      Consensus == Combined:   %d cells (%.1f%%)\n", 
                    n_cons_comb_agree, 100*n_cons_comb_agree/n_cells))
        cat(sprintf("      All three agree:         %d cells (%.1f%%)\n", 
                    n_all_agree, 100*n_all_agree/n_cells))
        
        # Show example of disagreements for debugging
        n_disagree <- sum(!mask, na.rm = TRUE)
        if (n_disagree > 0 && n_disagree < n_cells) {
            cat(sprintf("\n   Sample disagreements (first 5):\n"))
            disagree_idx <- which(!mask)[1:min(5, sum(!mask))]
            for (idx in disagree_idx) {
                barcode <- rownames(seurat_obj@meta.data)[idx]
                cat(sprintf("      %s: ind='%s', cons='%s', comb='%s'\n",
                            substr(barcode, 1, 20),
                            ind_labels[idx], cons_labels[idx], comb_labels[idx]))
            }
        }
        
    } else if (consensus_mode == "biconsensus") {
        # === BI-CONSENSUS FILTERING ===
        # Matches Python logic exactly:
        #   mask = (ind_labels == cons_labels)
        #
        # Bi-consensus requires TWO to match:
        #   1. ctpt_individual_prediction (per-cell transfer learning)
        #   2. ctpt_consensus_prediction (cluster majority vote)
        
        required_cols <- c("ctpt_individual_prediction", "ctpt_consensus_prediction")
        missing_cols <- setdiff(required_cols, colnames(seurat_obj@meta.data))
        if (length(missing_cols) > 0) {
            stop(sprintf("Missing required columns for bi-consensus: %s", 
                         paste(missing_cols, collapse = ", ")))
        }
        
        ind_labels <- as.character(seurat_obj$ctpt_individual_prediction)
        cons_labels <- as.character(seurat_obj$ctpt_consensus_prediction)
        
        # Two-way agreement check
        # Matches Python: mask = (ind == cons)
        mask <- (ind_labels == cons_labels)
        mask[is.na(mask)] <- FALSE
        
        final_label_col <- "ctpt_consensus_prediction"
        mode_label <- "BiConsensus"
        
        n_agree <- sum(mask, na.rm = TRUE)
        cat(sprintf("\n   Bi-consensus result:\n"))
        cat(sprintf("      Individual == Consensus: %d cells (%.1f%%)\n", 
                    n_agree, 100*n_agree/n_cells))
        
    } else {
        stop(sprintf("Unknown consensus_mode: %s", consensus_mode))
    }
    
    # === ADD CONSISTENCY STATUS TO SEURAT OBJECT ===
    # Matches Python: adata.obs['consistency_status'] = ...
    seurat_obj$consistency_status <- ifelse(mask, "Consistent", "Inconsistent")
    seurat_obj$consistency_label <- ifelse(
        mask, 
        as.character(seurat_obj@meta.data[[final_label_col]]), 
        "Inconsistent"
    )
    
    # === SUBSET TO CONSISTENT CELLS ===
    consistent_barcodes <- colnames(seurat_obj)[mask]
    inconsistent_barcodes <- colnames(seurat_obj)[!mask]
    n_consistent <- length(consistent_barcodes)
    n_inconsistent <- length(inconsistent_barcodes)
    
    if (n_consistent == 0) {
        cat("\n[WARNING] No consistent cells found!\n")
        
        # Save diagnostic info
        diag_path <- file.path(out_dir, sprintf("%s_consistency_diagnostics.txt", args$final_run_prefix))
        diag_lines <- c(
            "CONSISTENCY FILTERING DIAGNOSTICS",
            sprintf("Mode: %s", consensus_mode),
            sprintf("Total cells: %d", n_cells),
            sprintf("Consistent cells: 0"),
            "",
            "Column availability:",
            sprintf("  ctpt_individual_prediction: %s", 
                    "ctpt_individual_prediction" %in% colnames(seurat_obj@meta.data)),
            sprintf("  ctpt_consensus_prediction: %s", 
                    "ctpt_consensus_prediction" %in% colnames(seurat_obj@meta.data)),
            sprintf("  combined_annotation: %s", 
                    "combined_annotation" %in% colnames(seurat_obj@meta.data)),
            "",
            "Sample annotations (first 10 cells):"
        )
        
        # Add sample data
        sample_idx <- 1:min(10, n_cells)
        for (idx in sample_idx) {
            diag_lines <- c(diag_lines, sprintf(
                "  Cell %d: ind='%s', cons='%s', comb='%s'",
                idx,
                seurat_obj$ctpt_individual_prediction[idx],
                seurat_obj$ctpt_consensus_prediction[idx],
                if ("combined_annotation" %in% colnames(seurat_obj@meta.data)) 
                    seurat_obj$combined_annotation[idx] else "N/A"
            ))
        }
        
        writeLines(diag_lines, diag_path)
        cat(sprintf("   Saved diagnostics to: %s\n", diag_path))
        
        return(NULL)
    }
    
    seurat_consistent <- subset(seurat_obj, cells = consistent_barcodes)
    
    cat(sprintf("\n   ✓ Consistent cells: %d / %d (%.2f%%)\n", 
                n_consistent, n_cells, 100*n_consistent/n_cells))
    cat(sprintf("   ✓ Inconsistent cells: %d / %d (%.2f%%)\n", 
                n_inconsistent, n_cells, 100*n_inconsistent/n_cells))
    
    # === MIN CELLS PER TYPE FILTERING ===
    # Matches Python: filter cell types with < min_cells_per_type
    seurat_consistent_filtered <- seurat_consistent
    types_removed <- character(0)
    
    if (!is.null(args$min_cells_per_type) && args$min_cells_per_type > 0) {
        cat(sprintf("\n--- Applying min_cells_per_type filter: %d ---\n", args$min_cells_per_type))
        
        cell_type_counts <- table(seurat_consistent@meta.data[[final_label_col]])
        types_to_keep <- names(cell_type_counts[cell_type_counts >= args$min_cells_per_type])
        types_removed <- names(cell_type_counts[cell_type_counts < args$min_cells_per_type])
        
        if (length(types_removed) > 0) {
            cat(sprintf("   Removing %d cell types with < %d cells:\n", 
                        length(types_removed), args$min_cells_per_type))
            for (t in types_removed) {
                cat(sprintf("      - %s (%d cells)\n", t, cell_type_counts[t]))
            }
        }
        
        cells_to_keep <- colnames(seurat_consistent)[
            seurat_consistent@meta.data[[final_label_col]] %in% types_to_keep
        ]
        seurat_consistent_filtered <- subset(seurat_consistent, cells = cells_to_keep)
        
        cat(sprintf("\n   ✓ After filtering: %d cells, %d cell types\n", 
                    ncol(seurat_consistent_filtered), length(types_to_keep)))
    }
    
    # === SAVE OUTPUT FILES ===
    cat("\n--- Saving output files ---\n")
    
    # Barcodes list
    barcodes_path <- file.path(out_dir, sprintf("%s_%s_consistent_barcodes.txt", 
                                                 args$final_run_prefix, mode_label))
    writeLines(colnames(seurat_consistent_filtered), barcodes_path)
    cat(sprintf("   ✓ Barcodes: %s\n", basename(barcodes_path)))
    
    # Annotations CSV
    annotations_path <- file.path(out_dir, sprintf("%s_%s_annotations.csv", 
                                                    args$final_run_prefix, mode_label))
    cols_to_save <- intersect(
        c("seurat_clusters", "leiden", final_label_col, 
          "ctpt_individual_prediction", "ctpt_consensus_prediction", "combined_annotation",
          "ctpt_confidence", "prediction.score.max", "consistency_status"),
        colnames(seurat_consistent_filtered@meta.data)
    )
    write.csv(seurat_consistent_filtered@meta.data[, cols_to_save, drop=FALSE], 
              annotations_path, row.names = TRUE)
    cat(sprintf("   ✓ Annotations: %s\n", basename(annotations_path)))
    
    # Cell type summary
    summary_path <- file.path(out_dir, sprintf("%s_%s_celltype_summary.csv", 
                                                args$final_run_prefix, mode_label))
    summary_df <- tryCatch({
        ct_table <- as.data.frame(table(seurat_consistent_filtered@meta.data[[final_label_col]]), 
                                stringsAsFactors = FALSE)
        colnames(ct_table) <- c("CellType", "Count")
        ct_table$Percentage <- 100 * ct_table$Count / sum(ct_table$Count)
        ct_table <- ct_table[order(-ct_table$Count), ]
        ct_table
    }, error = function(e) {
        cat(sprintf("   [WARNING] Cell type summary failed: %s\n", e$message))
        data.frame(CellType = character(0), Count = integer(0), Percentage = numeric(0))
    })
    write.csv(summary_df, summary_path, row.names = FALSE)
    cat(sprintf("   ✓ Cell type summary: %s\n", basename(summary_path)))
    
    # UMAP visualization (consistent cells only)
    tryCatch({
        n_types <- length(unique(seurat_consistent_filtered@meta.data[[final_label_col]]))
        type_colors <- get_plot_colors(n_types)
        
        p_consistent <- DimPlot(seurat_consistent_filtered, reduction = "umap", 
                                group.by = final_label_col, pt.size = 1, label = FALSE,
                                cols = type_colors) +
            ggtitle(sprintf("Consistent Cells (%s)\nn=%d (%.1f%% of total)", 
                            mode_label, ncol(seurat_consistent_filtered),
                            100 * ncol(seurat_consistent_filtered) / n_cells)) +
            labs(color = "Cell Type") +
            theme_umap_clean()
        
        umap_path <- file.path(out_dir, sprintf("%s_%s_umap.png", 
                                                 args$final_run_prefix, mode_label))
        ggsave(umap_path, plot = p_consistent, width = 12, height = 10, dpi = args$fig_dpi)
        cat(sprintf("   ✓ UMAP plot: %s\n", basename(umap_path)))
    }, error = function(e) {
        cat(sprintf("   [WARNING] UMAP plot failed: %s\n", e$message))
    })
    
    # === UMAP WITH INCONSISTENT CELLS IN GREY ===
    cat("\n--- Generating UMAP with inconsistent cells highlighted ---\n")
    
    tryCatch({
        # Get UMAP coordinates from full object
        umap_coords <- Embeddings(seurat_obj, reduction = "umap")
        
        # Create plotting dataframe
        plot_df <- data.frame(
            barcode = rownames(umap_coords),
            UMAP_1 = umap_coords[, 1],
            UMAP_2 = umap_coords[, 2],
            cell_type = as.character(seurat_obj@meta.data[[final_label_col]]),
            consistency_status = seurat_obj$consistency_status,
            consistency_label = seurat_obj$consistency_label,
            stringsAsFactors = FALSE
        )
        
        # Add individual and consensus predictions for CSV
        plot_df$ctpt_individual_prediction <- as.character(seurat_obj$ctpt_individual_prediction)
        plot_df$ctpt_consensus_prediction <- as.character(seurat_obj$ctpt_consensus_prediction)
        if ("combined_annotation" %in% colnames(seurat_obj@meta.data)) {
            plot_df$combined_annotation <- as.character(seurat_obj$combined_annotation)
        }
        
        # Get unique cell types (excluding "Inconsistent")
        unique_cell_types <- sort(unique(plot_df$cell_type[plot_df$consistency_status == "Consistent"]))
        n_types <- length(unique_cell_types)
        
        # Use the global color palette
        type_colors <- get_plot_colors(n_types)
        names(type_colors) <- unique_cell_types
        
        # Add grey for inconsistent
        all_colors <- c(type_colors, "Inconsistent" = "grey70")
        
        # Create ordered factor for plotting (inconsistent first so they're behind)
        plot_df$plot_label <- factor(
            plot_df$consistency_label,
            levels = c("Inconsistent", unique_cell_types)
        )
        
        # Sort dataframe so inconsistent cells are plotted first (background)
        plot_df <- plot_df[order(plot_df$consistency_status == "Consistent"), ]
        
        # Create the plot
        p_with_grey <- ggplot(plot_df, aes(x = UMAP_1, y = UMAP_2, color = plot_label)) +
            geom_point(size = 1, alpha = 0.7) +
            scale_color_manual(
                values = all_colors,
                name = "Cell Type",
                guide = guide_legend(override.aes = list(size = 3, alpha = 1))
            ) +
            labs(
                title = sprintf("UMAP: Consistent vs Inconsistent Cells (%s)", mode_label),
                subtitle = sprintf("Consistent: %d (%.1f%%) | Inconsistent (grey): %d (%.1f%%)",
                                   n_consistent, 100 * n_consistent / n_cells,
                                   n_inconsistent, 100 * n_inconsistent / n_cells),
                x = "UMAP 1",
                y = "UMAP 2"
            ) +
            theme_umap_clean()
        
        umap_grey_path <- file.path(out_dir, sprintf("%s_%s_umap_with_inconsistent.png", 
                                                      args$final_run_prefix, mode_label))
        ggsave(umap_grey_path, plot = p_with_grey, width = 12, height = 10, dpi = args$fig_dpi)
        cat(sprintf("   ✓ UMAP with inconsistent (grey): %s\n", basename(umap_grey_path)))
        
        # === SAVE CSV WITH ALL CELLS AND CONSISTENCY STATUS ===
        csv_all_cells_path <- file.path(out_dir, sprintf("%s_%s_all_cells_consistency.csv", 
                                                          args$final_run_prefix, mode_label))
        
        csv_output_df <- plot_df %>%
            dplyr::select(
                barcode,
                UMAP_1,
                UMAP_2,
                cell_type,
                consistency_status,
                ctpt_individual_prediction,
                ctpt_consensus_prediction,
                dplyr::any_of("combined_annotation")
            ) %>%
            dplyr::arrange(consistency_status, cell_type)
        
        write.csv(csv_output_df, csv_all_cells_path, row.names = FALSE)
        cat(sprintf("   ✓ All cells consistency CSV: %s\n", basename(csv_all_cells_path)))
        
        # === SUMMARY TABLE: CONSISTENCY BY CELL TYPE ===
        consistency_summary <- plot_df %>%
            dplyr::group_by(cell_type) %>%
            dplyr::summarise(
                Total_Cells = n(),
                Consistent_Cells = sum(consistency_status == "Consistent"),
                Inconsistent_Cells = sum(consistency_status == "Inconsistent"),
                Consistency_Rate_Pct = 100 * Consistent_Cells / Total_Cells,
                .groups = "drop"
            ) %>%
            dplyr::arrange(desc(Total_Cells))
        
        consistency_summary_path <- file.path(out_dir, sprintf("%s_%s_consistency_by_celltype.csv", 
                                                                args$final_run_prefix, mode_label))
        write.csv(consistency_summary, consistency_summary_path, row.names = FALSE)
        cat(sprintf("   ✓ Consistency by cell type: %s\n", basename(consistency_summary_path)))
        
        # Print summary table
        cat("\n   Consistency by Cell Type:\n")
        cat("   " %+% paste(rep("-", 70), collapse="") %+% "\n")
        cat(sprintf("   %-30s %8s %10s %12s %8s\n", 
                    "Cell Type", "Total", "Consistent", "Inconsistent", "Rate(%)"))
        cat("   " %+% paste(rep("-", 70), collapse="") %+% "\n")
        for (i in 1:min(10, nrow(consistency_summary))) {
            row <- consistency_summary[i, ]
            cat(sprintf("   %-30s %8d %10d %12d %8.1f\n",
                        substr(row$cell_type, 1, 30),
                        row$Total_Cells,
                        row$Consistent_Cells,
                        row$Inconsistent_Cells,
                        row$Consistency_Rate_Pct))
        }
        if (nrow(consistency_summary) > 10) {
            cat(sprintf("   ... and %d more cell types\n", nrow(consistency_summary) - 10))
        }
        cat("   " %+% paste(rep("-", 70), collapse="") %+% "\n")
        
    }, error = function(e) {
        cat(sprintf("   [WARNING] UMAP with inconsistent cells failed: %s\n", e$message))
        traceback()
    })
    
    # Save Seurat object (consistent cells only)
    rds_path <- file.path(out_dir, sprintf("%s_%s_seurat.rds", args$final_run_prefix, mode_label))
    saveRDS(seurat_consistent_filtered, rds_path)
    cat(sprintf("   ✓ Seurat RDS: %s\n", basename(rds_path)))
    
    # Save full Seurat with consistency metadata
    full_rds_path <- file.path(out_dir, sprintf("%s_%s_full_with_consistency.rds", 
                                                 args$final_run_prefix, mode_label))
    saveRDS(seurat_obj, full_rds_path)
    cat(sprintf("   ✓ Full Seurat with consistency: %s\n", basename(full_rds_path)))
    
    # === EXPORT DECONVOLUTION FILES ===
    deconv_result <- NULL
    if (!is.null(args$st_data_dir)) {
        deconv_result <- export_deconvolution_files(
            args = args,
            seurat_sc = seurat_consistent_filtered,
            final_label_col = final_label_col,
            output_dir = out_dir
        )
    }
    
    # === SUMMARY ===
    cat("\n" %+% paste(rep("-", 50), collapse="") %+% "\n")
    cat("CONSISTENT CELLS EXPORT SUMMARY\n")
    cat(paste(rep("-", 50), collapse="") %+% "\n")
    cat(sprintf("   Mode:              %s\n", mode_label))
    cat(sprintf("   Input cells:       %d\n", n_cells))
    cat(sprintf("   Consistent cells:  %d (%.2f%%)\n", n_consistent, 100*n_consistent/n_cells))
    cat(sprintf("   Inconsistent cells: %d (%.2f%%)\n", n_inconsistent, 100*n_inconsistent/n_cells))
    cat(sprintf("   After filtering:   %d\n", ncol(seurat_consistent_filtered)))
    cat(sprintf("   Cell types:        %d\n", length(unique(seurat_consistent_filtered@meta.data[[final_label_col]]))))
    if (length(types_removed) > 0) {
        cat(sprintf("   Types removed:     %d\n", length(types_removed)))
    }
    if (!is.null(deconv_result)) {
        cat(sprintf("   Deconvolution files: ✓ (%d genes intersected)\n", deconv_result$n_common_genes))
    }
    cat(paste(rep("-", 50), collapse="") %+% "\n")
    
    return(list(
        seurat_consistent = seurat_consistent,
        seurat_filtered = seurat_consistent_filtered,
        seurat_full_with_consistency = seurat_obj,
        mode = mode_label,
        consensus_mode = consensus_mode,
        n_consistent = n_consistent,
        n_inconsistent = n_inconsistent,
        n_filtered = ncol(seurat_consistent_filtered),
        final_label_col = final_label_col,
        types_removed = types_removed,
        consistency_mask = mask,
        deconvolution = deconv_result
    ))
}

# ==============================================================================
# --- MAIN FUNCTION ---
# ==============================================================================

#' Main pipeline function
main <- function(args) {
    # Initialize global environment
    GLOBAL_ENV$ARGS <- args
    GLOBAL_ENV$RANDOM_SEED <- args$seed
    GLOBAL_ENV$OPTIMIZATION_CACHE <- new.env(hash = TRUE)
    GLOBAL_ENV$TRIAL_METADATA <- list()
    GLOBAL_ENV$CURRENT_STRATEGY_NAME <- ""
    GLOBAL_ENV$CURRENT_OPTIMIZATION_TARGET <- "balanced"
    GLOBAL_ENV$seurat_base <- NULL
    GLOBAL_ENV$seurat_full_data <- NULL
    GLOBAL_ENV$seurat_ref <- NULL
    GLOBAL_ENV$REF_LABELS_COL <- NULL
    GLOBAL_ENV$SPECIES_DB <- NULL
    GLOBAL_ENV$MITO_PREFIX <- NULL
    GLOBAL_ENV$IS_BATCHED_DATA <- FALSE
    GLOBAL_ENV$BATCH_COLUMN <- NULL
    GLOBAL_ENV$MARKER_DB <- NULL
    GLOBAL_ENV$GENE_STANDARDIZATION_APPLIED <- FALSE  # NEW: Track if standardization was applied
    
    set.seed(GLOBAL_ENV$RANDOM_SEED)
    dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Species-specific setup (unchanged)
    cat(sprintf("\n--- Setting up for '%s' species ---\n", args$species))
    if (args$species == "human") {
        check_and_load_bioc_package("org.Hs.eg.db")
        GLOBAL_ENV$SPECIES_DB <- org.Hs.eg.db
        GLOBAL_ENV$MITO_PREFIX <- "MT-"
    } else if (args$species == "mouse") {
        check_and_load_bioc_package("org.Mm.eg.db")
        GLOBAL_ENV$SPECIES_DB <- org.Mm.eg.db
        GLOBAL_ENV$MITO_PREFIX <- "mt-"
    } else {
        stop("Invalid species specified. Use 'human' or 'mouse'.")
    }
    
    # === LOAD MARKER DATABASE (NEW) ===
    cat("\n--- Loading Marker Gene Database ---\n")
    
    marker_db_result <- load_marker_database(
        marker_path = args$marker_db_path,
        species = args$marker_prior_species %||% args$species,  # Use marker_prior_species if set
        organ = args$marker_prior_organ
    )
    
    if (!is.null(marker_db_result)) {
        GLOBAL_ENV$MARKER_DB <- marker_db_result
        cat(sprintf("✅ Marker database loaded: %d cell types\n", length(marker_db_result$markers)))
    } else {
        GLOBAL_ENV$MARKER_DB <- NULL
        cat("[INFO] No marker database. MPS will not be calculated.\n")
    }
    
    # Reference Object Loading (unchanged)
    cat("\n--- Loading and Preprocessing Reference Seurat Object ---\n")
    tryCatch({
        seurat_ref_obj <- readRDS(args$reference_path)
        seurat_ref_obj <- NormalizeData(seurat_ref_obj, verbose = FALSE)
        seurat_ref_obj <- FindVariableFeatures(seurat_ref_obj, method = "vst", nfeatures = 2000, verbose = FALSE)
        seurat_ref_obj <- ScaleData(seurat_ref_obj, verbose = FALSE)
        seurat_ref_obj <- RunPCA(seurat_ref_obj, npcs = 105, verbose = FALSE)
        GLOBAL_ENV$seurat_ref <- seurat_ref_obj
        GLOBAL_ENV$REF_LABELS_COL <- args$reference_labels_col
        
        if (!(GLOBAL_ENV$REF_LABELS_COL %in% colnames(GLOBAL_ENV$seurat_ref@meta.data))) {
            stop(sprintf("Reference label column '%s' not found.", GLOBAL_ENV$REF_LABELS_COL))
        }
        cat(sprintf("✅ Reference object loaded from '%s'\n", args$reference_path))
        # === VERIFY REFERENCE GENE NAME FORMAT ===
        ref_genes <- rownames(seurat_ref_obj)
        ref_sample <- head(ref_genes, 10)
        
        # Detect reference gene format
        n_upper <- sum(grepl("^[A-Z0-9-]+$", ref_sample))
        n_title <- sum(grepl("^[A-Z][a-z0-9-]+", ref_sample))
        
        if (args$species == "human" && n_upper < length(ref_sample) * 0.5) {
            cat("[WARNING] Reference genes may not be in UPPERCASE format for human.\n")
            cat(sprintf("         Sample genes: %s\n", paste(ref_sample, collapse = ", ")))
        } else if (args$species == "mouse" && n_title < length(ref_sample) * 0.5) {
            cat("[WARNING] Reference genes may not be in Title Case format for mouse.\n")
            cat(sprintf("         Sample genes: %s\n", paste(ref_sample, collapse = ", ")))
        }
        
        cat(sprintf("   Reference genes (sample): %s\n", paste(head(ref_sample, 5), collapse = ", ")))
    }, error = function(e) { 
        stop(sprintf("Failed to load reference object. Error: %s", e$message)) 
    })
    
    # === STAGE 1: BAYESIAN OPTIMIZATION ===
    cat("\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
    cat("### STAGE 1: BAYESIAN OPTIMIZATION ###\n")
    cat(paste(rep("=", 70), collapse="") %+% "\n")
    
    stage1_output_dir <- file.path(args$output_dir, "stage_1_bayesian_optimization")
    dir.create(stage1_output_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Load and preprocess data
    data <- load_expression_data(args$data_dir)
    if (inherits(data, "Seurat")) {
        seurat_obj <- data
    } else {
        seurat_obj <- CreateSeuratObject(counts = data, project = "scRNA", min.cells = MIN_CELLS_PER_GENE)
    }
    
    # Gene name mapping if needed (unchanged)
    if (any(grepl("^ENSG|^ENSMUS", rownames(seurat_obj)))) {
        cat("[INFO] Mapping ENSEMBL IDs to gene symbols...\n")
        ensembl_ids <- gsub("\\..*$", "", rownames(seurat_obj))
        gene_symbols <- mapIds(GLOBAL_ENV$SPECIES_DB, keys = ensembl_ids, 
                               column = "SYMBOL", keytype = "ENSEMBL", multiVals = "first")
        unmapped <- which(is.na(gene_symbols))
        gene_symbols[unmapped] <- rownames(seurat_obj)[unmapped]
        unique_symbols <- make.unique(as.character(gene_symbols))
        counts_data <- GetAssayData(seurat_obj, assay = "RNA", layer = "counts")
        rownames(counts_data) <- unique_symbols
        seurat_obj[["RNA"]] <- CreateAssayObject(counts = counts_data)
        DefaultAssay(seurat_obj) <- "RNA"
    }
    
    # === BATCH DETECTION (NEW) ===
    cat("\n--- Detecting Batch Information ---\n")
    
    # Detect batch info (will extract from barcodes if needed)
    batch_info <- detect_batch_info(
        seurat_obj, 
        batch_col = args$batch_col,
        barcode_separator = args$barcode_separator
    )
    
    # CRITICAL: Update seurat_obj if batch was extracted from barcodes
    if (!is.null(batch_info$seurat_obj)) {
        seurat_obj <- batch_info$seurat_obj
    }
    
    # Handle --skip_integration flag
    if (args$skip_integration && batch_info$is_batched) {
        cat("[INFO] --skip_integration flag set. Disabling Harmony integration.\n")
        batch_info$is_batched <- FALSE
    }
    
    GLOBAL_ENV$IS_BATCHED_DATA <- batch_info$is_batched
    GLOBAL_ENV$BATCH_COLUMN <- batch_info$batch_column
    
    if (batch_info$is_batched) {
        cat("\n" %+% paste(rep("-", 50), collapse="") %+% "\n")
        cat("BATCHED DATA DETECTED - Using Harmony Integration\n")
        cat(paste(rep("-", 50), collapse="") %+% "\n")
        cat(sprintf("   Batch column: %s\n", batch_info$batch_column))
        cat(sprintf("   Number of batches: %d\n", batch_info$n_batches))
        cat(sprintf("   Source: %s\n", batch_info$source))
        cat("   Integration: Harmony (per-iteration)\n")
        cat(paste(rep("-", 50), collapse="") %+% "\n\n")
    } else {
        cat("\n" %+% paste(rep("-", 50), collapse="") %+% "\n")
        cat("SINGLE-SAMPLE DATA - Standard Pipeline\n")
        cat(paste(rep("-", 50), collapse="") %+% "\n\n")
    }
    
    # ST data intersection if provided (unchanged)
    if (!is.null(args$st_data_dir) && file.exists(args$st_data_dir)) {
        cat(sprintf("\n[INFO] Loading Spatial Data for gene intersection: %s\n", args$st_data_dir))
        st_data <- load_expression_data(args$st_data_dir)
        st_genes <- if (inherits(st_data, "Seurat")) rownames(st_data) else rownames(st_data)
        common_genes <- sort(intersect(rownames(seurat_obj), st_genes))
        cat(sprintf("   -> Common genes: %d\n", length(common_genes)))
        seurat_obj <- subset(seurat_obj, features = common_genes)
    }
    
    # QC filtering (unchanged)
    seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = MITO_REGEX_PATTERN)
    seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > MIN_GENES_PER_CELL & 
                             nFeature_RNA < MAX_GENES_PER_CELL & 
                             percent.mt < MAX_PCT_COUNTS_MT)
    seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", 
                                scale.factor = 10000, verbose = FALSE)
    
    GLOBAL_ENV$seurat_full_data <- seurat_obj
    cat(sprintf("✅ QC complete. Dataset: %d genes x %d cells\n", nrow(seurat_obj), ncol(seurat_obj)))
    
    # This ensures gene names in query data match the standardized marker database
    # Human: UPPERCASE (CD4, PTPRC, MT-CO1)
    # Mouse: Title Case (Cd4, Ptprc, mt-Co1)
    
    if (!is.null(args$marker_db_path) && file.exists(args$marker_db_path)) {
        cat("\n--- Standardizing Query Gene Names for MPS Compatibility ---\n")
        
        original_genes <- rownames(seurat_obj)
        n_original <- length(original_genes)
        
        # Apply species-specific standardization
        standardized_genes <- standardize_gene_names(original_genes, species = args$species)
        
        # Show sample transformations for verification
        cat(sprintf("   Species: %s\n", args$species))
        cat(sprintf("   Format: %s\n", ifelse(args$species == "human", "UPPERCASE", "Title Case")))
        cat("   Sample transformations:\n")
        
        # Find genes that changed
        changed_idx <- which(original_genes != standardized_genes)
        if (length(changed_idx) > 0) {
            sample_changed <- head(changed_idx, 5)
            for (idx in sample_changed) {
                cat(sprintf("      '%s' -> '%s'\n", original_genes[idx], standardized_genes[idx]))
            }
            cat(sprintf("   Total genes changed: %d / %d (%.1f%%)\n", 
                        length(changed_idx), n_original, 100 * length(changed_idx) / n_original))
        } else {
            cat("      No changes needed (genes already in correct format)\n")
        }
        
        # Check for duplicates after standardization
        if (any(duplicated(standardized_genes))) {
            n_dups <- sum(duplicated(standardized_genes))
            cat(sprintf("   [WARNING] %d duplicate gene names after standardization. Making unique.\n", n_dups))
            
            # Make unique while preserving order
            standardized_genes <- make.unique(standardized_genes, sep = "_dup")
            
            # Report duplicates
            dup_genes <- standardized_genes[duplicated(standardize_gene_names(original_genes, args$species))]
            if (length(dup_genes) > 0) {
                cat(sprintf("      Duplicated genes (first 5): %s\n", 
                            paste(head(unique(dup_genes), 5), collapse = ", ")))
            }
        }
        
        # === REBUILD SEURAT OBJECT WITH STANDARDIZED GENE NAMES ===
        # Extract counts matrix
        counts_data <- GetAssayData(seurat_obj, assay = "RNA", layer = "counts")
        
        # Rename rows
        rownames(counts_data) <- standardized_genes
        
        # Store original metadata
        original_metadata <- seurat_obj@meta.data
        
        # Create new Seurat object with standardized gene names
        seurat_obj <- CreateSeuratObject(
            counts = counts_data, 
            project = "scRNA_standardized",
            meta.data = original_metadata,
            min.cells = 0,  # Don't filter again
            min.features = 0
        )
        
        # Re-normalize after gene name change
        seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize",
                                    scale.factor = 10000, verbose = FALSE)
        
        # Mark that standardization was applied
        GLOBAL_ENV$GENE_STANDARDIZATION_APPLIED <- TRUE
        
        # Verify marker gene overlap after standardization
        if (!is.null(GLOBAL_ENV$MARKER_DB)) {
            # FIX: Handle both list and vector formats from marker database
            all_marker_genes <- unique(unlist(lapply(GLOBAL_ENV$MARKER_DB$markers, function(x) {
                # Check if x is a list with $genes element or direct vector
                if (is.list(x) && !is.null(x$genes)) {
                    return(x$genes)
                } else if (is.character(x)) {
                    # Direct character vector format
                    return(x)
                } else {
                    return(character(0))
                }
            })))
            
            query_genes_standardized <- rownames(seurat_obj)
            marker_overlap <- intersect(all_marker_genes, query_genes_standardized)
            
            cat(sprintf("\n   Marker gene overlap after standardization:\n"))
            cat(sprintf("      Marker DB genes: %d\n", length(all_marker_genes)))
            cat(sprintf("      Query genes: %d\n", length(query_genes_standardized)))
            cat(sprintf("      Overlapping: %d (%.1f%% of markers)\n", 
                        length(marker_overlap), 
                        100 * length(marker_overlap) / max(1, length(all_marker_genes))))
            
            # Show sample overlapping markers
            if (length(marker_overlap) > 0) {
                cat(sprintf("      Sample overlapping markers: %s\n", 
                            paste(head(marker_overlap, 10), collapse = ", ")))
            }
            
            # Warning if low overlap
            if (length(marker_overlap) < length(all_marker_genes) * 0.1) {
                cat("   [WARNING] Less than 10% of marker genes found in query data!\n")
                cat("             Check if species setting is correct.\n")
            }
        }
        
        cat("   ✅ Query gene names standardized successfully\n\n")
        
    } else {
        GLOBAL_ENV$GENE_STANDARDIZATION_APPLIED <- FALSE
        cat("[INFO] No marker database specified. Skipping gene name standardization.\n")
    }

    # Report batch composition after QC (NEW)
    if (batch_info$is_batched) {
        batch_sizes_after_qc <- table(seurat_obj@meta.data[[batch_info$batch_column]])
        cat(sprintf("   -> Batch sizes after QC: %s\n", 
                    paste(sprintf("%s=%d", names(batch_sizes_after_qc), batch_sizes_after_qc), 
                          collapse = ", ")))
    }
    
    # === USE IDENTICAL DATA FOR STAGE 1 AND STAGE 2 ===
    GLOBAL_ENV$seurat_base <- seurat_obj
    GLOBAL_ENV$seurat_base <- ensure_standard_assay(GLOBAL_ENV$seurat_base)
    
    cat(sprintf("   -> Using full dataset (%d cells) for optimization and final analysis\n", 
                ncol(GLOBAL_ENV$seurat_base)))
    
    # Run Bayesian Optimization
    target <- if (args$target == 'all') 'balanced' else args$target
    GLOBAL_ENV$CURRENT_OPTIMIZATION_TARGET <- target
    
    strategies <- list(
        "Exploit" = list(acq = 'poi', kappa = 2.576, eps = 0.0), 
        "BO-EI"   = list(acq = 'ei',  kappa = 2.576, eps = 0.0), 
        "Explore" = list(acq = 'ei',  kappa = 2.576, eps = 0.1)
    )
    
    results <- list()
    for (name in names(strategies)) {
        cat(sprintf("\n--- Running Strategy: %s ---\n", name))
        GLOBAL_ENV$CURRENT_STRATEGY_NAME <- name
        GLOBAL_ENV$TRIAL_METADATA <- list()
        
        strategy_result <- tryCatch({
            opt_result <- safe_bayesian_optimization(
                FUN = objective_function, 
                bounds = SEARCH_SPACE_BOUNDS, 
                init_points = args$n_init_points, 
                n_iter = args$n_calls - args$n_init_points, 
                acq = strategies[[name]]$acq, 
                kappa = strategies[[name]]$kappa, 
                eps = strategies[[name]]$eps, 
                verbose = FALSE,
                max_total_failures = (args$n_calls) * 3,
                seed = GLOBAL_ENV$RANDOM_SEED
            )
            
            if (!is.null(opt_result)) {
                opt_result$trial_metadata <- GLOBAL_ENV$TRIAL_METADATA
            }
            opt_result
            
        }, error = function(e) {
            cat(sprintf("   [CRITICAL ERROR] Strategy '%s' failed: %s\n", name, e$message))
            tryCatch({
                run_random_search_fallback(objective_function, SEARCH_SPACE_BOUNDS, 10, 
                                           seed = GLOBAL_ENV$RANDOM_SEED)
            }, error = function(e2) {
                NULL
            })
        })
        
        if (!is.null(strategy_result) && 
            !is.null(strategy_result$Best_Value) && 
            is.finite(strategy_result$Best_Value)) {
            results[[name]] <- strategy_result
            cat(sprintf("   ✓ Strategy '%s' completed. Best: %.4f\n", 
                        name, strategy_result$Best_Value))
        } else {
            cat(sprintf("   ✗ Strategy '%s' produced no valid results.\n", name))
        }
    }
    
    if (length(results) == 0) {
        stop("All optimization strategies failed. Check input data and parameters.")
    }
    
    best_overall_score <- -Inf
    best_params <- NULL
    winning_strategy <- NULL
    for (name in names(results)) {
        if (results[[name]]$Best_Value > best_overall_score) {
            best_overall_score <- results[[name]]$Best_Value
            best_params <- as.list(results[[name]]$Best_Par)
            winning_strategy <- name
        }
    }
    
    cat(sprintf("\n✅ Best Strategy: %s with score %.4f\n", winning_strategy, best_overall_score))
    
    generate_yield_csv(results, target, stage1_output_dir, args$output_prefix)
    plot_optimizer_convergence(results, target, stage1_output_dir, args$output_prefix)
    
    # ==========================================================================
    # === STAGE 1.5: EXPORT BEST PARAMETERS SUMMARY ===
    # ==========================================================================
    
    cat("\n--- Exporting Best Parameters Summary ---\n")
    
    best_params_summary_path <- file.path(stage1_output_dir, 
                                           paste0(args$output_prefix, "_best_parameters_summary.txt"))
    
    # Collect all metric information from the winning strategy
    winning_result <- results[[winning_strategy]]
    
    # Extract detailed metrics if available from trial metadata
    best_trial_metrics <- NULL
    if (!is.null(winning_result$trial_metadata) && length(winning_result$trial_metadata) > 0) {
        # Find the trial with the best score
        trial_scores <- sapply(winning_result$trial_metadata, function(t) {
            if (!is.null(t$score)) t$score else -Inf
        })
        best_trial_idx <- which.max(trial_scores)
        if (length(best_trial_idx) > 0) {
            best_trial_metrics <- winning_result$trial_metadata[[best_trial_idx]]
        }
    }
    
    # Build summary lines
    summary_lines <- c(
        paste(rep("=", 80), collapse = ""),
        "BAYESIAN OPTIMIZATION - BEST PARAMETERS SUMMARY",
        paste(rep("=", 80), collapse = ""),
        "",
        sprintf("Generated: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
        sprintf("Output Directory: %s", stage1_output_dir),
        "",
        paste(rep("-", 80), collapse = ""),
        "WINNING STRATEGY",
        paste(rep("-", 80), collapse = ""),
        sprintf("Strategy Name: %s", winning_strategy),
        sprintf("Best Composite Score: %.6f", best_overall_score),
        "",
        paste(rep("-", 80), collapse = ""),
        "OPTIMAL PARAMETERS",
        paste(rep("-", 80), collapse = ""),
        sprintf("n_hvg (Number of Highly Variable Genes): %d", as.integer(best_params$n_hvg)),
        sprintf("n_pcs (Number of Principal Components): %d", as.integer(best_params$n_pcs)),
        sprintf("n_neighbors (UMAP/Clustering Neighbors): %d", as.integer(best_params$n_neighbors)),
        sprintf("resolution (Clustering Resolution): %.4f", best_params$resolution),
        ""
    )
    
    # Add detailed metrics if available
    if (!is.null(best_trial_metrics)) {
        summary_lines <- c(summary_lines,
            paste(rep("-", 80), collapse = ""),
            "DETAILED METRICS FROM BEST TRIAL",
            paste(rep("-", 80), collapse = ""),
            ""
        )
        
        # CAS (Cell Annotation Score) metrics
        if (!is.null(best_trial_metrics$cas_mean)) {
            summary_lines <- c(summary_lines,
                ">>> CAS (Cell Annotation Score) <<<",
                sprintf("  CAS Mean: %.4f", best_trial_metrics$cas_mean),
                sprintf("  CAS Median: %.4f", best_trial_metrics$cas_median %||% NA),
                sprintf("  CAS Std Dev: %.4f", best_trial_metrics$cas_std %||% NA),
                sprintf("  CAS Min: %.4f", best_trial_metrics$cas_min %||% NA),
                sprintf("  CAS Max: %.4f", best_trial_metrics$cas_max %||% NA),
                ""
            )
        }
        
        # MCS (Marker Confidence Score) metrics
        if (!is.null(best_trial_metrics$mcs)) {
            summary_lines <- c(summary_lines,
                ">>> MCS (Marker Confidence Score) <<<",
                sprintf("  MCS: %.4f", best_trial_metrics$mcs),
                ""
            )
        }
        
        # Silhouette Score
        if (!is.null(best_trial_metrics$silhouette)) {
            summary_lines <- c(summary_lines,
                ">>> Silhouette Score <<<",
                sprintf("  Silhouette: %.4f", best_trial_metrics$silhouette),
                ""
            )
        }
        
        # MPS (Marker Prior Score) if available
        if (!is.null(best_trial_metrics$mps)) {
            summary_lines <- c(summary_lines,
                ">>> MPS (Marker Prior Score) <<<",
                sprintf("  MPS: %.4f", best_trial_metrics$mps),
                ""
            )
        }
        
        # Clustering info
        if (!is.null(best_trial_metrics$n_clusters)) {
            summary_lines <- c(summary_lines,
                ">>> Clustering Information <<<",
                sprintf("  Number of Clusters: %d", best_trial_metrics$n_clusters),
                sprintf("  Number of Cells: %d", best_trial_metrics$n_cells %||% NA),
                ""
            )
        }
        
        # Individual component scores for composite
        if (!is.null(best_trial_metrics$component_scores)) {
            summary_lines <- c(summary_lines,
                ">>> Component Scores (for Composite) <<<"
            )
            for (comp_name in names(best_trial_metrics$component_scores)) {
                summary_lines <- c(summary_lines,
                    sprintf("  %s: %.4f", comp_name, best_trial_metrics$component_scores[[comp_name]])
                )
            }
            summary_lines <- c(summary_lines, "")
        }
    }
    
    # Add strategy comparison
    summary_lines <- c(summary_lines,
        paste(rep("-", 80), collapse = ""),
        "ALL STRATEGIES COMPARISON",
        paste(rep("-", 80), collapse = ""),
        sprintf("%-15s %15s %10s %10s %10s %10s", 
                "Strategy", "Best_Score", "n_hvg", "n_pcs", "n_neighbors", "resolution")
    )
    
    for (strat_name in names(results)) {
        strat_result <- results[[strat_name]]
        strat_params <- as.list(strat_result$Best_Par)
        
        is_winner <- ifelse(strat_name == winning_strategy, " *WINNER*", "")
        
        summary_lines <- c(summary_lines,
            sprintf("%-15s %15.6f %10d %10d %10d %10.4f%s",
                    strat_name,
                    strat_result$Best_Value,
                    as.integer(strat_params$n_hvg),
                    as.integer(strat_params$n_pcs),
                    as.integer(strat_params$n_neighbors),
                    strat_params$resolution,
                    is_winner)
        )
    }
    
    # Add search space bounds
    summary_lines <- c(summary_lines,
        "",
        paste(rep("-", 80), collapse = ""),
        "SEARCH SPACE BOUNDS",
        paste(rep("-", 80), collapse = ""),
        sprintf("n_hvg: [%d, %d]", SEARCH_SPACE_BOUNDS$n_hvg[1], SEARCH_SPACE_BOUNDS$n_hvg[2]),
        sprintf("n_pcs: [%d, %d]", SEARCH_SPACE_BOUNDS$n_pcs[1], SEARCH_SPACE_BOUNDS$n_pcs[2]),
        sprintf("n_neighbors: [%d, %d]", SEARCH_SPACE_BOUNDS$n_neighbors[1], SEARCH_SPACE_BOUNDS$n_neighbors[2]),
        sprintf("resolution: [%.2f, %.2f]", SEARCH_SPACE_BOUNDS$resolution[1], SEARCH_SPACE_BOUNDS$resolution[2]),
        ""
    )
    
    # Add optimization configuration
    summary_lines <- c(summary_lines,
        paste(rep("-", 80), collapse = ""),
        "OPTIMIZATION CONFIGURATION",
        paste(rep("-", 80), collapse = ""),
        sprintf("Optimization Target: %s", target),
        sprintf("Model Type: %s", args$model_type),
        sprintf("Total Iterations (n_calls): %d", args$n_calls),
        sprintf("Initial Points (n_init_points): %d", args$n_init_points),
        sprintf("Random Seed: %d", args$seed),
        sprintf("Species: %s", args$species),
        ""
    )
    
    # Add batch information if applicable
    if (GLOBAL_ENV$IS_BATCHED_DATA) {
        summary_lines <- c(summary_lines,
            paste(rep("-", 80), collapse = ""),
            "BATCH INTEGRATION",
            paste(rep("-", 80), collapse = ""),
            sprintf("Batched Data: YES"),
            sprintf("Batch Column: %s", GLOBAL_ENV$BATCH_COLUMN),
            sprintf("Integration Method: Harmony"),
            ""
        )
    } else {
        summary_lines <- c(summary_lines,
            paste(rep("-", 80), collapse = ""),
            "BATCH INTEGRATION",
            paste(rep("-", 80), collapse = ""),
            sprintf("Batched Data: NO (single sample)"),
            ""
        )
    }
    
    # Add MPS configuration if marker database was used
    if (!is.null(GLOBAL_ENV$MARKER_DB)) {
        summary_lines <- c(summary_lines,
            paste(rep("-", 80), collapse = ""),
            "MARKER PRIOR SCORE (MPS) CONFIGURATION",
            paste(rep("-", 80), collapse = ""),
            sprintf("Marker Database: %s", args$marker_db_path),
            sprintf("MPS N Top Genes: %d", args$mps_n_top_genes),
            sprintf("MPS Bonus Weight: %.2f", args$mps_bonus_weight),
            sprintf("DEG Ranking Method: %s", args$deg_ranking_method),
            sprintf("DEG Weights: FC=%.2f, Expr=%.2f, Pct=%.2f", 
                    args$deg_weight_fc, args$deg_weight_expr, args$deg_weight_pct),
            sprintf("Cell Types in Database: %d", length(GLOBAL_ENV$MARKER_DB$markers)),
            ""
        )
    }
    
    # Add data summary
    summary_lines <- c(summary_lines,
        paste(rep("-", 80), collapse = ""),
        "DATA SUMMARY",
        paste(rep("-", 80), collapse = ""),
        sprintf("Input Data: %s", args$data_dir),
        sprintf("Reference Data: %s", args$reference_path),
        sprintf("Reference Labels Column: %s", args$reference_labels_col),
        sprintf("Total Genes (after QC): %d", nrow(GLOBAL_ENV$seurat_base)),
        sprintf("Total Cells (after QC): %d", ncol(GLOBAL_ENV$seurat_base)),
        ""
    )
    
    # Footer
    summary_lines <- c(summary_lines,
        paste(rep("=", 80), collapse = ""),
        "END OF SUMMARY",
        paste(rep("=", 80), collapse = "")
    )
    
    # Write to file
    writeLines(summary_lines, best_params_summary_path)
    cat(sprintf("✅ Best parameters summary saved to: %s\n", best_params_summary_path))
    
    # Also create a simple CSV with just the best parameters for easy parsing
    best_params_csv_path <- file.path(stage1_output_dir, 
                                       paste0(args$output_prefix, "_best_parameters.csv"))
    
    best_params_df <- data.frame(
        parameter = c("n_hvg", "n_pcs", "n_neighbors", "resolution", 
                      "best_score", "winning_strategy", "optimization_target"),
        value = c(as.integer(best_params$n_hvg), 
                  as.integer(best_params$n_pcs),
                  as.integer(best_params$n_neighbors), 
                  best_params$resolution,
                  best_overall_score, 
                  winning_strategy, 
                  target),
        stringsAsFactors = FALSE
    )
    
    # Add metrics if available
    if (!is.null(best_trial_metrics)) {
        metrics_to_add <- c("cas_mean", "cas_median", "mcs", "silhouette", "mps", "n_clusters")
        for (metric_name in metrics_to_add) {
            if (!is.null(best_trial_metrics[[metric_name]])) {
                best_params_df <- rbind(best_params_df, data.frame(
                    parameter = metric_name,
                    value = as.character(best_trial_metrics[[metric_name]]),
                    stringsAsFactors = FALSE
                ))
            }
        }
    }
    
    write.csv(best_params_df, best_params_csv_path, row.names = FALSE)
    cat(sprintf("✅ Best parameters CSV saved to: %s\n", best_params_csv_path))
    
    # ==========================================================================
    # === STAGE 1.6: OPTIONAL FULL ANALYSIS IN STAGE 1 (NEW) ===
    # ==========================================================================
    # This creates consistent outputs in Stage 1 that mirror Stage 2
    # Useful for comparing optimization results without running full pipeline
    # ==========================================================================
    
    if (args$generate_stage1_full_output %||% FALSE) {
        cat("\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
        cat("### STAGE 1.6: GENERATING FULL ANALYSIS IN STAGE 1 ###\n")
        cat(paste(rep("=", 70), collapse="") %+% "\n")
        
        stage1_full_dir <- file.path(stage1_output_dir, "full_analysis")
        dir.create(stage1_full_dir, showWarnings = FALSE, recursive = TRUE)
        
        # Run visualization pipeline with best params
        cat("\n--- Running visualization pipeline with best parameters ---\n")
        
        viz_result_s1 <- tryCatch({
            run_visualization_pipeline(
                optimal_params = best_params,
                output_dir = stage1_full_dir,
                seurat_input = GLOBAL_ENV$seurat_base,
                data_dir = NULL
            )
        }, error = function(e) {
            cat(sprintf("[WARNING] Stage 1 full analysis failed: %s\n", e$message))
            NULL
        })
        
        if (!is.null(viz_result_s1)) {
            seurat_s1 <- viz_result_s1$seurat
            cas_csv_path_s1 <- viz_result_s1$cas_path
            
            # === OPTIONAL: MPS Analysis in Stage 1 ===
            if (!is.null(GLOBAL_ENV$MARKER_DB)) {
                cat("\n--- MPS Analysis (Stage 1) ---\n")
                
                mps_result_s1 <- calculate_mps(
                    seurat_obj = seurat_s1,
                    marker_db = GLOBAL_ENV$MARKER_DB,
                    group_by = "ctpt_consensus_prediction",
                    n_top_genes = args$mps_n_top_genes,
                    expand_abbreviations = args$expand_abbreviations,
                    deg_ranking_method = args$deg_ranking_method,
                    deg_weights = c(fc = args$deg_weight_fc,
                                    expr = args$deg_weight_expr, 
                                    pct = args$deg_weight_pct),
                    verbose = TRUE
                )
                
                if (!is.null(mps_result_s1$scores)) {
                    mps_df_s1 <- do.call(rbind, lapply(names(mps_result_s1$scores), function(ct) {
                        s <- mps_result_s1$scores[[ct]]
                        data.frame(
                            cell_type = ct,
                            matched_to = s$matched_type %||% NA,
                            precision = s$precision %||% NA,
                            recall = s$recall %||% NA,
                            f1 = s$f1 %||% NA,
                            n_degs = s$n_degs %||% 0,
                            n_canonical = s$n_canonical %||% 0,
                            n_overlap = s$n_overlap %||% 0,
                            stringsAsFactors = FALSE
                        )
                    }))
                    mps_csv_path_s1 <- file.path(stage1_full_dir, 
                                                  paste0(args$output_prefix, "_mps_detailed.csv"))
                    write.csv(mps_df_s1, mps_csv_path_s1, row.names = FALSE)
                    cat(sprintf("   ✓ MPS saved to: %s\n", basename(mps_csv_path_s1)))
                }
            }
            
            # === OPTIONAL: Refinement in Stage 1 ===
            if (!is.null(args$cas_refine_threshold) && args$refinement_depth > 0) {
                cat("\n--- Refinement Analysis (Stage 1) ---\n")
                
                seurat_s1 <- run_iterative_refinement_pipeline(
                    args = args,
                    seurat_s2 = seurat_s1,
                    cas_csv_path_s2 = cas_csv_path_s1,
                    output_base_dir = stage1_full_dir  # Output to Stage 1 directory
                )
            } else {
                seurat_s1$combined_annotation <- as.character(seurat_s1$ctpt_consensus_prediction)
            }
            
            # === OPTIONAL: Consistent cells export in Stage 1 ===
            cat("\n--- Consistent Cells Export (Stage 1) ---\n")
            
            consensus_mode_s1 <- if (!is.null(args$cas_refine_threshold) && args$refinement_depth > 0) {
                "refinement_threeway"
            } else {
                "biconsensus"
            }
            
            # Temporarily modify args for Stage 1 output
            args_s1 <- args
            args_s1$output_dir <- stage1_full_dir
            args_s1$final_run_prefix <- paste0(args$output_prefix, "_stage1")
            
            export_result_s1 <- export_consistent_cells(args_s1, seurat_s1, consensus_mode_s1)
            
            # Save Stage 1 Seurat object
            seurat_s1_path <- file.path(stage1_full_dir, 
                                         paste0(args$output_prefix, "_stage1_seurat.rds"))
            saveRDS(seurat_s1, seurat_s1_path)
            cat(sprintf("   ✓ Stage 1 Seurat saved to: %s\n", basename(seurat_s1_path)))
            
            cat("\n✅ Stage 1 full analysis complete\n")
        }
    } else {
        cat("\n[INFO] Stage 1 full analysis skipped (use --generate_stage1_full_output to enable)\n")
    }

    # === STAGE 2: FINAL ANALYSIS ===
    cat("\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
    cat("### STAGE 2: FINAL ANALYSIS WITH OPTIMAL PARAMETERS ###\n")
    cat(paste(rep("=", 70), collapse="") %+% "\n")
    
    stage2_output_dir <- file.path(args$output_dir, "stage_2_final_analysis")
    
    cat(sprintf("   -> Using same dataset as Stage 1: %d cells\n", ncol(GLOBAL_ENV$seurat_base)))
    if (GLOBAL_ENV$IS_BATCHED_DATA) {
        cat(sprintf("   -> Batched mode: Harmony integration with '%s'\n", GLOBAL_ENV$BATCH_COLUMN))
    }
    
    viz_result <- run_visualization_pipeline(
        optimal_params = best_params,
        output_dir = stage2_output_dir,
        seurat_input = GLOBAL_ENV$seurat_base,
        data_dir = NULL
    )
    
    seurat_s2 <- viz_result$seurat
    cas_csv_path_s2 <- viz_result$cas_path
    
    if (!is.null(GLOBAL_ENV$MARKER_DB)) {
        cat("\n--- Detailed MPS Analysis with Enhanced DEG Ranking ---\n")
        mps_result <- calculate_mps(
            seurat_obj = seurat_s2,
            marker_db = GLOBAL_ENV$MARKER_DB,
            group_by = "ctpt_consensus_prediction",
            n_top_genes = args$mps_n_top_genes,
            expand_abbreviations = args$expand_abbreviations,
            deg_ranking_method = args$deg_ranking_method,  # Uses command-line arg
            deg_weights = c(fc = args$deg_weight_fc,
                            expr = args$deg_weight_expr, 
                            pct = args$deg_weight_pct),
            verbose = TRUE
        )
        
        # Optional: Save MPS results to CSV
        if (!is.null(mps_result$scores)) {
            mps_df <- do.call(rbind, lapply(names(mps_result$scores), function(ct) {
                s <- mps_result$scores[[ct]]
                data.frame(
                    cell_type = ct,
                    matched_to = s$matched_type %||% NA,
                    precision = s$precision %||% NA,
                    recall = s$recall %||% NA,
                    f1 = s$f1 %||% NA,
                    n_degs = s$n_degs %||% 0,
                    n_canonical = s$n_canonical %||% 0,
                    n_overlap = s$n_overlap %||% 0,
                    stringsAsFactors = FALSE
                )
            }))
            mps_csv_path <- file.path(stage2_output_dir, 
                                       paste0(args$final_run_prefix, "_mps_detailed.csv"))
            write.csv(mps_df, mps_csv_path, row.names = FALSE)
            cat(sprintf("   Saved detailed MPS to: %s\n", mps_csv_path))
        }
    }
    
    # === STAGE 3: ITERATIVE REFINEMENT ===
    if (!is.null(args$cas_refine_threshold) && args$refinement_depth > 0) {
        cat("\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
        cat("### STAGE 3: ITERATIVE REFINEMENT PIPELINE ###\n")
        cat(paste(rep("=", 70), collapse="") %+% "\n")
        
        seurat_s2 <- run_iterative_refinement_pipeline(
                args = args,
                seurat_s2 = seurat_s2,
                cas_csv_path_s2 = cas_csv_path_s2,
                output_base_dir = NULL  # Use default Stage 1/Stage 2 split
            )
    } else {
        cat("\n[INFO] Refinement skipped\n")
        seurat_s2$combined_annotation <- as.character(seurat_s2$ctpt_consensus_prediction)
    }
    
    # === STAGE 4: EXPORT CONSISTENT CELLS ===
    cat("\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
    cat("### STAGE 4: EXPORT CONSISTENT CELLS ###\n")
    cat(paste(rep("=", 70), collapse="") %+% "\n")
    
    consensus_mode <- if (!is.null(args$cas_refine_threshold) && args$refinement_depth > 0) {
        "refinement_threeway"
    } else {
        "biconsensus"
    }
    export_result <- export_consistent_cells(args, seurat_s2, consensus_mode)
    
    # ==========================================================================
    # === STAGE 5: MARKER-BASED ANNOTATION ON FILTERED CELLS ===
    # ==========================================================================
    # This runs AFTER consistency filtering, using only high-confidence cells
    # Low-confidence (inconsistent) cells are shown in grey on UMAP 1
    # ==========================================================================
    
    if (!is.null(GLOBAL_ENV$MARKER_DB) && !is.null(export_result)) {
        cat("\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
        cat("### STAGE 5: MARKER-BASED ANNOTATION ###\n")
        cat(paste(rep("=", 70), collapse="") %+% "\n")
        
        marker_annot_dir <- file.path(args$output_dir, "stage_2_final_analysis", "marker_based_annotation")
        dir.create(marker_annot_dir, showWarnings = FALSE, recursive = TRUE)
        
        # Get objects from export result
        seurat_consistent <- export_result$seurat_filtered
        seurat_full <- export_result$seurat_full_with_consistency
        consistency_mask <- export_result$consistency_mask
        
        cat(sprintf("   Total cells: %d\n", ncol(seurat_full)))
        cat(sprintf("   Consistent cells: %d\n", ncol(seurat_consistent)))
        
        # ======================================================================
        # PART A: MARKER ANNOTATION ON ALL CELLS (existing functionality)
        # Inconsistent cells shown as grey
        # ======================================================================
        cat("\n--- Part A: Marker annotation on ALL cells (grey for inconsistent) ---\n")
        
        # Calculate MPS using ALL cells but group by final label
        mps_result_all <- calculate_mps(
            seurat_obj = seurat_full,
            marker_db = GLOBAL_ENV$MARKER_DB,
            group_by = export_result$final_label_col,
            n_top_genes = args$mps_n_top_genes,
            expand_abbreviations = args$expand_abbreviations,
            deg_ranking_method = args$deg_ranking_method,
            deg_weights = c(fc = args$deg_weight_fc,
                            expr = args$deg_weight_expr, 
                            pct = args$deg_weight_pct),
            verbose = TRUE
        )
        
        if (!is.null(mps_result_all$scores)) {
            # Build marker annotation lookup from all-cells MPS
            marker_annot_df_all <- do.call(rbind, lapply(names(mps_result_all$scores), function(ct) {
                s <- mps_result_all$scores[[ct]]
                data.frame(
                    original_annotation = ct,
                    marker_based_annotation = s$matched_type %||% NA,
                    mps_precision = s$precision %||% NA,
                    mps_recall = s$recall %||% NA,
                    mps_f1 = s$f1 %||% NA,
                    n_degs = s$n_degs %||% 0,
                    n_canonical = s$n_canonical %||% 0,
                    n_overlap = s$n_overlap %||% 0,
                    stringsAsFactors = FALSE
                )
            }))
            
            # Save all-cells cluster summary
            marker_summary_all_csv <- file.path(marker_annot_dir,
                paste0(args$final_run_prefix, "_marker_annotation_summary_all_cells.csv"))
            write.csv(marker_annot_df_all, marker_summary_all_csv, row.names = FALSE)
            cat(sprintf("   ✓ All-cells cluster summary: %s\n", basename(marker_summary_all_csv)))
            
            # Create lookup for all cells
            lookup_all <- setNames(marker_annot_df_all$marker_based_annotation,
                                   marker_annot_df_all$original_annotation)
            
            # Add marker annotation to full object
            seurat_full$marker_based_annotation_all <- NA_character_
            final_labels <- as.character(seurat_full@meta.data[[export_result$final_label_col]])
            
            for (i in seq_len(ncol(seurat_full))) {
                if (consistency_mask[i]) {
                    orig_label <- final_labels[i]
                    if (!is.null(lookup_all[[orig_label]]) && !is.na(lookup_all[[orig_label]])) {
                        seurat_full$marker_based_annotation_all[i] <- lookup_all[[orig_label]]
                    }
                }
            }
            
            # Save all-cells cell-level annotations
            cell_annot_all_df <- data.frame(
                cell_barcode = colnames(seurat_full),
                original_annotation = final_labels,
                marker_based_annotation = seurat_full$marker_based_annotation_all,
                consistency_status = seurat_full$consistency_status,
                stringsAsFactors = FALSE
            )
            
            cell_annot_all_csv <- file.path(marker_annot_dir,
                paste0(args$final_run_prefix, "_marker_annotation_cells_all.csv"))
            write.csv(cell_annot_all_df, cell_annot_all_csv, row.names = FALSE)
            cat(sprintf("   ✓ All-cells annotations: %s\n", basename(cell_annot_all_csv)))
            
            # Generate UMAP with grey for inconsistent (existing functionality)
            if ("umap" %in% names(seurat_full@reductions)) {
                umap_coords_full <- Embeddings(seurat_full, reduction = "umap")
                unique_marker_types_all <- sort(unique(na.omit(seurat_full$marker_based_annotation_all)))
                n_marker_types_all <- length(unique_marker_types_all)
                
                if (n_marker_types_all > 0) {
                    marker_colors_all <- get_plot_colors(n_marker_types_all)
                    names(marker_colors_all) <- unique_marker_types_all
                    all_colors <- c(marker_colors_all, "Filtered" = "grey70")
                    
                    plot_labels_all <- ifelse(
                        is.na(seurat_full$marker_based_annotation_all),
                        "Filtered",
                        seurat_full$marker_based_annotation_all
                    )
                    
                    plot_df_all <- data.frame(
                        UMAP_1 = umap_coords_full[, 1],
                        UMAP_2 = umap_coords_full[, 2],
                        marker_annotation = factor(plot_labels_all, 
                            levels = c("Filtered", unique_marker_types_all)),
                        stringsAsFactors = FALSE
                    )
                    plot_df_all <- plot_df_all[order(plot_df_all$marker_annotation != "Filtered"), ]
                    
                    n_filtered <- sum(plot_labels_all == "Filtered")
                    n_annotated <- sum(plot_labels_all != "Filtered")
                    
                    p_umap_all_grey <- ggplot(plot_df_all, aes(x = UMAP_1, y = UMAP_2, color = marker_annotation)) +
                        geom_point(size = 1, alpha = 0.7) +
                        scale_color_manual(values = all_colors, name = "Cell Type") +
                        labs(
                            title = "Marker-Based Annotation (All Cells, DEGs from all)",
                            subtitle = sprintf("Annotated: %d | Filtered (grey): %d", n_annotated, n_filtered),
                            x = "UMAP 1", y = "UMAP 2"
                        ) +
                        theme_umap_clean()
                    
                    umap_all_grey_path <- file.path(marker_annot_dir,
                        paste0(args$final_run_prefix, "_umap_all_cells_with_grey.png"))
                    ggsave(umap_all_grey_path, plot = p_umap_all_grey, width = 12, height = 10, dpi = args$fig_dpi)
                    cat(sprintf("   ✓ UMAP all cells (with grey): %s\n", basename(umap_all_grey_path)))
                }
            }
        }
        
        # ======================================================================
        # PART B: MARKER ANNOTATION ON CONSISTENT CELLS ONLY (NEW)
        # DEGs computed only from consistent cells, no grey cells in output
        # ======================================================================
        cat("\n--- Part B: Marker annotation on CONSISTENT CELLS ONLY ---\n")
        cat(sprintf("   Processing %d consistent cells only\n", ncol(seurat_consistent)))

        # Calculate MPS using ONLY consistent cells
        mps_result_consistent <- calculate_mps(
            seurat_obj = seurat_consistent,
            marker_db = GLOBAL_ENV$MARKER_DB,
            group_by = export_result$final_label_col,
            n_top_genes = args$mps_n_top_genes,
            expand_abbreviations = args$expand_abbreviations,
            deg_ranking_method = args$deg_ranking_method,
            deg_weights = c(fc = args$deg_weight_fc,
                            expr = args$deg_weight_expr, 
                            pct = args$deg_weight_pct),
            verbose = TRUE
        )

        if (!is.null(mps_result_consistent$scores)) {
            # Build marker annotation dataframe for consistent cells only
            marker_annot_df_consistent <- do.call(rbind, lapply(names(mps_result_consistent$scores), function(ct) {
                s <- mps_result_consistent$scores[[ct]]
                
                # Safely handle overlapping_genes
                overlap_genes_str <- if (!is.null(s$overlapping_genes) && length(s$overlapping_genes) > 0) {
                    paste(s$overlapping_genes, collapse = ";")
                } else {
                    ""
                }
                
                data.frame(
                    original_annotation = ct,
                    marker_based_annotation = s$matched_type %||% NA,
                    mps_precision = s$precision %||% NA,
                    mps_recall = s$recall %||% NA,
                    mps_f1 = s$f1 %||% NA,
                    n_degs = s$n_degs %||% 0,
                    n_canonical = s$n_canonical %||% 0,
                    n_overlap = s$n_overlap %||% 0,
                    overlapping_genes = overlap_genes_str,
                    stringsAsFactors = FALSE
                )
            }))
            
            # Save consistent-cells cluster summary
            marker_summary_consistent_csv <- file.path(marker_annot_dir,
                paste0(args$final_run_prefix, "_marker_annotation_summary_consistent_only.csv"))
            write.csv(marker_annot_df_consistent, marker_summary_consistent_csv, row.names = FALSE)
            cat(sprintf("   ✓ Consistent-only cluster summary: %s\n", basename(marker_summary_consistent_csv)))
            
            # Create lookup for consistent cells
            lookup_consistent <- setNames(marker_annot_df_consistent$marker_based_annotation,
                                        marker_annot_df_consistent$original_annotation)
            
            # === FIXED: Add marker annotation to consistent cells object ===
            # Get original labels
            consistent_labels <- as.character(seurat_consistent@meta.data[[export_result$final_label_col]])
            
            # Create annotation vector WITHOUT names (critical fix)
            marker_annotations <- vapply(consistent_labels, function(lbl) {
                if (lbl %in% names(lookup_consistent) && !is.na(lookup_consistent[[lbl]])) {
                    return(lookup_consistent[[lbl]])
                } else {
                    return(NA_character_)
                }
            }, FUN.VALUE = character(1), USE.NAMES = FALSE)
            
            # Add directly to metadata slot
            seurat_consistent@meta.data$marker_based_annotation <- marker_annotations
            
            cat(sprintf("   ✓ Added marker annotations to %d cells\n", 
                        sum(!is.na(marker_annotations))))
            
            # Save consistent-cells cell-level annotations
            cell_annot_consistent_df <- data.frame(
                cell_barcode = colnames(seurat_consistent),
                original_annotation = consistent_labels,
                marker_based_annotation = seurat_consistent@meta.data$marker_based_annotation,
                stringsAsFactors = FALSE
            )
            
            cell_annot_consistent_csv <- file.path(marker_annot_dir,
                paste0(args$final_run_prefix, "_marker_annotation_cells_consistent_only.csv"))
            write.csv(cell_annot_consistent_df, cell_annot_consistent_csv, row.names = FALSE)
            cat(sprintf("   ✓ Consistent-only cell annotations: %s\n", basename(cell_annot_consistent_csv)))
            
            # Cell type distribution for consistent cells
            ct_dist_consistent <- as.data.frame(table(seurat_consistent@meta.data$marker_based_annotation))
            colnames(ct_dist_consistent) <- c("MarkerBasedCellType", "Count")
            ct_dist_consistent$Percentage <- 100 * ct_dist_consistent$Count / sum(ct_dist_consistent$Count)
            ct_dist_consistent <- ct_dist_consistent[order(-ct_dist_consistent$Count), ]
            
            ct_dist_csv <- file.path(marker_annot_dir,
                paste0(args$final_run_prefix, "_marker_celltype_distribution_consistent_only.csv"))
            write.csv(ct_dist_consistent, ct_dist_csv, row.names = FALSE)
            cat(sprintf("   ✓ Cell type distribution: %s\n", basename(ct_dist_csv)))
            
            # Generate UMAP for consistent cells only (NO grey cells)
            if ("umap" %in% names(seurat_consistent@reductions)) {
                umap_coords_consistent <- Embeddings(seurat_consistent, reduction = "umap")
                unique_marker_types_consistent <- sort(unique(na.omit(seurat_consistent@meta.data$marker_based_annotation)))
                n_marker_types_consistent <- length(unique_marker_types_consistent)
                
                if (n_marker_types_consistent > 0) {
                    marker_colors_consistent <- get_plot_colors(n_marker_types_consistent)
                    names(marker_colors_consistent) <- unique_marker_types_consistent
                    
                    # Filter out cells with NA marker annotation (if any)
                    valid_cells <- !is.na(seurat_consistent@meta.data$marker_based_annotation)
                    
                    plot_df_consistent <- data.frame(
                        UMAP_1 = umap_coords_consistent[valid_cells, 1],
                        UMAP_2 = umap_coords_consistent[valid_cells, 2],
                        marker_annotation = seurat_consistent@meta.data$marker_based_annotation[valid_cells],
                        cell_barcode = colnames(seurat_consistent)[valid_cells],
                        stringsAsFactors = FALSE
                    )
                    
                    # UMAP: Consistent cells only with marker-based annotation
                    p_umap_consistent <- ggplot(plot_df_consistent, 
                                                aes(x = UMAP_1, y = UMAP_2, color = marker_annotation)) +
                        geom_point(size = 1.2, alpha = 0.8) +
                        scale_color_manual(values = marker_colors_consistent, name = "Cell Type") +
                        labs(
                            title = "Marker-Based Annotation (Consistent Cells Only)",
                            subtitle = sprintf("n = %d cells, %d cell types (DEGs from consistent cells)", 
                                            nrow(plot_df_consistent), n_marker_types_consistent),
                            x = "UMAP 1", y = "UMAP 2"
                        ) +
                        theme_umap_clean()
                    
                    umap_consistent_path <- file.path(marker_annot_dir,
                        paste0(args$final_run_prefix, "_umap_consistent_only_marker_annotation.png"))
                    ggsave(umap_consistent_path, plot = p_umap_consistent, 
                        width = 12, height = 10, dpi = args$fig_dpi)
                    cat(sprintf("   ✓ UMAP consistent only: %s\n", basename(umap_consistent_path)))
                    
                    # Save UMAP coordinates for consistent cells
                    coords_consistent_df <- data.frame(
                        cell_barcode = colnames(seurat_consistent),
                        UMAP_1 = umap_coords_consistent[, 1],
                        UMAP_2 = umap_coords_consistent[, 2],
                        original_annotation = consistent_labels,
                        marker_based_annotation = seurat_consistent@meta.data$marker_based_annotation,
                        stringsAsFactors = FALSE
                    )
                    
                    coords_consistent_csv <- file.path(marker_annot_dir,
                        paste0(args$final_run_prefix, "_umap_coordinates_consistent_only.csv"))
                    write.csv(coords_consistent_df, coords_consistent_csv, row.names = FALSE)
                    cat(sprintf("   ✓ UMAP coordinates (consistent): %s\n", basename(coords_consistent_csv)))
                    
                    # === COMPARISON: Original vs Marker-based on consistent cells ===
                    p_original_consistent <- DimPlot(seurat_consistent, reduction = "umap",
                                                    group.by = export_result$final_label_col,
                                                    pt.size = 0.8) +
                        ggtitle("Original Annotation\n(Consistent Cells)") +
                        theme_umap_clean() +
                        theme(legend.position = "bottom",
                            legend.text = element_text(size = 7))
                    
                    p_marker_consistent <- p_umap_consistent + 
                        ggtitle("Marker-Based Annotation\n(Consistent Cells)") +
                        theme(legend.position = "bottom",
                            legend.text = element_text(size = 7))
                    
                    p_comparison_consistent <- p_original_consistent | p_marker_consistent
                    
                    comparison_consistent_path <- file.path(marker_annot_dir,
                        paste0(args$final_run_prefix, "_umap_comparison_consistent_only.png"))
                    ggsave(comparison_consistent_path, plot = p_comparison_consistent, 
                        width = 20, height = 10, dpi = args$fig_dpi)
                    cat(sprintf("   ✓ UMAP comparison (consistent): %s\n", basename(comparison_consistent_path)))
                    
                } else {
                    cat("   [WARNING] No marker-based annotations for consistent cells.\n")
                }
            } else {
                cat("   [WARNING] No UMAP reduction in consistent cells object.\n")
            }
            
            # === COMPARISON BETWEEN ALL-CELLS AND CONSISTENT-ONLY MPS ===
            cat("\n--- MPS Comparison: All Cells vs Consistent Only ---\n")
            
            if (!is.null(mps_result_all$scores) && !is.null(mps_result_consistent$scores)) {
                comparison_df <- data.frame(
                    cell_type = character(),
                    mps_f1_all_cells = numeric(),
                    mps_f1_consistent_only = numeric(),
                    f1_difference = numeric(),
                    stringsAsFactors = FALSE
                )
                
                all_types <- union(names(mps_result_all$scores), names(mps_result_consistent$scores))
                
                for (ct in all_types) {
                    f1_all <- if (ct %in% names(mps_result_all$scores)) {
                        mps_result_all$scores[[ct]]$f1 %||% NA
                    } else { NA }
                    
                    f1_consistent <- if (ct %in% names(mps_result_consistent$scores)) {
                        mps_result_consistent$scores[[ct]]$f1 %||% NA
                    } else { NA }
                    
                    f1_diff <- if (!is.na(f1_all) && !is.na(f1_consistent)) {
                        f1_consistent - f1_all
                    } else { NA }
                    
                    comparison_df <- rbind(comparison_df, data.frame(
                        cell_type = ct,
                        mps_f1_all_cells = f1_all,
                        mps_f1_consistent_only = f1_consistent,
                        f1_difference = f1_diff,
                        stringsAsFactors = FALSE
                    ))
                }
                
                comparison_df <- comparison_df[order(-comparison_df$f1_difference, na.last = TRUE), ]
                
                comparison_csv <- file.path(marker_annot_dir,
                    paste0(args$final_run_prefix, "_mps_comparison_all_vs_consistent.csv"))
                write.csv(comparison_df, comparison_csv, row.names = FALSE)
                cat(sprintf("   ✓ MPS comparison: %s\n", basename(comparison_csv)))
                
                # Print summary
                avg_f1_all <- mean(comparison_df$mps_f1_all_cells, na.rm = TRUE)
                avg_f1_consistent <- mean(comparison_df$mps_f1_consistent_only, na.rm = TRUE)
                
                cat(sprintf("\n   MPS F1 Summary:\n"))
                cat(sprintf("      All cells mean F1:        %.4f\n", avg_f1_all))
                cat(sprintf("      Consistent-only mean F1:  %.4f\n", avg_f1_consistent))
                cat(sprintf("      Difference:               %.4f\n", avg_f1_consistent - avg_f1_all))
            }
            
            # Save updated Seurat object with marker annotations
            seurat_consistent_annotated_path <- file.path(marker_annot_dir,
                paste0(args$final_run_prefix, "_seurat_consistent_marker_annotated.rds"))
            saveRDS(seurat_consistent, seurat_consistent_annotated_path)
            cat(sprintf("   ✓ Seurat object (consistent + marker): %s\n", 
                        basename(seurat_consistent_annotated_path)))
            
        } else {
            cat("   [INFO] MPS calculation on consistent cells returned no scores.\n")
        }
        
        cat("\n   ✅ Marker-based annotation complete (both all-cells and consistent-only)\n")
        
    } else {
        cat("\n[INFO] Marker-based annotation skipped (no marker DB or no consistent cells)\n")
    }

    # === FINAL SUMMARY ===
    cat("\n" %+% paste(rep("=", 70), collapse="") %+% "\n")
    cat("### PIPELINE COMPLETED SUCCESSFULLY ###\n")
    cat(paste(rep("=", 70), collapse="") %+% "\n")
    cat(sprintf("   Output directory: %s\n", args$output_dir))
    cat(sprintf("   Final cells: %d\n", ncol(seurat_s2)))
    if (GLOBAL_ENV$IS_BATCHED_DATA) {
        cat(sprintf("   Integration mode: BATCHED (Harmony)\n"))
        cat(sprintf("   Batch column: %s\n", GLOBAL_ENV$BATCH_COLUMN))
    } else {
        cat(sprintf("   Integration mode: SINGLE-SAMPLE\n"))
    }
    if (args$model_type == "mps_bonus") {
        cat(sprintf("   MPS scoring: ADDITIVE BONUS (weight=%.2f)\n", args$mps_bonus_weight))
    } else if (args$model_type == "mps_integrated") {
        cat(sprintf("   MPS scoring: MULTIPLICATIVE (legacy)\n"))
    }
    if (!is.null(export_result)) {
        cat(sprintf("   Consistent cells exported: %d\n", export_result$n_filtered))
    }
    
    return(list(
        seurat_obj = seurat_s2,
        optimal_params = best_params,
        export_result = export_result,
        is_batched = GLOBAL_ENV$IS_BATCHED_DATA,
        batch_column = GLOBAL_ENV$BATCH_COLUMN
    ))
}

# ==============================================================================
# --- ARGUMENT PARSING ---
# ==============================================================================

#' Parse command line arguments
parse_arguments <- function() {
    parser <- ArgumentParser(description = "Integrated Bayesian Optimization Pipeline for scRNA-seq")
    
    # ==========================================================================
    # Input/Output
    # ==========================================================================
    parser$add_argument("--data_dir", required = TRUE, 
                        help = "Path to input data (10x directory, .h5, or .rds)")
    parser$add_argument("--output_dir", default = "output", 
                        help = "Output directory")
    parser$add_argument("--output_prefix", default = "scrna", 
                        help = "Prefix for output files")
    parser$add_argument("--final_run_prefix", default = "final", 
                        help = "Prefix for final analysis files")
    
    # ==========================================================================
    # Reference
    # ==========================================================================
    parser$add_argument("--reference_path", required = TRUE, 
                        help = "Path to reference Seurat object (.rds)")
    parser$add_argument("--reference_labels_col", default = "cell_type", 
                        help = "Column in reference containing cell type labels")
    parser$add_argument("--reference_assay", default = "RNA", 
                        help = "Assay to use from reference object")
    
    # ==========================================================================
    # Species
    # ==========================================================================
    parser$add_argument("--species", default = "human", 
                        help = "Species (human or mouse)")
    
    # ==========================================================================
    # Batch Integration
    # ==========================================================================
    parser$add_argument("--batch_col", default = "NULL", 
                        help = "Column name for batch information. Use 'NULL' for auto-detect.")
    parser$add_argument("--barcode_separator", default = "_",
                        help = "Separator character in barcodes for batch extraction")
    parser$add_argument("--harmony_theta", type = "double", default = 2.0,
                        help = "Harmony theta parameter")
    parser$add_argument("--harmony_lambda", type = "double", default = 1.0,
                        help = "Harmony lambda parameter")
    parser$add_argument("--harmony_max_iter", type = "integer", default = 20,
                        help = "Maximum Harmony iterations")
    parser$add_argument("--skip_integration", action = "store_true",
                        help = "Skip batch integration even if multiple batches detected")
    
    # ==========================================================================
    # Optimization
    # ==========================================================================
    parser$add_argument("--target", default = "balanced", 
                        help = "Optimization target: balanced, weighted_cas, simple_cas, mcs, mps, all")
    parser$add_argument("--model_type", default = "annotation", 
                        help = "Model type: annotation, structural, silhouette, mps_integrated, mps_bonus")
    parser$add_argument("--n_calls", type = "integer", default = 50, 
                        help = "Total optimization iterations")
    parser$add_argument("--n_init_points", type = "integer", default = 10, 
                        help = "Initial random sampling points")
    parser$add_argument("--marker_gene_model", default = "all", 
                        help = "Marker gene filtering: all or non-mitochondrial")
    
    # ==========================================================================
    # HVG Filtering
    # ==========================================================================
    parser$add_argument("--hvg_min_mean", type = "double", default = -999999, 
                        help = "Min mean expression for HVG (default: -999999 for -Inf)")
    parser$add_argument("--hvg_max_mean", type = "double", default = 999999, 
                        help = "Max mean expression for HVG (default: 999999 for Inf)")
    parser$add_argument("--hvg_min_disp", type = "double", default = -999999, 
                        help = "Min dispersion for HVG (default: -999999 for -Inf)")
    
    # ==========================================================================
    # CAS Aggregation
    # ==========================================================================
    parser$add_argument("--cas_aggregation_method", default = "leiden", 
                        help = "How to aggregate CAS scores: leiden or consensus")
    
    # ==========================================================================
    # Subsampling (deprecated)
    # ==========================================================================
    parser$add_argument("--subsample_n_cells", type = "integer", default = -1, 
                        help = "[DEPRECATED] Ignored for consistency. Use -1 for no subsampling.")
    
    # ==========================================================================
    # Refinement
    # ==========================================================================
    parser$add_argument("--cas_refine_threshold", type = "double", default = -1.0,
                        help = "CAS threshold for refinement (e.g., 70). Use -1 to disable.")
    parser$add_argument("--refinement_depth", type = "integer", default = 3,
                        help = "Maximum refinement iterations")
    parser$add_argument("--min_cells_refinement", type = "integer", default = 50,
                        help = "Minimum cells to continue refinement")
    
    # ==========================================================================
    # Consistent Cells Export
    # ==========================================================================
    parser$add_argument("--min_cells_per_type", type = "integer", default = -1,
                        help = "Filter cell types with fewer cells. Use -1 to disable.")
    
    # ==========================================================================
    # Spatial Data
    # ==========================================================================
    parser$add_argument("--st_data_dir", default = "NULL", 
                        help = "Path to spatial transcriptomics data. Use 'NULL' to disable.")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    parser$add_argument("--fig_dpi", type = "integer", default = 300, 
                        help = "DPI for saved figures")
    
    # ==========================================================================
    # General
    # ==========================================================================
    parser$add_argument("--seed", type = "integer", default = 42, 
                        help = "Random seed for reproducibility")
    
    # ==========================================================================
    # MPS Configuration
    # ==========================================================================
    parser$add_argument("--marker_db_path", default = "NULL",
                        help = "Path to marker gene database CSV. Use 'NULL' to disable.")
    parser$add_argument("--mps_n_top_genes", type = "integer", default = 50,
                        help = "Number of top DEGs to check for MPS calculation")
    parser$add_argument("--mps_min_pct", type = "double", default = 0.1,
                        help = "Minimum fraction of cells expressing gene for DEG")
    parser$add_argument("--mps_logfc_threshold", type = "double", default = 0.25,
                        help = "Log fold change threshold for DEGs")
    parser$add_argument("--mps_bonus_weight", type = "double", default = 0.2,
                        help = "Weight for MPS bonus in additive scoring (0-1)")
    parser$add_argument("--marker_prior_species", 
                        type = "character",
                        default = "human",
                        help = "Species filter for marker database (human/mouse)")
    parser$add_argument("--marker_prior_organ", 
                        type = "character",
                        default = "NULL",
                        help = "Organ/tissue filter for marker database (e.g., Brain, Liver). Use 'NULL' for no filter.")
    # ==========================================================================
    # DEG Ranking Method
    # ==========================================================================
    parser$add_argument("--deg_ranking_method", default = "composite",
                        help = "DEG ranking: original (log2FC only) or composite (weighted)")
    parser$add_argument("--deg_weight_fc", type = "double", default = 0.4,
                        help = "Weight for log2FC in composite DEG ranking [0-1]")
    parser$add_argument("--deg_weight_expr", type = "double", default = 0.3,
                        help = "Weight for expression level in composite DEG ranking [0-1]")
    parser$add_argument("--deg_weight_pct", type = "double", default = 0.3,
                        help = "Weight for detection rate specificity [0-1]")
    
    # ==========================================================================
    # Cell Type Abbreviation Expansion
    # ==========================================================================
    parser$add_argument("--expand_abbreviations", action = "store_true",
                        help = "Expand cell type abbreviations (e.g., OPC -> Oligodendrocyte Precursor Cell)")
    
    # ==========================================================================
    # Parse arguments
    # ==========================================================================
    args <- parser$parse_args()
    
    # ==========================================================================
    # Post-processing: Convert sentinel values
    # ==========================================================================
    
    # Convert string "NULL" to actual NULL
    if (!is.null(args$st_data_dir) && args$st_data_dir == "NULL") {
        args$st_data_dir <- NULL
    }
    if (!is.null(args$marker_db_path) && args$marker_db_path == "NULL") {
        args$marker_db_path <- NULL
    }
    if (!is.null(args$batch_col) && args$batch_col == "NULL") {
        args$batch_col <- NULL
    }
    if (!is.null(args$marker_prior_organ) && args$marker_prior_organ == "NULL") {
        args$marker_prior_organ <- NULL
    }
    # Convert -1 sentinel values to NULL
    if (!is.null(args$subsample_n_cells) && args$subsample_n_cells == -1) {
        args$subsample_n_cells <- NULL
    }
    if (!is.null(args$min_cells_per_type) && args$min_cells_per_type == -1) {
        args$min_cells_per_type <- NULL
    }
    if (!is.null(args$cas_refine_threshold) && args$cas_refine_threshold == -1.0) {
        args$cas_refine_threshold <- NULL
    }
    
    # Convert sentinel values to Inf/-Inf for HVG parameters
    if (!is.null(args$hvg_min_mean) && args$hvg_min_mean <= -999998) {
        args$hvg_min_mean <- -Inf
    }
    if (!is.null(args$hvg_max_mean) && args$hvg_max_mean >= 999998) {
        args$hvg_max_mean <- Inf
    }
    if (!is.null(args$hvg_min_disp) && args$hvg_min_disp <= -999998) {
        args$hvg_min_disp <- -Inf
    }
    
    # Validate DEG weights sum to 1
    deg_weight_sum <- args$deg_weight_fc + args$deg_weight_expr + args$deg_weight_pct
    if (abs(deg_weight_sum - 1.0) > 0.001) {
        cat(sprintf("[WARNING] DEG weights sum to %.3f, normalizing to 1.0\n", deg_weight_sum))
        args$deg_weight_fc <- args$deg_weight_fc / deg_weight_sum
        args$deg_weight_expr <- args$deg_weight_expr / deg_weight_sum
        args$deg_weight_pct <- args$deg_weight_pct / deg_weight_sum
    }
    
    # Validate MPS bonus weight
    if (args$mps_bonus_weight < 0 || args$mps_bonus_weight > 1) {
        cat("[WARNING] mps_bonus_weight should be between 0 and 1, clamping value\n")
        args$mps_bonus_weight <- max(0, min(1, args$mps_bonus_weight))
    }
    
    # Set default for expand_abbreviations if NULL (action="store_true" returns NULL if not set)
    if (is.null(args$expand_abbreviations)) {
        args$expand_abbreviations <- FALSE
    }
    
    # Set default for skip_integration if NULL
    if (is.null(args$skip_integration)) {
        args$skip_integration <- FALSE
    }
    
    return(args)
}

# ==============================================================================
# --- MAIN ENTRY POINT ---
# ==============================================================================

if (!interactive()) {
    args <- parse_arguments()
    result <- main(args)
    cat("\n✅ Pipeline completed successfully!\n")
}