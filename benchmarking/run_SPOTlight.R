#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(SPOTlight)
  library(Seurat)
  library(Matrix)
  library(data.table)
  library(pheatmap)
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(RColorBrewer)
  library(viridis)
  library(FNN)  # For nearest neighbor distance calculation
})

# =============================================================================
# ACCURACY IMPROVEMENT SETTINGS (NEW)
# =============================================================================
# NMF model parameters for better convergence
NMF_MAX_ITER <- 500        # Increased from default ~100
NMF_CONV_THRESH <- 1e-5    # Convergence threshold
MIN_PROP_THRESHOLD <- 0.01 # Filter out very low proportions as noise

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
parser <- ArgumentParser(description = "Run SPOTlight for spatial deconvolution")

parser$add_argument("--sc_counts", required = TRUE,
                    help = "Path to scRNA-seq counts CSV (genes x cells)")
parser$add_argument("--sc_labels", required = TRUE,
                    help = "Path to cell type labels CSV")
parser$add_argument("--st_counts", required = TRUE,
                    help = "Path to spatial counts CSV (genes x spots)")
parser$add_argument("--output_csv", required = TRUE,
                    help = "Output CSV for proportions")
parser$add_argument("--output_plot", required = TRUE,
                    help = "Output heatmap PNG")
parser$add_argument("--use_dummy_coords", action = "store_true", default = FALSE,
                    help = "Generate dummy coordinates if not available")
parser$add_argument("--st_coords", default = NULL,
                    help = "Path to spatial coordinates CSV, or 'auto' to extract from counts")
parser$add_argument("--ground_truth", default = NULL,
                    help = "Path to ground truth proportions CSV (for simulated data)")
parser$add_argument("--n_hvg", type = "integer", default = 3000,
                    help = "Number of highly variable genes to use")
parser$add_argument("--n_cells_per_type", type = "integer", default = 100,
                    help = "Max cells per cell type for training")
parser$add_argument("--transpose_sc", action = "store_true", default = FALSE,
                    help = "Transpose scRNA-seq counts (if cells are rows)")
parser$add_argument("--transpose_st", action = "store_true", default = FALSE,
                    help = "Transpose spatial counts (if spots are rows)")
# NEW ARGUMENTS FOR ACCURACY
parser$add_argument("--n_top_markers", type = "integer", default = 200,
                    help = "Number of top marker genes per cell type (default: 200)")
parser$add_argument("--min_pct", type = "double", default = 0.1,
                    help = "Minimum percentage of cells expressing marker (default: 0.1)")
parser$add_argument("--logfc_threshold", type = "double", default = 0.15,
                    help = "Log fold change threshold for markers (default: 0.15)")
parser$add_argument("--model_gene_scale", action = "store_true", default = TRUE,
                    help = "Scale genes in NMF model (default: TRUE)")
parser$add_argument("--min_prop", type = "double", default = 0.01,
                    help = "Minimum proportion threshold - values below set to 0 (default: 0.01)")
parser$add_argument("--hex_orientation", type = "integer", default = 0,
                    help = "Hexagon orientation angle in degrees (0 = flat-top, 30 = pointy-top)")

args <- parser$parse_args()

# -----------------------------------------------------------------------------
# Helper Function: Get Seurat version-compatible assay data
# -----------------------------------------------------------------------------
get_counts_matrix <- function(seurat_obj) {
  seurat_version <- packageVersion("SeuratObject")
  
  if (seurat_version >= "5.0.0") {
    counts <- GetAssayData(seurat_obj, layer = "counts")
  } else {
    counts <- GetAssayData(seurat_obj, slot = "counts")
  }
  
  return(counts)
}

# -----------------------------------------------------------------------------
# Helper Function: Extract cell type from spot name
# -----------------------------------------------------------------------------
extract_celltype_from_spot <- function(spot_name) {
  if (grepl("_Pure_", spot_name)) {
    celltype <- sub("^Batch_[0-9]+_Pure_", "", spot_name)
  } else if (grepl("_Mix_", spot_name)) {
    celltype <- sub("^Batch_[0-9]+_Mix_[0-9]+pct_", "", spot_name)
    celltype <- sub("_[0-9]+pct_.*$", "", celltype)
  } else {
    celltype <- spot_name
  }
  
  celltype <- gsub("_", " ", celltype)
  
  return(celltype)
}

# -----------------------------------------------------------------------------
# Helper Function: Normalize cell type name for matching
# -----------------------------------------------------------------------------
normalize_name <- function(name) {
  name <- tolower(name)
  name <- gsub("\\s+", " ", name)
  name <- trimws(name)
  return(name)
}

# -----------------------------------------------------------------------------
# Helper Function: Find best matching column name
# -----------------------------------------------------------------------------
find_matching_column <- function(celltype, column_names) {
  norm_celltype <- normalize_name(celltype)
  norm_columns <- sapply(column_names, normalize_name)
  
  exact_match <- which(norm_columns == norm_celltype)
  if (length(exact_match) > 0) {
    return(column_names[exact_match[1]])
  }
  
  for (i in seq_along(column_names)) {
    if (grepl(norm_celltype, norm_columns[i], fixed = TRUE) ||
        grepl(norm_columns[i], norm_celltype, fixed = TRUE)) {
      return(column_names[i])
    }
  }
  
  return(NA)
}

# -----------------------------------------------------------------------------
# Helper Function: Reorder columns based on spot names
# -----------------------------------------------------------------------------
reorder_columns_by_spots <- function(proportions_df) {
  spot_names <- rownames(proportions_df)
  col_names <- colnames(proportions_df)
  
  cat("\n=== Reordering columns to match spot order ===\n")
  
  ordered_cols <- c()
  matched_spots <- c()
  
  for (spot in spot_names) {
    celltype <- extract_celltype_from_spot(spot)
    matching_col <- find_matching_column(celltype, col_names)
    
    if (!is.na(matching_col) && !(matching_col %in% ordered_cols)) {
      ordered_cols <- c(ordered_cols, matching_col)
      matched_spots <- c(matched_spots, spot)
      cat("  ", spot, " -> ", celltype, " -> ", matching_col, "\n")
    }
  }
  
  remaining_cols <- setdiff(col_names, ordered_cols)
  if (length(remaining_cols) > 0) {
    cat("\n  Unmatched columns (appended at end):\n")
    for (col in remaining_cols) {
      cat("    ", col, "\n")
    }
    ordered_cols <- c(ordered_cols, remaining_cols)
  }
  
  proportions_reordered <- proportions_df[, ordered_cols, drop = FALSE]
  
  cat("\n  Final column order:\n")
  for (i in seq_along(ordered_cols)) {
    cat("    ", i, ": ", ordered_cols[i], "\n")
  }
  
  return(proportions_reordered)
}

# -----------------------------------------------------------------------------
# Helper Functions for Coordinate Extraction (NEW)
# -----------------------------------------------------------------------------

#' Extract coordinates from count matrix
#' Supports: row/col columns OR spot name patterns like spot_5_10
extract_coordinates_from_counts <- function(counts_df, spot_names) {
  cat("\n=== Attempting to extract coordinates from count matrix ===\n")
  
  # Method 1: Check for row/col columns in the data
  col_names_lower <- tolower(colnames(counts_df))
  
  has_row <- any(col_names_lower %in% c("row", "rows"))
  has_col <- any(col_names_lower %in% c("col", "cols", "column", "columns"))
  
  if (has_row && has_col) {
    cat("  Found row/col columns in data\n")
    
    row_idx <- which(col_names_lower %in% c("row", "rows"))[1]
    col_idx <- which(col_names_lower %in% c("col", "cols", "column", "columns"))[1]
    
    coords <- data.frame(
      barcode = spot_names,
      x = as.numeric(counts_df[[row_idx]]),
      y = as.numeric(counts_df[[col_idx]]),
      stringsAsFactors = FALSE
    )
    
    # Remove row/col columns from counts
    cols_to_remove <- c(row_idx, col_idx)
    counts_df <- counts_df[, -cols_to_remove, drop = FALSE]
    
    cat("  Extracted coordinates for", nrow(coords), "spots\n")
    return(list(coords = coords, counts = counts_df, method = "row_col"))
  }
  
  # Method 2: Try to parse coordinates from spot names (e.g., "spot_5_10")
  cat("  Trying to extract coordinates from spot names...\n")
  
  # Pattern: spot_X_Y or similar
  pattern <- "^(?:spot_?)?(\\d+)[_x](\\d+)$"
  
  coords_list <- lapply(spot_names, function(name) {
    # Try multiple patterns
    match <- regmatches(name, regexec("(\\d+)[_x](\\d+)", name, perl = TRUE))[[1]]
    if (length(match) == 3) {
      return(c(as.numeric(match[2]), as.numeric(match[3])))
    }
    return(c(NA, NA))
  })
  
  coords_matrix <- do.call(rbind, coords_list)
  
  if (sum(!is.na(coords_matrix[, 1])) > nrow(coords_matrix) * 0.5) {
    coords <- data.frame(
      barcode = spot_names,
      x = coords_matrix[, 1],
      y = coords_matrix[, 2],
      stringsAsFactors = FALSE
    )
    
    # Remove spots without valid coordinates
    valid_coords <- !is.na(coords$x)
    coords <- coords[valid_coords, ]
    
    cat("  Extracted coordinates from", nrow(coords), "spot names\n")
    return(list(coords = coords, counts = counts_df, method = "spot_names"))
  }
  
  cat("  Could not extract coordinates from count matrix\n")
  return(list(coords = NULL, counts = counts_df, method = "none"))
}

#' Detect data type based on file content
detect_data_type <- function(counts_df, spot_names) {
  # Check for row/col columns
  col_names_lower <- tolower(colnames(counts_df))
  has_row_col <- any(col_names_lower %in% c("row", "rows")) && 
                 any(col_names_lower %in% c("col", "cols", "column", "columns"))
  
  # Check for coordinate patterns in spot names
  coord_pattern <- sum(grepl("\\d+[_x]\\d+", spot_names)) > length(spot_names) * 0.5
  
  # Check for Visium-style barcodes
  visium_pattern <- sum(grepl("^[ACGT]+-1$", spot_names)) > length(spot_names) * 0.5
  
  if (has_row_col || coord_pattern) {
    return("simulated")
  } else if (visium_pattern) {
    return("visium")
  } else {
    return("unknown")
  }
}

# -----------------------------------------------------------------------------
# Helper Functions for Spatial Visualization
# -----------------------------------------------------------------------------

#' Calculate optimal hexagon radius for perfect tessellation (no overlap, no gaps)
#' 
#' For pointy-top hexagons: nearest_neighbor_distance = sqrt(3) * radius
#' For flat-top hexagons: nearest_neighbor_distance = 2 * radius
#'
#' @param coords Data frame with x, y columns
#' @param orientation Rotation angle (30 = pointy-top, 0 = flat-top)
#' @return Radius for perfect tessellation
calculate_hex_radius <- function(coords, orientation = 30) {
  if (nrow(coords) < 2) {
    return(100)  # Default radius
  }
  
  # Get x and y coordinates
  xy_matrix <- as.matrix(coords[, c("x", "y")])
  
  # Find nearest neighbor distances
  nn <- get.knn(xy_matrix, k = 1)
  median_dist <- median(nn$nn.dist)
  
  # Calculate radius based on hexagon orientation
  if (orientation == 30) {
    # Pointy-top: nearest_neighbor = sqrt(3) * radius
    radius <- median_dist / sqrt(3)
  } else {
    # Flat-top: nearest_neighbor = 2 * radius (for horizontal neighbors)
    # But diagonal neighbors are at sqrt(3) * radius
    # Use sqrt(3) for safety
    radius <- median_dist / sqrt(3)
  }
  
  # NO multiplier for exact tessellation
  # Use 1.001 only if you see anti-aliasing gaps in PNG output
  radius <- radius * 1.0
  
  return(radius)
}

#' Create hexagon vertices for a single point
#' @param rotation Rotation angle in degrees (0 = flat top, 30 = pointy top)
create_hexagon <- function(x, y, radius, rotation = 0) {
  # Convert rotation from degrees to radians
  rot_rad <- rotation * pi / 180
  angles <- seq(0, 2 * pi, length.out = 7)[1:6] + rot_rad
  data.frame(
    x = x + radius * cos(angles),
    y = y + radius * sin(angles)
  )
}

#' Create hexagon polygon data for all spots
create_all_hexagons <- function(coords, radius, rotation = 0) {
  hex_list <- lapply(1:nrow(coords), function(i) {
    hex <- create_hexagon(coords$x[i], coords$y[i], radius, rotation)
    hex$id <- coords$barcode[i]
    hex$spot_idx <- i
    hex
  })
  do.call(rbind, hex_list)
}

#' Create hexagon polygon data for background (unmatched) spots
create_background_hexagons <- function(coords_full, matched_mask, radius, rotation = 0) {
  # Get unmatched spots
  unmatched_idx <- which(!matched_mask)
  
  if (length(unmatched_idx) == 0) {
    return(NULL)
  }
  
  unmatched_coords <- coords_full[unmatched_idx, ]
  
  hex_list <- lapply(1:nrow(unmatched_coords), function(i) {
    hex <- create_hexagon(unmatched_coords$x[i], unmatched_coords$y[i], radius, rotation)
    hex$id <- unmatched_coords$barcode[i]
    hex$spot_idx <- unmatched_idx[i]
    hex
  })
  do.call(rbind, hex_list)
}

#' Plot spatial intensity maps - Grid of cell type proportion heatmaps
#' Now supports showing ALL spots including unmatched ones as grey background
plot_spatial_intensity_maps <- function(proportions, coords, output_path, 
                                         hex_radius = NULL,
                                         coords_full = NULL, matched_mask = NULL,
                                         rotation = 0) {
  
  # Merge proportions with coordinates - PRESERVE COUNT MATRIX ORDER
  prop_df <- as.data.frame(proportions)
  prop_df$barcode <- rownames(prop_df)
  
  # Rotate coordinates 90 degrees clockwise: (x, y) -> (y, -x)
  # IMPORTANT: Use the same max_x reference for both matched and full coords
  
  # Determine max_x from full coords if available (for consistent rotation)
  if (!is.null(coords_full)) {
    max_x_ref <- max(coords_full$x, na.rm = TRUE)
    
    # Rotate full coords first
    coords_full_rotated <- coords_full
    coords_full_rotated$x <- coords_full$y
    coords_full_rotated$y <- max_x_ref - coords_full$x
    coords_full <- coords_full_rotated
    
    # Rotate matched coords using SAME reference
    coords_rotated <- coords
    coords_rotated$x <- coords$y
    coords_rotated$y <- max_x_ref - coords$x
    coords <- coords_rotated
  } else {
    # No full coords - rotate matched coords only
    max_x_ref <- max(coords$x, na.rm = TRUE)
    coords_rotated <- coords
    coords_rotated$x <- coords$y
    coords_rotated$y <- max_x_ref - coords$x
    coords <- coords_rotated
  }
  
  # Use match() to preserve the order from proportions (count matrix order)
  coord_idx <- match(prop_df$barcode, coords$barcode)
  valid_idx <- !is.na(coord_idx)
  
  merged <- data.frame(
    barcode = prop_df$barcode[valid_idx],
    x = coords$x[coord_idx[valid_idx]],
    y = coords$y[coord_idx[valid_idx]]
  )
  # Add proportion columns
  for (col in setdiff(colnames(prop_df), "barcode")) {
    merged[[col]] <- prop_df[[col]][valid_idx]
  }
  
  if (nrow(merged) == 0) {
    warning("No matching barcodes between proportions and coordinates!")
    return(NULL)
  }
  
  # Calculate hex radius from ALL coordinates if available (for consistent sizing)
  if (is.null(hex_radius)) {
    if (!is.null(coords_full)) {
      hex_radius <- calculate_hex_radius(coords_full, orientation = rotation)
    } else {
      hex_radius <- calculate_hex_radius(merged, orientation = rotation)
    }
  }
  
  # Get cell type columns
  cell_types <- setdiff(colnames(prop_df), "barcode")
  
  # Create hexagon polygons for matched spots
  hex_polys <- create_all_hexagons(merged, hex_radius, rotation = rotation)
  
  # Create background hexagons for unmatched spots
  bg_hex_polys <- NULL
  has_unmatched <- FALSE
  if (!is.null(coords_full) && !is.null(matched_mask)) {
    bg_hex_polys <- create_background_hexagons(coords_full, matched_mask, hex_radius, rotation = rotation)
    has_unmatched <- !is.null(bg_hex_polys) && nrow(bg_hex_polys) > 0
  }
  
  # Calculate axis limits from ALL coordinates
  if (!is.null(coords_full)) {
    x_range <- range(coords_full$x)
    y_range <- range(coords_full$y)
    x_expand <- diff(x_range) * 0.05
    y_expand <- diff(y_range) * 0.05
    xlim <- c(x_range[1] - x_expand, x_range[2] + x_expand)
    ylim <- c(y_range[1] - y_expand, y_range[2] + y_expand)
  } else {
    xlim <- NULL
    ylim <- NULL
  }
  
  # Create a plot for each cell type
  plot_list <- lapply(cell_types, function(ct) {
    # Add proportion values to hexagon data
    hex_data <- hex_polys
    hex_data$value <- merged[[ct]][hex_data$spot_idx]
    
    p <- ggplot()
    
    # Add background (unmatched) spots first
    if (has_unmatched) {
      p <- p + geom_polygon(data = bg_hex_polys, 
                            aes(x = x, y = y, group = id),
                            fill = "grey80", color = NA, 
                            linewidth = 0, alpha = 0.5)
    }
    
    # Add matched spots with color
    p <- p + geom_polygon(data = hex_data, 
                          aes(x = x, y = y, group = id, fill = value),
                          color = NA, linewidth = 0) +
      scale_fill_viridis(option = "plasma", 
                         limits = c(max(0, min(hex_data$value, na.rm = TRUE) - 1e-8),
                                    max(hex_data$value, na.rm = TRUE) + 1e-8),
                         name = "Proportion") +
      coord_fixed(xlim = xlim, ylim = ylim) +
      theme_minimal() +
      theme(
        panel.grid = element_blank(),
        axis.text = element_blank(),
        axis.title = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 10),
        legend.position = "right",
        legend.key.size = unit(0.5, "cm")
      ) +
      ggtitle(ct)
    
    return(p)
  })
  
  # Arrange in grid
  n_types <- length(cell_types)
  n_cols <- min(4, n_types)
  n_rows <- ceiling(n_types / n_cols)
  
  # Save plot
  png(output_path, width = n_cols * 5, height = n_rows * 5, units = "in", res = 1000)
  grid.arrange(grobs = plot_list, ncol = n_cols)
  dev.off()
  
  cat(sprintf("Saved intensity maps to: %s\n", output_path))
  
  if (has_unmatched) {
    n_unmatched <- sum(!matched_mask)
    cat(sprintf("  (Showing %d matched spots + %d unmatched spots as grey background)\n",
                nrow(merged), n_unmatched))
  }
}

#' Plot spatial dominant type map
#' MATCHING PYTHON SCRIPT STYLE:
#' - Tab20 colormap
#' - Matched fonts/sizes
#' - Background spots as LightGrey
#' - Legend format including "No count data"
plot_spatial_dominant_type <- function(proportions, coords, output_path, 
                                        hex_radius = NULL,
                                        coords_full = NULL, matched_mask = NULL,
                                        rotation = 0) {
  
  # Merge proportions with coordinates - PRESERVE COUNT MATRIX ORDER
  prop_df <- as.data.frame(proportions)
  prop_df$barcode <- rownames(prop_df)
  
  # Rotate coordinates 90 degrees clockwise: (x, y) -> (y, -x)
  # Determine max_x from full coords if available (for consistent rotation)
  if (!is.null(coords_full)) {
    max_x_ref <- max(coords_full$x, na.rm = TRUE)
    
    # Rotate full coords first
    coords_full_rotated <- coords_full
    coords_full_rotated$x <- coords_full$y
    coords_full_rotated$y <- max_x_ref - coords_full$x
    coords_full <- coords_full_rotated
    
    # Rotate matched coords using SAME reference
    coords_rotated <- coords
    coords_rotated$x <- coords$y
    coords_rotated$y <- max_x_ref - coords$x
    coords <- coords_rotated
  } else {
    # No full coords - rotate matched coords only
    max_x_ref <- max(coords$x, na.rm = TRUE)
    coords_rotated <- coords
    coords_rotated$x <- coords$y
    coords_rotated$y <- max_x_ref - coords$x
    coords <- coords_rotated
  }
  
  # Use match() to preserve the order from proportions (count matrix order)
  coord_idx <- match(prop_df$barcode, coords$barcode)
  valid_idx <- !is.na(coord_idx)
  
  merged <- data.frame(
    barcode = prop_df$barcode[valid_idx],
    x = coords$x[coord_idx[valid_idx]],
    y = coords$y[coord_idx[valid_idx]]
  )
  # Add proportion columns
  for (col in setdiff(colnames(prop_df), "barcode")) {
    merged[[col]] <- prop_df[[col]][valid_idx]
  }
  
  if (nrow(merged) == 0) {
    warning("No matching barcodes between proportions and coordinates!")
    return(NULL)
  }
  
  # Calculate hex radius from ALL coordinates if available
  if (is.null(hex_radius)) {
    if (!is.null(coords_full)) {
      hex_radius <- calculate_hex_radius(coords_full, orientation = rotation)
    } else {
      hex_radius <- calculate_hex_radius(merged, orientation = rotation)
    }
  }
  
  # Get cell type columns and find dominant type
  cell_types <- setdiff(colnames(prop_df), "barcode")
  prop_matrix <- as.matrix(merged[, cell_types])
  merged$dominant_type <- cell_types[apply(prop_matrix, 1, which.max)]
  merged$max_proportion <- apply(prop_matrix, 1, max)
  
  # Create hexagon polygons for matched spots
  hex_polys <- create_all_hexagons(merged, hex_radius, rotation = rotation)
  hex_polys$dominant_type <- merged$dominant_type[hex_polys$spot_idx]
  hex_polys$max_proportion <- merged$max_proportion[hex_polys$spot_idx]
  
  # Create background hexagons for unmatched spots
  bg_hex_polys <- NULL
  has_unmatched <- FALSE
  if (!is.null(coords_full) && !is.null(matched_mask)) {
    bg_hex_polys <- create_background_hexagons(coords_full, matched_mask, hex_radius, rotation = rotation)
    has_unmatched <- !is.null(bg_hex_polys) && nrow(bg_hex_polys) > 0
  }
  
  # Calculate axis limits from ALL coordinates
  if (!is.null(coords_full)) {
    x_range <- range(coords_full$x)
    y_range <- range(coords_full$y)
    x_expand <- diff(x_range) * 0.05
    y_expand <- diff(y_range) * 0.05
    xlim <- c(x_range[1] - x_expand, x_range[2] + x_expand)
    ylim <- c(y_range[1] - y_expand, y_range[2] + y_expand)
  } else {
    xlim <- NULL
    ylim <- NULL
  }
  
  # ===========================================================================
  # COLOR PALETTE ASSIGNMENT
  # ===========================================================================
  # 1. Standard Tab20 Palette (Matched to Python/Matplotlib)
  tab20_colors <- c(
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5"
  )
  
    # 2. Define Explicit Mapping for Known Cell Types
  specific_mapping <- c(
    "Astro AQP4 SLC1A2"       = "#1f77b4", # Dark Blue
    "Endo CLDN5 SLC7A5"       = "#2ca02c", # Dark Green
    "L2-3 CUX2 ACVR1C THSD7A" = "#9467bd", # Dark Purple
    "Micro P2RY12 APBB1IP"    = "#e377c2", # Pink
    "OPC PDGFRA PCDH15"       = "#bcbd22", # Olive Green
    "Oligo MOG OPALIN"        = "#9edae5"  # Light Cyan
  )

  #specific_mapping <- c(
  #  "Astro GFAP AQP1"      = "#1f77b4", # Dark Blue
  #  "Endo CLDN5 IL1R1"     = "#ff7f0e", # Dark Orange
  #  "Micro P2RY12 APBB1IP" = "#98df8a", # Light Green
  #  "OPC PDGFRA PCDH15"    = "#9467bd", # Dark Purple
  #  "Oligo MOG GSN"        = "#c49c94", # Light Brown
  #  "Oligo MOG OPALIN"     = "#7f7f7f", # Dark Gray
  #  "PC P2RY14 GRM8"       = "#dbdb8d", # Light Olive
  #  "T SKAP1 CD247"        = "#9edae5"  # Light Cyan
  #)

  # 2. Define Explicit Mapping for Known Cell Types
  #specific_mapping <- c(
  #  "Brain vascular cells"      = "#1F77B4", # 0: Blue
  #  "Cortex immune cells"      = "#ff7f0e", # 0: Blue
  #  "Dorsal midbrain OPC"       = "#97e188", # 1: Green
  #  "Dorsal midbrain glioblast" = "#9467bd", # 2: Purple
  #  "Pons OPC"   = "#c49c94", # 3: Pink
  #  "Pons neural crest cells"   = "#7f7f7f", # 3: Pink
  #  "Subcortex neuron"          = "#bcbd22", # 4: Olive
  #  "Thalamus glioblast"        = "#9edae5"  # 5: Light Cyan
  #)

  
  #specific_mapping <- c(
  #  "Astrocyte"      = "#1F77B4", # 0: Blue
  #  "Microglia-PVM"       = "#98df8a", # 1: Green
  #  "Oligodendrocyte" = "#8c564b", # 2: Purple
  #  "OPC"   = "#7f7f7f", # 3: Pink
  #  "Pax6"          = "#9edae5" # 4: Olive
  #)
  # 2. Define Explicit Mapping for Known Cell Types (Matches Simulation Ground Truth)
  #specific_mapping <- c(
  #  "093 RT-ZI Gnb3 Gaba"                    = "#1f77b4", # Dark Blue (Duplicate)
  #  "101 ZI Pax6 Gaba"                       = "#1f77b4", # Dark Blue (Duplicate)
  #  "145 MH Tac2 Glut"                       = "#aec7e8", # Light Blue
  #  "146 LH Pou4f1 Sox1 Glut"                = "#ff7f0e", # Orange
  #  "147 AD Serpinb7 Glut"                   = "#ffbb78", # Light Orange
  #  "148 AV Col27a1 Glut"                    = "#2ca02c", # Green
  #  "149 PVT-PT Ntrk1 Glut"                  = "#98df8a", # Light Green
  #  "150 CM-IAD-CL-PCN Sema5b Glut"          = "#d62728", # Red
  #  "151 TH Prkcd Grin2c Glut"               = "#ff9896", # Light Red
  #  "152 RE-Xi Nox4 Glut"                    = "#9467bd", # Purple
  #  "154 PF Fzd5 Glut"                       = "#c5b0d5", # Light Purple
  #  "163 APN C1ql2 Glut"                     = "#8c564b", # Brown
  #  "164 APN C1ql4 Glut"                     = "#c49c94", # Light Brown
  #  "168 SPA-SPFm-SPFp-POL-PIL-PoT Sp9 Glut" = "#e377c2", # Pink
  #  "202 PRT Tcf7l2 Gaba"                    = "#f7b6d2", # Light Pink
  #  "213 SCsg Gabrr2 Gaba"                   = "#7f7f7f", # Gray
  #  "264 PRNc Otp Gly-Gaba"                  = "#c7c7c7", # Light Gray
  #  "318 Astro-NT NN"                        = "#bcbd22", # Olive
  #  "326 OPC NN"                             = "#dbdb8d", # Light Olive
  #  "327 Oligo NN"                           = "#17becf", # Cyan
  #  "333 Endo NN"                            = "#9edae5", # Light Cyan (Duplicate)
  #  "334 Microglia NN"                       = "#9edae5"  # Light Cyan (Duplicate)
  #)
  # 3. Determine types present in the dataset (alphabetical order)
  present_types <- sort(cell_types)
  
  # 4. Construct the final color vector
  # Initialize with NA
  type_colors <- setNames(rep(NA, length(present_types)), present_types)
  
  # First pass: Assign specific colors if cell types exist in data
  for (ct in names(specific_mapping)) {
    if (ct %in% present_types) {
      type_colors[ct] <- specific_mapping[ct]
    }
  }
  
  # Second pass: Assign remaining types (if any) using unused Tab20 colors
  remaining_types <- names(type_colors)[is.na(type_colors)]
  if (length(remaining_types) > 0) {
    # Find hex codes already used
    used_hex <- as.character(na.omit(type_colors))
    # Find available codes from Tab20
    available_hex <- setdiff(tab20_colors, used_hex)
    
    # If not enough colors, fallback to colorRamp
    if (length(available_hex) < length(remaining_types)) {
      available_hex <- colorRampPalette(tab20_colors)(length(remaining_types))
    }
    
    type_colors[remaining_types] <- available_hex[1:length(remaining_types)]
  }
  
  # Prepare data for plotting
  # We combine matched and unmatched into one dataframe to control the legend properly
  plot_data <- hex_polys
  plot_data$fill_group <- plot_data$dominant_type
  plot_data$alpha_val <- 1.0
  
  legend_breaks <- present_types
  
  if (has_unmatched) {
    bg_data <- bg_hex_polys
    bg_data$dominant_type <- "No count data" 
    bg_data$max_proportion <- 0
    bg_data$fill_group <- "No count data"
    bg_data$alpha_val <- 0.5
    
    # Ensure columns match for rbind
    common_cols <- intersect(colnames(plot_data), colnames(bg_data))
    plot_data <- rbind(plot_data[, common_cols], bg_data[, common_cols])
    
    # Add grey to color map for "No count data"
    type_colors <- c(type_colors, "No count data" = "lightgrey")
    
    # Add "No count data" to breaks to ensure it appears at bottom of legend
    legend_breaks <- c(legend_breaks, "No count data")
  }
  
  # Convert fill_group to factor to respect legend order
  plot_data$fill_group <- factor(plot_data$fill_group, levels = legend_breaks)

  # Create plot
  p <- ggplot(plot_data, aes(x = x, y = y, group = id, fill = fill_group, alpha = alpha_val)) +
    geom_polygon(color = NA, linewidth = 0) +
    scale_fill_manual(
      values = type_colors, 
      name = "Cell Type",
      breaks = legend_breaks
    ) +
    scale_alpha_identity() + # Uses the values in alpha_val column directly
    coord_fixed(xlim = xlim, ylim = ylim) +
    theme_void() + # Removes axes, ticks, grid
    ggtitle("Dominant Cell Type per Spot") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold", margin = margin(b = 10)),
      legend.position = "right",
      legend.title = element_text(size = 15, face = "bold"),
      legend.text = element_text(size = 14),
      legend.key.size = unit(0.8, "cm"),
      plot.margin = margin(10, 10, 10, 10)
    )
  
  # Save plot (Match Python output: 12x10 inches, 300 DPI)
  ggsave(output_path, p, width = 12, height = 10, dpi = 300)
  
  cat(sprintf("Saved dominant type map to: %s\n", output_path))
  
  if (has_unmatched) {
    n_unmatched <- sum(!matched_mask)
    cat(sprintf("  (Showing %d matched spots + %d unmatched spots as grey background)\n",
                nrow(merged), n_unmatched))
  }
}

# -----------------------------------------------------------------------------
# Plot Co-occurrence Heatmap
# -----------------------------------------------------------------------------
#' Plot cell type co-occurrence (correlation) heatmap
#' Shows Pearson correlation between cell type proportions across spots
plot_cooccurrence_heatmap <- function(proportions, output_path) {
  
  # Get cell type columns only (exclude any metadata columns)
  prop_matrix <- as.matrix(proportions)
  
  # Compute Pearson correlation between cell types
  corr_matrix <- cor(prop_matrix, method = "pearson")
  
  # Handle NA values (can occur if a cell type has zero variance)
  corr_matrix[is.na(corr_matrix)] <- 0
  
  # Create heatmap
  png(output_path, width = 10, height = 8, units = "in", res = 300)
  
  pheatmap(
    corr_matrix,
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    display_numbers = TRUE,
    number_format = "%.2f",
    number_color = "black",
    fontsize_number = 8,
    color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),  # RdBu_r equivalent
    breaks = seq(-1, 1, length.out = 101),  # Center at 0
    main = "Cell Type Co-occurrence (Correlation)",
    fontsize = 10,
    fontsize_row = 9,
    fontsize_col = 9,
    border_color = "white",
    cellwidth = 25,
    cellheight = 25
  )
  
  dev.off()
  
  cat(sprintf("Saved co-occurrence heatmap to: %s\n", output_path))
}

# -----------------------------------------------------------------------------
# Plot Correlation with Ground Truth (NEW)
# -----------------------------------------------------------------------------
plot_correlation_with_ground_truth <- function(predicted, ground_truth, output_path) {
  cat("\n=== Generating correlation plot with ground truth ===\n")
  
  # Align spots
  common_spots <- intersect(rownames(predicted), rownames(ground_truth))
  if (length(common_spots) == 0) {
    warning("No common spots between predictions and ground truth!")
    return(NULL)
  }
  
  cat("  Common spots:", length(common_spots), "\n")
  
  pred_aligned <- predicted[common_spots, , drop = FALSE]
  gt_aligned <- ground_truth[common_spots, , drop = FALSE]
  
  # Align cell types
  common_types <- intersect(colnames(pred_aligned), colnames(gt_aligned))
  if (length(common_types) == 0) {
    # Try matching with normalized names
    pred_cols_norm <- sapply(colnames(pred_aligned), normalize_name)
    gt_cols_norm <- sapply(colnames(gt_aligned), normalize_name)
    
    matches <- list()
    for (i in seq_along(pred_cols_norm)) {
      for (j in seq_along(gt_cols_norm)) {
        if (pred_cols_norm[i] == gt_cols_norm[j]) {
          matches[[colnames(pred_aligned)[i]]] <- colnames(gt_aligned)[j]
        }
      }
    }
    
    if (length(matches) == 0) {
      warning("No matching cell types between predictions and ground truth!")
      return(NULL)
    }
    
    common_types <- names(matches)
    gt_matched <- gt_aligned[, unlist(matches), drop = FALSE]
    colnames(gt_matched) <- common_types
    gt_aligned <- gt_matched
  }
  
  pred_aligned <- pred_aligned[, common_types, drop = FALSE]
  gt_aligned <- gt_aligned[, common_types, drop = FALSE]
  
  cat("  Common cell types:", length(common_types), "\n")
  
  # Flatten for scatter plot
  pred_vec <- as.vector(as.matrix(pred_aligned))
  gt_vec <- as.vector(as.matrix(gt_aligned))
  
  # Calculate correlation
  cor_val <- cor(pred_vec, gt_vec, method = "pearson")
  rmse_val <- sqrt(mean((pred_vec - gt_vec)^2))
  
  cat(sprintf("  Pearson correlation: %.4f\n", cor_val))
  cat(sprintf("  RMSE: %.4f\n", rmse_val))
  
  # Create scatter plot
  plot_df <- data.frame(
    ground_truth = gt_vec,
    predicted = pred_vec,
    cell_type = rep(common_types, each = nrow(pred_aligned))
  )
  
  p <- ggplot(plot_df, aes(x = ground_truth, y = predicted, color = cell_type)) +
    geom_point(alpha = 0.6, size = 2) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    theme_minimal() +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5)
    ) +
    labs(
      title = sprintf("Predicted vs Ground Truth (r=%.3f, RMSE=%.3f)", cor_val, rmse_val),
      x = "Ground Truth Proportion",
      y = "Predicted Proportion",
      color = "Cell Type"
    ) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1))
  
  ggsave(output_path, p, width = 10, height = 8, dpi = 300)
  cat(sprintf("  Saved correlation plot to: %s\n", output_path))
  
  # Return metrics
  return(list(correlation = cor_val, rmse = rmse_val))
}
# =============================================================================
# SMART TRANSPOSE FUNCTION
# =============================================================================
#' Smart auto-transpose that uses gene name matching instead of dimension heuristics
#' @param mat Matrix to potentially transpose
#' @param reference_genes Optional vector of known gene names (from SC data)
#' @param data_type String describing data type for logging ("SC" or "ST")
#' @return Properly oriented matrix (genes x samples)
smart_transpose <- function(mat, reference_genes = NULL, data_type = "Data") {
  cat(sprintf("\n  %s format detection:\n", data_type))
  cat(sprintf("    Input dimensions: %d x %d\n", nrow(mat), ncol(mat)))
  
  row_names <- rownames(mat)
  col_names <- colnames(mat)
  
  # Method 1: Use reference genes if provided (most reliable)
  if (!is.null(reference_genes) && length(reference_genes) > 0) {
    row_gene_overlap <- sum(row_names %in% reference_genes)
    col_gene_overlap <- sum(col_names %in% reference_genes)
    
    cat(sprintf("    Reference gene overlap - Rows: %d, Cols: %d\n", 
                row_gene_overlap, col_gene_overlap))
    
    if (row_gene_overlap > col_gene_overlap && row_gene_overlap > 100) {
      cat("    -> Rows are genes (correct orientation)\n")
      return(mat)
    } else if (col_gene_overlap > row_gene_overlap && col_gene_overlap > 100) {
      cat("    -> Columns are genes, transposing to genes x samples...\n")
      return(t(mat))
    }
  }
  
  # Method 2: Check for spot/barcode patterns in row/column names
  spot_patterns <- c(
    "^spot[_-]?\\d+",           # spot_0, spot-1, spot0
    "^\\d+[_x]\\d+$",           # 0_0, 5x10
    "^[ACGT]{6,}-1$",           # Visium barcodes
    "^Batch_\\d+_"              # Simulated data pattern
  )
  
  row_spot_matches <- sum(sapply(spot_patterns, function(p) sum(grepl(p, row_names, ignore.case = TRUE))))
  col_spot_matches <- sum(sapply(spot_patterns, function(p) sum(grepl(p, col_names, ignore.case = TRUE))))
  
  cat(sprintf("    Spot pattern matches - Rows: %d, Cols: %d\n", 
              row_spot_matches, col_spot_matches))
  
  if (row_spot_matches > col_spot_matches && row_spot_matches > length(row_names) * 0.3) {
    cat("    -> Rows are spots, transposing to genes x spots...\n")
    return(t(mat))
  } else if (col_spot_matches > row_spot_matches && col_spot_matches > length(col_names) * 0.3) {
    cat("    -> Columns are spots (correct orientation)\n")
    return(mat)
  }
  
  # Method 3: Check for gene name patterns
  gene_patterns <- c(
    "^[A-Z][A-Z0-9]*$",         # Standard gene symbols (A1BG, TP53)
    "^[A-Z]+\\d+$",             # Gene with numbers (ABC1)
    "^MT-",                      # Mitochondrial genes
    "^RP[SL]\\d+",              # Ribosomal genes
    "^ENSG\\d+",                # Ensembl IDs
    "^ENSMUSG\\d+"              # Mouse Ensembl IDs
  )
  
  row_gene_matches <- sum(sapply(gene_patterns, function(p) sum(grepl(p, row_names))))
  col_gene_matches <- sum(sapply(gene_patterns, function(p) sum(grepl(p, col_names))))
  
  cat(sprintf("    Gene pattern matches - Rows: %d, Cols: %d\n", 
              row_gene_matches, col_gene_matches))
  
  if (row_gene_matches > col_gene_matches && row_gene_matches > length(row_names) * 0.3) {
    cat("    -> Rows are genes (correct orientation)\n")
    return(mat)
  } else if (col_gene_matches > row_gene_matches && col_gene_matches > length(col_names) * 0.3) {
    cat("    -> Columns are genes, transposing to genes x samples...\n")
    return(t(mat))
  }
  
  # Method 4: Fallback - assume more features (genes) than samples
  cat("    -> Using dimension heuristic (more rows = genes)\n")
  if (nrow(mat) < ncol(mat)) {
    cat("    -> Transposing (assuming columns are genes)...\n")
    return(t(mat))
  }
  
  cat("    -> Keeping original orientation\n")
  return(mat)
}
# -----------------------------------------------------------------------------
# Load Data Functions
# -----------------------------------------------------------------------------
load_counts <- function(filepath, transpose = FALSE, reference_genes = NULL, data_type = "Data") {
  cat("Reading:", filepath, "\n")
  
  dt <- fread(filepath, header = TRUE)
  df <- as.data.frame(dt)
  
  rownames(df) <- df[[1]]
  df <- df[, -1, drop = FALSE]
  
  mat <- as.matrix(df)
  
  # Manual transpose takes priority
  if (transpose) {
    cat("  Manual transpose requested...\n")
    mat <- t(mat)
  } else {
    # Use smart transpose for auto-detection
    mat <- smart_transpose(mat, reference_genes = reference_genes, data_type = data_type)
  }
  
  mode(mat) <- "numeric"
  
  cat("  Final dimensions:", nrow(mat), "genes x", ncol(mat), "cells/spots\n")
  return(mat)
}

load_counts_with_coords <- function(filepath, transpose = FALSE, reference_genes = NULL) {
  cat("Reading:", filepath, "\n")
  
  dt <- fread(filepath, header = TRUE)
  df <- as.data.frame(dt)
  
  # Store first column as row names
  row_ids <- df[[1]]
  df <- df[, -1, drop = FALSE]
  
  # Check for coordinate columns BEFORE transposing
  col_names_lower <- tolower(colnames(df))
  coord_cols <- c()
  coords <- NULL
  
  # Look for row/col columns
  row_col_idx <- which(col_names_lower %in% c("row", "rows"))
  col_col_idx <- which(col_names_lower %in% c("col", "cols", "column", "columns"))
  
  if (length(row_col_idx) > 0 && length(col_col_idx) > 0) {
    cat("  Found embedded coordinates (row/col columns)\n")
    
    coords <- data.frame(
      barcode = row_ids,
      x = as.numeric(df[[row_col_idx[1]]]),
      y = as.numeric(df[[col_col_idx[1]]]),
      stringsAsFactors = FALSE
    )
    
    # Remove coordinate columns
    coord_cols <- c(row_col_idx[1], col_col_idx[1])
    df <- df[, -coord_cols, drop = FALSE]
    
    cat("  Extracted coordinates for", nrow(coords), "spots\n")
  }
  
  rownames(df) <- row_ids
  mat <- as.matrix(df)
  
  # Manual transpose takes priority
  if (transpose) {
    cat("  Manual transpose requested...\n")
    mat <- t(mat)
  } else {
    # Use smart transpose for auto-detection
    mat <- smart_transpose(mat, reference_genes = reference_genes, data_type = "ST")
  }
  
  mode(mat) <- "numeric"
  
  # Update coordinates to match transposed data if needed
  if (!is.null(coords)) {
    # Coordinates should match column names (spots)
    spot_names <- colnames(mat)
    coords <- coords[coords$barcode %in% spot_names, ]
  }
  
  # If no row/col columns, try to extract from column names (spot names)
  if (is.null(coords)) {
    spot_names <- colnames(mat)
    
    # Try pattern matching for spot_X_Y format
    coords_list <- lapply(spot_names, function(name) {
      match <- regmatches(name, regexec("(\\d+)[_x](\\d+)", name, perl = TRUE))[[1]]
      if (length(match) == 3) {
        return(c(as.numeric(match[2]), as.numeric(match[3])))
      }
      return(c(NA, NA))
    })
    
    coords_matrix <- do.call(rbind, coords_list)
    
    if (sum(!is.na(coords_matrix[, 1])) > length(spot_names) * 0.5) {
      coords <- data.frame(
        barcode = spot_names,
        x = coords_matrix[, 1],
        y = coords_matrix[, 2],
        stringsAsFactors = FALSE
      )
      cat("  Extracted coordinates from spot names for", sum(!is.na(coords$x)), "spots\n")
    }
  }
  
  cat("  Final dimensions:", nrow(mat), "genes x", ncol(mat), "spots\n")
  
  return(list(counts = mat, coords = coords))
}

load_labels <- function(filepath) {
  cat("Reading labels:", filepath, "\n")
  
  dt <- fread(filepath, header = TRUE)
  df <- as.data.frame(dt)
  
  if ("cell_id" %in% colnames(df)) {
    cell_col <- "cell_id"
  } else if ("barcode" %in% colnames(df)) {
    cell_col <- "barcode"
  } else {
    cell_col <- colnames(df)[1]
  }
  
  if ("cell_type" %in% colnames(df)) {
    type_col <- "cell_type"
  } else if ("CellType" %in% colnames(df)) {
    type_col <- "CellType"
  } else if ("label" %in% colnames(df)) {
    type_col <- "label"
  } else {
    type_col <- colnames(df)[2]
  }
  
  cat("  Using cell ID column:", cell_col, "\n")
  cat("  Using cell type column:", type_col, "\n")
  
  labels <- as.character(df[[type_col]])
  names(labels) <- df[[cell_col]]
  
  cat("  Number of cells:", length(labels), "\n")
  cat("  Cell types:", length(unique(labels)), "\n")
  
  return(labels)
}

# -----------------------------------------------------------------------------
# Load Coordinates Function (handles multiple formats)
# -----------------------------------------------------------------------------
load_coordinates <- function(filepath) {
  cat("Loading spatial coordinates from:", filepath, "\n")
  
  coords_df <- fread(filepath, header = FALSE)
  
  # Handle different coordinate file formats
  # Format 1: barcode,x,y (3 columns)
  # Format 2: barcode,col1,col2,col3,x,y (6 columns)
  if (ncol(coords_df) == 3) {
    colnames(coords_df) <- c("barcode", "x", "y")
  } else if (ncol(coords_df) >= 5) {
    # Assume last two columns are x, y coordinates
    colnames(coords_df) <- c("barcode", paste0("col", 1:(ncol(coords_df)-3)), "x", "y")
  } else {
    stop("Coordinate file must have at least 3 columns (barcode, x, y)")
  }
  
  # Keep only barcode, x, y for processing
  coords <- coords_df[, c("barcode", "x", "y")]
  coords$x <- as.numeric(coords$x)
  coords$y <- as.numeric(coords$y)
  
  cat("  Loaded", nrow(coords), "spot coordinates\n")
  
  return(as.data.frame(coords))
}

load_ground_truth <- function(filepath) {
  cat("Loading ground truth from:", filepath, "\n")
  
  dt <- fread(filepath, header = TRUE)
  df <- as.data.frame(dt)
  
  # First column should be spot IDs
  rownames(df) <- df[[1]]
  df <- df[, -1, drop = FALSE]
  
  # Convert to numeric matrix
  mat <- as.matrix(df)
  mode(mat) <- "numeric"
  
  cat("  Ground truth dimensions:", nrow(mat), "spots x", ncol(mat), "cell types\n")
  
  return(as.data.frame(mat))
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
cat("\n")
cat("==============================================================================\n")
cat("                         SPOTlight Deconvolution\n")
cat("==============================================================================\n")

cat("\n=== Loading scRNA-seq data ===\n")
# Load SC data first (no reference genes available yet)
sc_counts <- load_counts(args$sc_counts, transpose = args$transpose_sc, 
                         reference_genes = NULL, data_type = "SC")
sc_labels <- load_labels(args$sc_labels)

# Get SC gene names for reference
sc_genes <- rownames(sc_counts)
cat("  SC reference genes:", length(sc_genes), "\n")

cat("\n=== Loading spatial data ===\n")
# Use SC genes as reference for ST data orientation detection
st_result <- load_counts_with_coords(args$st_counts, transpose = args$transpose_st,
                                      reference_genes = sc_genes)
st_counts <- st_result$counts
embedded_coords <- st_result$coords

# Detect data type
spot_names <- colnames(st_counts)
data_type <- detect_data_type(as.data.frame(t(st_counts)), spot_names)
cat("\n  Detected data type:", data_type, "\n")

# -----------------------------------------------------------------------------
# Match cells between counts and labels
# -----------------------------------------------------------------------------
cat("\n=== Validating data ===\n")

cells_in_counts <- colnames(sc_counts)
cells_in_labels <- names(sc_labels)

cat("  Cells in counts:", length(cells_in_counts), "\n")
cat("  Cells in labels:", length(cells_in_labels), "\n")

common_cells <- intersect(cells_in_counts, cells_in_labels)
cat("  Matched cells:", length(common_cells), "\n")

if (length(common_cells) == 0) {
  cat("\nWARNING: No matching cell IDs found!\n")
  cat("First 5 cells in counts:", paste(head(cells_in_counts, 5), collapse = ", "), "\n")
  cat("First 5 cells in labels:", paste(head(cells_in_labels, 5), collapse = ", "), "\n")
  stop("No matching cells between counts and labels!")
}

sc_counts <- sc_counts[, common_cells, drop = FALSE]
sc_labels <- sc_labels[common_cells]

cat("  After filtering:", ncol(sc_counts), "cells\n")
cat("  Cell types:\n")
print(table(sc_labels))

# -----------------------------------------------------------------------------
# Create Seurat Objects
# -----------------------------------------------------------------------------
cat("\n=== Creating Seurat objects ===\n")

sc_counts[sc_counts < 0] <- 0
sc_counts <- round(sc_counts)

sc_seurat <- CreateSeuratObject(
  counts = sc_counts,
  project = "scRNA",
  min.cells = 0,
  min.features = 0
)

sc_seurat$cell_type <- sc_labels[colnames(sc_seurat)]
Idents(sc_seurat) <- sc_seurat$cell_type

cat("  scRNA-seq Seurat object created:", ncol(sc_seurat), "cells\n")

# Process scRNA-seq data
cat("\n=== Processing scRNA-seq data ===\n")
sc_seurat <- NormalizeData(sc_seurat, verbose = FALSE)
sc_seurat <- FindVariableFeatures(
  sc_seurat, 
  selection.method = "vst", 
  nfeatures = args$n_hvg,
  verbose = FALSE
)
sc_seurat <- ScaleData(sc_seurat, verbose = FALSE)

hvg <- VariableFeatures(sc_seurat)
cat("  Selected", length(hvg), "highly variable genes\n")

# Create spatial Seurat object
st_counts[st_counts < 0] <- 0
st_counts <- round(st_counts)

common_genes <- intersect(rownames(sc_counts), rownames(st_counts))
cat("  Common genes between sc and st:", length(common_genes), "\n")

if (length(common_genes) < 100) {
  cat("\nGenes in scRNA-seq:", paste(head(rownames(sc_counts), 10), collapse = ", "), "\n")
  cat("Genes in spatial:", paste(head(rownames(st_counts), 10), collapse = ", "), "\n")
  stop("Too few common genes!")
}

st_counts_filtered <- st_counts[common_genes, , drop = FALSE]

st_seurat <- CreateSeuratObject(
  counts = st_counts_filtered,
  project = "Spatial",
  min.cells = 0,
  min.features = 0
)

cat("  Spatial Seurat object created:", ncol(st_seurat), "spots\n")

# -----------------------------------------------------------------------------
# Handle Coordinates for Spatial Visualization
# -----------------------------------------------------------------------------
# Initialize variables for full coordinate tracking
coords_for_plots <- NULL
coords_full <- NULL
matched_mask <- NULL

# Priority: 1) External coords file, 2) "auto" extraction, 3) Embedded coords, 4) Dummy coords
if (!is.null(args$st_coords) && args$st_coords != "auto") {
  # External coordinate file provided
  cat("\n=== Loading external coordinates ===\n")
  cat("  Loading coordinates from:", args$st_coords, "\n")
  
  # Load ALL coordinates using the multi-format function
  coords_loaded <- load_coordinates(args$st_coords)
  
  # Store ALL coordinates for background plotting
  coords_full <- coords_loaded
  
  # Get spot barcodes from count matrix
  spot_barcodes <- colnames(st_seurat)
  
  # Create matched mask - TRUE for spots that have count data
  matched_mask <- coords_loaded$barcode %in% spot_barcodes
  
  # Get matched coordinates
  coords_matched <- coords_loaded[matched_mask, ]
  
  if (nrow(coords_matched) == 0) {
    warning("No matching barcodes between coordinates and count matrix!")
    coords_for_plots <- NULL
  } else {
    # Report matching statistics
    n_total_coords <- nrow(coords_loaded)
    n_matched <- sum(matched_mask)
    n_unmatched <- sum(!matched_mask)
    
    cat(sprintf("  Total spots in coordinate file: %d\n", n_total_coords))
    cat(sprintf("  Matched spots (have count data): %d\n", n_matched))
    cat(sprintf("  Unmatched spots (no count data): %d\n", n_unmatched))
    
    if (n_unmatched > 0) {
      cat("  -> Unmatched spots will be shown as grey background in spatial plots\n")
    }
    
    # Save matched coords for plotting
    coords_for_plots <- coords_matched
  }
  
} else if ((!is.null(args$st_coords) && args$st_coords == "auto") || !is.null(embedded_coords)) {
  # Use embedded coordinates (either explicitly requested or auto-detected)
  cat("\n=== Using embedded coordinates ===\n")
  
  if (!is.null(embedded_coords)) {
    coords_for_plots <- embedded_coords
    
    # Filter to only spots in the count matrix
    spot_barcodes <- colnames(st_seurat)
    coords_for_plots <- coords_for_plots[coords_for_plots$barcode %in% spot_barcodes, ]
    
    # Remove NA coordinates
    coords_for_plots <- coords_for_plots[!is.na(coords_for_plots$x) & !is.na(coords_for_plots$y), ]
    
    cat("  Using", nrow(coords_for_plots), "spots with valid coordinates\n")
    
    if (nrow(coords_for_plots) == 0) {
      warning("No valid coordinates found!")
      coords_for_plots <- NULL
    }
  } else {
    cat("  No embedded coordinates found, will use dummy coordinates\n")
  }
  
} else if (args$use_dummy_coords) {
  # Generate dummy coordinates
  cat("\n=== Generating dummy coordinates ===\n")
  n_spots <- ncol(st_seurat)
  n_side <- ceiling(sqrt(n_spots))
  
  coords_for_plots <- data.frame(
    barcode = colnames(st_seurat),
    x = rep(1:n_side, length.out = n_spots),
    y = rep(1:n_side, each = n_side)[1:n_spots],
    stringsAsFactors = FALSE
  )
  
  cat("  Generated dummy grid:", n_side, "x", n_side, "\n")
  
} else {
  # No coordinates available
  cat("\n=== No coordinates available ===\n")
  cat("  Spatial visualizations will be skipped\n")
  cat("  Use --st_coords, --st_coords auto, or --use_dummy_coords to enable\n")
}

st_seurat <- NormalizeData(st_seurat, verbose = FALSE)

# -----------------------------------------------------------------------------
# Get marker genes for SPOTlight (IMPROVED)
# -----------------------------------------------------------------------------
cat("\n=== Finding marker genes (IMPROVED SETTINGS) ===\n")

cell_type_counts <- table(sc_seurat$cell_type)
cat("  Cell type distribution:\n")
print(cell_type_counts)

min_cells_for_markers <- 10
valid_cell_types <- names(cell_type_counts[cell_type_counts >= min_cells_for_markers])

if (length(valid_cell_types) < length(cell_type_counts)) {
  cat("  Filtering cell types with <", min_cells_for_markers, "cells\n")
  cat("  Keeping", length(valid_cell_types), "cell types\n")
  
  cells_to_keep <- colnames(sc_seurat)[sc_seurat$cell_type %in% valid_cell_types]
  sc_seurat <- subset(sc_seurat, cells = cells_to_keep)
}

Idents(sc_seurat) <- sc_seurat$cell_type

# IMPROVED: Use lower thresholds and find more markers
cat(sprintf("  Using min.pct = %.2f, logfc.threshold = %.2f\n", 
            args$min_pct, args$logfc_threshold))

markers <- FindAllMarkers(
  sc_seurat,
  only.pos = TRUE,
  min.pct = args$min_pct,           # IMPROVED: Lower threshold (0.1 vs 0.25)
  logfc.threshold = args$logfc_threshold,  # IMPROVED: Lower threshold (0.15 vs 0.25)
  verbose = FALSE
)

cat("  Found", nrow(markers), "marker genes\n")

# IMPROVED: Select more top markers per cell type
top_markers <- markers %>%
  group_by(cluster) %>%
  slice_head(n = args$n_top_markers) %>%  # IMPROVED: More markers (200 vs 100)
  ungroup() %>%
  as.data.frame()

cat("  Selected top", args$n_top_markers, "markers per cell type:", nrow(top_markers), "total\n")
cat("  Marker genes data.frame columns:", paste(colnames(top_markers), collapse = ", "), "\n")

# Report markers per cell type
markers_per_type <- table(top_markers$cluster)
cat("  Markers per cell type:\n")
print(markers_per_type)

# -----------------------------------------------------------------------------
# Run SPOTlight (IMPROVED)
# -----------------------------------------------------------------------------
cat("\n=== Running SPOTlight (COMPREHENSIVE FIX) ===\n")

# STEP 1: Check marker gene quality and remove non-unique markers
cat("\n--- Step 1: Marker Gene Quality Check ---\n")

gene_freq <- table(top_markers$gene)
overlapping_genes <- names(gene_freq[gene_freq > 1])
cat("  Genes marking multiple cell types:", length(overlapping_genes), "\n")

if (length(overlapping_genes) > 0) {
  cat("  Removing overlapping markers for cleaner signal...\n")
  top_markers_clean <- top_markers[!top_markers$gene %in% overlapping_genes, ]
} else {
  top_markers_clean <- top_markers
}

# Verify each cell type still has sufficient markers
markers_per_type <- table(top_markers_clean$cluster)
cat("  Unique markers per cell type:\n")
print(markers_per_type)

# Fallback: if any type has < 10 markers, use original markers
if (any(markers_per_type < 10)) {
  cat("  WARNING: Some types have too few unique markers, using all markers\n")
  top_markers_clean <- top_markers
}

# STEP 2: Balance training data
cat("\n--- Step 2: Balancing Training Data ---\n")

set.seed(42)
cell_type_counts <- table(sc_seurat$cell_type)
min_cells_available <- min(cell_type_counts)
target_cells <- min(min_cells_available, args$n_cells_per_type, 200)

cat("  Cell type distribution in reference:\n")
print(cell_type_counts)
cat("  Sampling", target_cells, "cells per type for balanced training\n")

sampled_cells <- c()
for (ct in unique(sc_seurat$cell_type)) {
  ct_cells <- colnames(sc_seurat)[sc_seurat$cell_type == ct]
  n_sample <- min(length(ct_cells), target_cells)
  sampled_cells <- c(sampled_cells, sample(ct_cells, n_sample))
}

sc_seurat_sub <- subset(sc_seurat, cells = sampled_cells)

cat("  Final training set:\n")
print(table(sc_seurat_sub$cell_type))

# STEP 3: Prepare matrices
cat("\n--- Step 3: Preparing Matrices ---\n")

sc_counts_sparse <- get_counts_matrix(sc_seurat_sub)
st_counts_sparse <- get_counts_matrix(st_seurat)

cat("  SC training matrix:", nrow(sc_counts_sparse), "x", ncol(sc_counts_sparse), "\n")
cat("  ST matrix:", nrow(st_counts_sparse), "x", ncol(st_counts_sparse), "\n")

# Use marker genes + HVGs
marker_genes <- unique(top_markers_clean$gene)
common_hvg <- intersect(hvg, rownames(sc_counts_sparse))
common_hvg <- intersect(common_hvg, rownames(st_counts_sparse))

# Prioritize marker genes
final_genes <- unique(c(marker_genes, common_hvg))
final_genes <- intersect(final_genes, rownames(sc_counts_sparse))
final_genes <- intersect(final_genes, rownames(st_counts_sparse))

cat("  Marker genes:", length(marker_genes), "\n")
cat("  Total genes for model:", length(final_genes), "\n")

# STEP 4: Verify data quality
cat("\n--- Step 4: Data Quality Check ---\n")

# Check for zero-variance genes
sc_gene_vars <- apply(as.matrix(sc_counts_sparse[final_genes, ]), 1, var)
st_gene_vars <- apply(as.matrix(st_counts_sparse[final_genes, ]), 1, var)

zero_var_sc <- sum(sc_gene_vars == 0)
zero_var_st <- sum(st_gene_vars == 0)

cat("  Zero-variance genes in SC:", zero_var_sc, "\n")
cat("  Zero-variance genes in ST:", zero_var_st, "\n")

# Remove zero-variance genes
valid_genes <- final_genes[sc_gene_vars > 0 & st_gene_vars > 0]
cat("  Valid genes after filtering:", length(valid_genes), "\n")

# Get cell type labels
cell_types_vec <- as.character(sc_seurat_sub$cell_type)
names(cell_types_vec) <- colnames(sc_seurat_sub)

# STEP 5: Run SPOTlight
cat("\n--- Step 5: Running SPOTlight ---\n")

spotlight_result <- tryCatch({
  SPOTlight(
    x = sc_counts_sparse,
    y = st_counts_sparse,
    groups = cell_types_vec,
    mgs = top_markers_clean,
    hvg = valid_genes,
    weight_id = "avg_log2FC",
    group_id = "cluster",
    gene_id = "gene",
    model = "ns",
    scale = TRUE
  )
}, error = function(e) {
  cat("  ERROR in SPOTlight:", e$message, "\n")
  cat("  Trying with default parameters...\n")
  
  SPOTlight(
    x = sc_counts_sparse,
    y = st_counts_sparse,
    groups = cell_types_vec,
    mgs = top_markers,  # Use original markers
    hvg = common_hvg,
    weight_id = "avg_log2FC",
    group_id = "cluster",
    gene_id = "gene"
  )
})

cat("  SPOTlight completed!\n")

# STEP 6: Validate results
cat("\n--- Step 6: Results Validation ---\n")

proportions_raw <- spotlight_result$mat
prop_max <- apply(proportions_raw, 1, max)
prop_dominant <- apply(proportions_raw, 1, which.max)

cat("  Proportion statistics:\n")
cat("    Mean max proportion:", round(mean(prop_max), 4), "\n")
cat("    Spots with 100% single type:", sum(prop_max > 0.99), "/", nrow(proportions_raw), "\n")

dominant_types <- colnames(proportions_raw)[prop_dominant]
cat("  Dominant type distribution:\n")
print(table(dominant_types))

# WARNING if results look degenerate
if (mean(prop_max) > 0.95) {
  cat("\n  ⚠️  WARNING: Results appear degenerate (most spots assigned to single type)\n")
  cat("  Consider:\n")
  cat("    1. Using different marker gene settings\n")
  cat("    2. Checking if cell types are truly distinct\n")
  cat("    3. Using a different deconvolution method\n")
}

# -----------------------------------------------------------------------------
# Extract Results (IMPROVED with noise filtering)
# -----------------------------------------------------------------------------
cat("\n=== Extracting results (with noise filtering) ===\n")

proportions <- spotlight_result$mat

proportions <- as.data.frame(proportions)
rownames(proportions) <- colnames(st_seurat)

row_sums <- rowSums(proportions)
row_sums[row_sums == 0] <- 1
proportions <- sweep(proportions, 1, row_sums, "/")

proportions[is.na(proportions)] <- 0

# IMPROVED: Apply minimum proportion threshold to reduce noise
cat(sprintf("  Applying minimum proportion threshold: %.3f\n", args$min_prop))
n_below_threshold <- sum(proportions > 0 & proportions < args$min_prop)
proportions[proportions < args$min_prop] <- 0

# Re-normalize after thresholding
row_sums <- rowSums(proportions)
row_sums[row_sums == 0] <- 1
proportions <- sweep(proportions, 1, row_sums, "/")

cat(sprintf("  Set %d values below threshold to 0 and re-normalized\n", n_below_threshold))

cat("  Proportions matrix:", nrow(proportions), "spots x", ncol(proportions), "cell types\n")
cat("  Cell types in results:\n")
print(colnames(proportions))

# -----------------------------------------------------------------------------
# Reorder columns based on spot names
# -----------------------------------------------------------------------------
proportions <- reorder_columns_by_spots(proportions)

# -----------------------------------------------------------------------------
# Save Output
# -----------------------------------------------------------------------------
cat("\n=== Saving results ===\n")

output_dir <- dirname(args$output_csv)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_df <- data.frame(spot_id = rownames(proportions), proportions, check.names = FALSE)
fwrite(output_df, args$output_csv)
cat("  Saved:", args$output_csv, "\n")

spotlight_output <- file.path(dirname(args$output_csv), "spotlight_full_results.rds")
saveRDS(spotlight_result, spotlight_output)
cat("  Saved full results:", spotlight_output, "\n")

# -----------------------------------------------------------------------------
# Generate Heatmap
# -----------------------------------------------------------------------------
cat("\n=== Generating heatmap ===\n")

max_spots <- 100
if (nrow(proportions) > max_spots) {
  set.seed(42)
  sample_idx <- sample(1:nrow(proportions), max_spots)
  plot_data <- proportions[sample_idx, , drop = FALSE]
  cat("  Sampling", max_spots, "spots for visualization\n")
} else {
  plot_data <- proportions
}

# Remove columns with all zeros
nonzero_cols <- colSums(plot_data) > 0
plot_data <- plot_data[, nonzero_cols, drop = FALSE]

# Check if we have any data left
if (ncol(plot_data) == 0 || nrow(plot_data) == 0) {
  cat("  Warning: No non-zero data to plot!\n")
  png(args$output_plot, width = 800, height = 600, res = 100)
  plot.new()
  text(0.5, 0.5, "No non-zero proportions to display", cex = 1.5)
  dev.off()
} else {
  # Calculate data range
  data_range <- range(plot_data, na.rm = TRUE)
  cat(sprintf("  Data range: [%.4f, %.4f]\n", data_range[1], data_range[2]))
  
  png(args$output_plot, width = 1200, height = 800, res = 100)
  
  # Check if data has enough variance for color scaling
  data_variance <- data_range[2] - data_range[1]
  
  if (data_variance < 1e-10) {
    # All values are essentially the same - create a simple text plot
    cat("  Warning: Data has no variance, creating placeholder plot\n")
    plot.new()
    title(main = "SPOTlight Predicted Proportions")
    text(0.5, 0.5, sprintf("All values are approximately %.4f\n(No variance to display as heatmap)", 
                           data_range[1]), cex = 1.2)
  } else {
    # Create breaks with guaranteed uniqueness
    min_val <- data_range[1]
    max_val <- data_range[2]
    
    # Expand range slightly if too narrow (but not zero variance)
    if (data_variance < 0.01) {
      center <- (min_val + max_val) / 2
      min_val <- max(0, center - 0.01)
      max_val <- min(1, center + 0.01)
    }
    
    # Generate breaks
    n_colors <- 50
    breaks <- seq(min_val, max_val, length.out = n_colors + 1)
    
    # Ensure breaks are strictly increasing (handle floating point issues)
    breaks <- sort(unique(signif(breaks, 10)))
    
    # If we still don't have enough unique breaks, create manual ones
    if (length(breaks) < 3) {
      breaks <- c(min_val, (min_val + max_val) / 2, max_val)
    }
    
    n_colors <- length(breaks) - 1
    colors <- colorRampPalette(c("white", "blue", "red"))(n_colors)
    
    tryCatch({
      pheatmap(
        as.matrix(plot_data),
        cluster_rows = FALSE,
        cluster_cols = FALSE,
        show_rownames = (nrow(plot_data) <= 50),
        show_colnames = TRUE,
        main = "SPOTlight Predicted Proportions",
        color = colors,
        breaks = breaks,
        fontsize = 8,
        scale = "none"  # IMPORTANT: Disable internal scaling
      )
    }, error = function(e) {
      cat(sprintf("  Warning: pheatmap error: %s\n", e$message))
      cat("  Falling back to basic heatmap...\n")
      
      # Ultimate fallback - use base R heatmap
      heatmap(
        as.matrix(plot_data),
        Rowv = NA,
        Colv = NA,
        scale = "none",
        col = colorRampPalette(c("white", "blue", "red"))(50),
        main = "SPOTlight Predicted Proportions",
        margins = c(10, 5)
      )
    })
  }
  
  dev.off()
}

cat("  Saved:", args$output_plot, "\n")

# -----------------------------------------------------------------------------
# Generate Spatial Plots (with full spot display)
# -----------------------------------------------------------------------------
if (!is.null(coords_for_plots)) {
  cat("\n=== Generating spatial visualizations ===\n")
  
  intensity_path <- file.path(output_dir, "spatial_intensity_maps.png")
  dominant_path <- file.path(output_dir, "spatial_dominant_type.png")
  cooccurrence_path <- file.path(output_dir, "cooccurrence_heatmap.png")
  
  # Pass full coordinates and matched mask for background spot display
  plot_spatial_intensity_maps(proportions, coords_for_plots, intensity_path,
                               coords_full = coords_full, matched_mask = matched_mask,
                               rotation = args$hex_orientation)
  plot_spatial_dominant_type(proportions, coords_for_plots, dominant_path,
                              coords_full = coords_full, matched_mask = matched_mask,
                              rotation = args$hex_orientation)
  plot_cooccurrence_heatmap(proportions, cooccurrence_path)
} else {
  cat("\n=== Skipping spatial visualizations (no coordinates available) ===\n")
  
  # Still generate co-occurrence heatmap (doesn't need coordinates)
  cat("\n=== Generating co-occurrence heatmap ===\n")
  cooccurrence_path <- file.path(output_dir, "cooccurrence_heatmap.png")
  plot_cooccurrence_heatmap(proportions, cooccurrence_path)
}

# -----------------------------------------------------------------------------
# Ground Truth Comparison (for simulated data)
# -----------------------------------------------------------------------------
if (!is.null(args$ground_truth)) {
  cat("\n=== Comparing with ground truth ===\n")
  
  ground_truth <- load_ground_truth(args$ground_truth)
  
  corr_plot_path <- file.path(output_dir, "correlation_with_ground_truth.png")
  metrics <- plot_correlation_with_ground_truth(proportions, ground_truth, corr_plot_path)
  
  if (!is.null(metrics)) {
    # Save metrics
    metrics_path <- file.path(output_dir, "ground_truth_metrics.csv")
    metrics_df <- data.frame(
      metric = c("pearson_correlation", "rmse"),
      value = c(metrics$correlation, metrics$rmse)
    )
    fwrite(metrics_df, metrics_path)
    cat("  Saved metrics to:", metrics_path, "\n")
  }
}

cat("\n==============================================================================\n")
cat("                    SPOTlight COMPLETED SUCCESSFULLY!\n")
cat("==============================================================================\n")

cat("\n=== Summary Statistics ===\n")
cat("Average proportions per cell type:\n")
print(round(colMeans(proportions), 4))

cat("\nMax proportion per cell type:\n")
print(round(apply(proportions, 2, max), 4))

cat("\n=== Output Files ===\n")
cat("  Proportions CSV:", args$output_csv, "\n")
cat("  Heatmap:", args$output_plot, "\n")
if (!is.null(coords_for_plots)) {
  cat("  Spatial intensity maps:", file.path(output_dir, "spatial_intensity_maps.png"), "\n")
  cat("  Dominant type map:", file.path(output_dir, "spatial_dominant_type.png"), "\n")
}
cat("  Co-occurrence heatmap:", file.path(output_dir, "cooccurrence_heatmap.png"), "\n")
if (!is.null(args$ground_truth)) {
  cat("  Ground truth correlation:", file.path(output_dir, "correlation_with_ground_truth.png"), "\n")
}

cat("\nDone.\n")