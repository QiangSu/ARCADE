#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(spacexr)
  library(Matrix)
  library(data.table)
  library(pheatmap)
  library(ggplot2)
  library(gridExtra)
  library(RColorBrewer)
  library(viridis)
  library(FNN)  # For nearest neighbor distance calculation
})

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
parser <- ArgumentParser(description = "Run RCTD for spatial deconvolution (supports multiple dataset formats)")

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
                    help = "Path to spatial coordinates CSV, or 'auto' to extract from count matrix")
parser$add_argument("--ground_truth", default = NULL,
                    help = "Path to ground truth proportions CSV for simulated data")
parser$add_argument("--doublet_mode", default = "doublet",
                    help = "RCTD mode: doublet, full, or multi")
parser$add_argument("--n_cores", type = "integer", default = 4,
                    help = "Number of cores")
parser$add_argument("--transpose_sc", action = "store_true", default = FALSE,
                    help = "Transpose scRNA-seq counts (if cells are rows)")
parser$add_argument("--transpose_st", action = "store_true", default = FALSE,
                    help = "Transpose spatial counts (if spots are rows)")
parser$add_argument("--hex_orientation", type = "integer", default = 0,
                    help = "Hexagon orientation angle in degrees (0 = flat-top, 30 = pointy-top)")

args <- parser$parse_args()

# -----------------------------------------------------------------------------
# Helper Function: Extract cell type from spot name
# -----------------------------------------------------------------------------
extract_celltype_from_spot <- function(spot_name) {
  # Remove batch prefix like "Batch_01_Pure_" or "Batch_02_Mix_"
  # Pattern: Batch_XX_Pure_ or Batch_XX_Mix_XXpct_
  
  # First try to extract from "Pure" spots
  if (grepl("_Pure_", spot_name)) {
    celltype <- sub("^Batch_[0-9]+_Pure_", "", spot_name)
  } else if (grepl("_Mix_", spot_name)) {
    # For mix spots like "Batch_01_Mix_50pct_TypeA_50pct_TypeB"
    # Extract the first cell type after percentage
    celltype <- sub("^Batch_[0-9]+_Mix_[0-9]+pct_", "", spot_name)
    celltype <- sub("_[0-9]+pct_.*$", "", celltype)
  } else {
    celltype <- spot_name
  }
  
  # Replace underscores with spaces to match column names
  celltype <- gsub("_", " ", celltype)
  
  return(celltype)
}

# -----------------------------------------------------------------------------
# Helper Function: Normalize cell type name for matching
# -----------------------------------------------------------------------------
normalize_name <- function(name) {
  # Convert to lowercase, remove extra spaces, standardize
  name <- tolower(name)
  name <- gsub("\\s+", " ", name)  # Multiple spaces to single
  name <- trimws(name)
  return(name)
}

# -----------------------------------------------------------------------------
# Helper Function: Find best matching column name
# -----------------------------------------------------------------------------
find_matching_column <- function(celltype, column_names) {
  # Normalize the cell type from spot name
  norm_celltype <- normalize_name(celltype)
  
  # Normalize all column names
  norm_columns <- sapply(column_names, normalize_name)
  
  # Exact match
  exact_match <- which(norm_columns == norm_celltype)
  if (length(exact_match) > 0) {
    return(column_names[exact_match[1]])
  }
  
  # Partial match (column contains celltype or vice versa)
  for (i in seq_along(column_names)) {
    if (grepl(norm_celltype, norm_columns[i], fixed = TRUE) ||
        grepl(norm_columns[i], norm_celltype, fixed = TRUE)) {
      return(column_names[i])
    }
  }
  
  return(NA)
}
# =============================================================================
# SMART TRANSPOSE FUNCTION 
# =============================================================================

#' Smart auto-transpose that uses gene name matching instead of dimension heuristics
#' @param mat Matrix to potentially transpose
#' @param reference_genes Optional set of known gene names (from SC data)
#' @param data_type String for logging ("SC" or "ST")
#' @return Matrix in correct orientation (genes x samples for RCTD)
smart_transpose <- function(mat, reference_genes = NULL, data_type = "Data") {
  cat(sprintf("  %s format detection:\n", data_type))
  cat(sprintf("    Input dimensions: %d x %d\n", nrow(mat), ncol(mat)))
  
  row_names <- rownames(mat)
  col_names <- colnames(mat)
  
  # Heuristic 1: Check overlap with reference genes
  if (!is.null(reference_genes) && length(reference_genes) > 0) {
    row_gene_overlap <- length(intersect(row_names, reference_genes))
    col_gene_overlap <- length(intersect(col_names, reference_genes))
    
    cat(sprintf("    Reference gene overlap - Rows: %d, Cols: %d\n", 
                row_gene_overlap, col_gene_overlap))
    
    if (row_gene_overlap > 100 && row_gene_overlap > col_gene_overlap) {
      cat("    -> Rows are genes (correct for RCTD)\n")
      return(mat)
    } else if (col_gene_overlap > 100 && col_gene_overlap > row_gene_overlap) {
      cat("    -> Columns are genes, transposing to genes x samples...\n")
      return(t(mat))
    }
  }
  
  # Heuristic 2: Check for barcode/spot patterns in names
  barcode_patterns <- c(
    "^[ACGT]{16}-\\d+$",
    "^spot_\\d+_\\d+$",
    "^spot_?\\d+x\\d+$",
    "^\\d+_\\d+$",
    "^cell_\\d+$",
    "^Cell\\d+$",
    "^row\\d+_col\\d+$"
  )
  
  count_barcode_matches <- function(names) {
    if (is.null(names) || length(names) == 0) return(0)
    test_names <- head(names, 50)
    matches <- sapply(test_names, function(n) {
      any(sapply(barcode_patterns, function(p) grepl(p, n, ignore.case = TRUE)))
    })
    sum(matches)
  }
  
  row_barcode_matches <- count_barcode_matches(row_names)
  col_barcode_matches <- count_barcode_matches(col_names)
  
  cat(sprintf("    Barcode pattern matches - Rows: %d/50, Cols: %d/50\n",
              row_barcode_matches, col_barcode_matches))
  
  # For RCTD: genes should be rows, samples/spots should be columns
  if (row_barcode_matches > col_barcode_matches && row_barcode_matches >= 5) {
    cat("    -> Rows are barcodes/spots, transposing to genes x samples...\n")
    return(t(mat))
  } else if (col_barcode_matches > row_barcode_matches && col_barcode_matches >= 5) {
    cat("    -> Columns are barcodes/spots (correct for RCTD)\n")
    return(mat)
  }
  
  # Heuristic 3: Check for common gene name patterns
  gene_patterns <- c(
    "^[A-Z][A-Z0-9]+$",
    "^[A-Z]+\\d+$",
    "^MT-",
    "^RP[SL]\\d+",
    "^LINC\\d+",
    "^LOC\\d+",
    "^[A-Z]{2,}[0-9]*$"
  )
  
  count_gene_matches <- function(names) {
    if (is.null(names) || length(names) == 0) return(0)
    test_names <- head(names, 100)
    matches <- sapply(test_names, function(n) {
      any(sapply(gene_patterns, function(p) grepl(p, n)))
    })
    sum(matches)
  }
  
  row_gene_matches <- count_gene_matches(row_names)
  col_gene_matches <- count_gene_matches(col_names)
  
  cat(sprintf("    Gene pattern matches - Rows: %d/100, Cols: %d/100\n",
              row_gene_matches, col_gene_matches))
  
  if (row_gene_matches > col_gene_matches && row_gene_matches >= 20) {
    cat("    -> Rows are genes (correct for RCTD)\n")
    return(mat)
  } else if (col_gene_matches > row_gene_matches && col_gene_matches >= 20) {
    cat("    -> Columns are genes, transposing to genes x samples...\n")
    return(t(mat))
  }
  
  # Heuristic 4: Fall back with warning
  cat("    -> Could not determine orientation from names\n")
  cat("    -> WARNING: Using dimension heuristic - verify results!\n")
  
  # For typical spatial data: fewer spots than genes in full transcriptome
  # But simulated data might have more spots
  # This is unreliable, so just return as-is with warning
  cat("    -> Keeping original orientation (use --transpose_st if incorrect)\n")
  return(mat)
}
# -----------------------------------------------------------------------------
# Helper Function: Reorder columns based on spot names
# -----------------------------------------------------------------------------
reorder_columns_by_spots <- function(proportions_df) {
  spot_names <- rownames(proportions_df)
  col_names <- colnames(proportions_df)
  
  cat("\n=== Reordering columns to match spot order ===\n")
  
  # Extract cell types from spot names and find matching columns
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
  
  # Add remaining columns that weren't matched
  remaining_cols <- setdiff(col_names, ordered_cols)
  if (length(remaining_cols) > 0) {
    cat("\n  Unmatched columns (appended at end):\n")
    for (col in remaining_cols) {
      cat("    ", col, "\n")
    }
    ordered_cols <- c(ordered_cols, remaining_cols)
  }
  
  # Reorder the dataframe columns
  proportions_reordered <- proportions_df[, ordered_cols, drop = FALSE]
  
  cat("\n  Final column order:\n")
  for (i in seq_along(ordered_cols)) {
    cat("    ", i, ": ", ordered_cols[i], "\n")
  }
  
  return(proportions_reordered)
}

# -----------------------------------------------------------------------------
# NEW: Extract coordinates from count matrix (for simulated data)
# -----------------------------------------------------------------------------
extract_coordinates_from_counts <- function(st_counts_df, spot_names) {
  cat("\n=== Attempting to extract coordinates from count matrix ===\n")
  
  # Method 1: Check for row/col columns in the data frame
  if ("row" %in% colnames(st_counts_df) && "col" %in% colnames(st_counts_df)) {
    cat("  Found 'row' and 'col' columns in count matrix\n")
    coords <- data.frame(
      barcode = spot_names,
      x = as.numeric(st_counts_df$col),
      y = as.numeric(st_counts_df$row)
    )
    return(coords)
  }
  
  # Method 2: Parse spot names for coordinate patterns
  # Pattern: spot_X_Y or similar
  coord_pattern <- "spot_([0-9]+)_([0-9]+)"
  matches <- regmatches(spot_names, regexec(coord_pattern, spot_names))
  
  if (all(sapply(matches, length) == 3)) {
    cat("  Extracting coordinates from spot names (pattern: spot_X_Y)\n")
    coords <- data.frame(
      barcode = spot_names,
      x = as.numeric(sapply(matches, `[`, 2)),
      y = as.numeric(sapply(matches, `[`, 3))
    )
    return(coords)
  }
  
  # Method 3: Check for X_Y pattern at end of spot names
  coord_pattern2 <- "_([0-9]+)_([0-9]+)$"
  matches2 <- regmatches(spot_names, regexec(coord_pattern2, spot_names))
  
  if (all(sapply(matches2, length) == 3)) {
    cat("  Extracting coordinates from spot name suffixes (pattern: *_X_Y)\n")
    coords <- data.frame(
      barcode = spot_names,
      x = as.numeric(sapply(matches2, `[`, 2)),
      y = as.numeric(sapply(matches2, `[`, 3))
    )
    return(coords)
  }
  
  # Method 4: Check for row_col format (row5_col10)
  coord_pattern3 <- "row([0-9]+)_col([0-9]+)"
  matches3 <- regmatches(spot_names, regexec(coord_pattern3, spot_names, ignore.case = TRUE))
  
  if (all(sapply(matches3, length) == 3)) {
    cat("  Extracting coordinates from spot names (pattern: rowX_colY)\n")
    coords <- data.frame(
      barcode = spot_names,
      x = as.numeric(sapply(matches3, `[`, 3)),  # col = x
      y = as.numeric(sapply(matches3, `[`, 2))   # row = y
    )
    return(coords)
  }
  
  cat("  Could not extract coordinates from count matrix\n")
  return(NULL)
}

# -----------------------------------------------------------------------------
# NEW: Detect dataset type
# -----------------------------------------------------------------------------
detect_dataset_type <- function(st_counts_df, spot_names, st_coords_arg) {
  cat("\n=== Detecting dataset type ===\n")
  
  # Check if coordinates file provided
  if (!is.null(st_coords_arg) && st_coords_arg != "auto") {
    cat("  Dataset type: Real Visium (external coordinates provided)\n")
    return("visium")
  }
  
  # Check for embedded coordinates
  has_row_col <- "row" %in% colnames(st_counts_df) && "col" %in% colnames(st_counts_df)
  
  # Check spot naming patterns
  has_spot_pattern <- any(grepl("spot_[0-9]+_[0-9]+", spot_names))
  has_rowcol_pattern <- any(grepl("row[0-9]+_col[0-9]+", spot_names, ignore.case = TRUE))
  has_suffix_pattern <- any(grepl("_[0-9]+_[0-9]+$", spot_names))
  
  if (has_row_col || has_spot_pattern || has_rowcol_pattern || has_suffix_pattern) {
    cat("  Dataset type: Simulated (embedded coordinates detected)\n")
    if (has_row_col) cat("    - Found row/col columns\n")
    if (has_spot_pattern) cat("    - Found spot_X_Y naming pattern\n")
    if (has_rowcol_pattern) cat("    - Found rowX_colY naming pattern\n")
    if (has_suffix_pattern) cat("    - Found _X_Y suffix pattern\n")
    return("simulated")
  }
  
  # Check for Visium-style barcodes
  visium_pattern <- "^[ACGT]+-[0-9]+$"
  if (any(grepl(visium_pattern, spot_names))) {
    cat("  Dataset type: Real Visium (Visium-style barcodes detected)\n")
    return("visium")
  }
  
  cat("  Dataset type: Unknown (defaulting to generic)\n")
  return("unknown")
}

# -----------------------------------------------------------------------------
# Helper Functions for Spatial Visualization
# -----------------------------------------------------------------------------
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
  # Pointy-top (30) or Flat-top (0)
  # For dense packing: nearest_neighbor = sqrt(3) * radius
  radius <- median_dist / sqrt(3)
  
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
                                         hex_angle = 0) {
  
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
      hex_radius <- calculate_hex_radius(coords_full, orientation = hex_angle)
    } else {
      hex_radius <- calculate_hex_radius(merged, orientation = hex_angle)
    }
  }
  
  # Get cell type columns
  cell_types <- setdiff(colnames(prop_df), "barcode")
  
  # Create hexagon polygons for matched spots
  hex_polys <- create_all_hexagons(merged, hex_radius, rotation = hex_angle)
  
  # Create background hexagons for unmatched spots
  bg_hex_polys <- NULL
  has_unmatched <- FALSE
  if (!is.null(coords_full) && !is.null(matched_mask)) {
    bg_hex_polys <- create_background_hexagons(coords_full, matched_mask, hex_radius, rotation = hex_angle)
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
                            fill = "lightgrey", color = NA, 
                            linewidth = 0, alpha = 0.3)
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
      theme_void() +
      theme(
        text = element_text(family = "sans"),
        panel.grid = element_blank(),
        axis.text = element_blank(),
        axis.title = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
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
  png(output_path, width = n_cols * 5, height = n_rows * 5, units = "in", res = 300)
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
#' Now supports showing ALL spots including unmatched ones as grey background
plot_spatial_dominant_type <- function(proportions, coords, output_path, 
                                        hex_radius = NULL,
                                        coords_full = NULL, matched_mask = NULL,
                                        hex_angle = 0) {
  
  # Merge proportions with coordinates - PRESERVE COUNT MATRIX ORDER
  prop_df <- as.data.frame(proportions)
  prop_df$barcode <- rownames(prop_df)
  
  # Determine max_x from full coords if available (for consistent rotation)
  if (!is.null(coords_full)) {
    max_x_ref <- max(coords_full$x, na.rm = TRUE)
    
    # Rotate full coords first: (x, y) -> (y, max_x - x)
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
  
  # Use match() to preserve the order from proportions
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
      hex_radius <- calculate_hex_radius(coords_full, orientation = hex_angle)
    } else {
      hex_radius <- calculate_hex_radius(merged, orientation = hex_angle)
    }
  }
  
  # Get cell type columns and find dominant type
  cell_types <- setdiff(colnames(prop_df), "barcode")
  prop_matrix <- as.matrix(merged[, cell_types])
  merged$dominant_type <- cell_types[apply(prop_matrix, 1, which.max)]
  merged$max_proportion <- apply(prop_matrix, 1, max)
  
  # Create hexagon polygons for matched spots
  hex_polys <- create_all_hexagons(merged, hex_radius, rotation = hex_angle)
  hex_polys$dominant_type <- merged$dominant_type[hex_polys$spot_idx]
  hex_polys$max_proportion <- merged$max_proportion[hex_polys$spot_idx]
  
  # Create background hexagons for unmatched spots
  bg_hex_polys <- NULL
  has_unmatched <- FALSE
  if (!is.null(coords_full) && !is.null(matched_mask)) {
    bg_hex_polys <- create_background_hexagons(coords_full, matched_mask, hex_radius, rotation = hex_angle)
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
  # 1. Standard Tab20 Palette (for fallback)
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
  # "101 ZI Pax6 Gaba"                       = "#1f77b4", # Dark Blue (Duplicate)
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

  # 3. Determine types present in current data
  present_types <- sort(unique(merged$dominant_type))
  
  # 4. Construct the final color vector
  # Initialize with NA
  final_colors <- setNames(rep(NA, length(present_types)), present_types)
  
  # First pass: Assign specific colors if cell types exist in data
  for (ct in names(specific_mapping)) {
    if (ct %in% present_types) {
      final_colors[ct] <- specific_mapping[ct]
    }
  }
  
  # Second pass: Assign remaining types (if any) using unused Tab20 colors
  remaining_types <- names(final_colors)[is.na(final_colors)]
  if (length(remaining_types) > 0) {
    # Find hex codes already used
    used_hex <- as.character(na.omit(final_colors))
    # Find available codes from Tab20
    available_hex <- setdiff(tab20_colors, used_hex)
    
    # If not enough colors, fallback to colorRamp
    if (length(available_hex) < length(remaining_types)) {
      available_hex <- colorRampPalette(tab20_colors)(length(remaining_types))
    }
    
    final_colors[remaining_types] <- available_hex[1:length(remaining_types)]
  }
  
  # Create plot
  p <- ggplot()
  
  # Add background (unmatched) spots first
  if (has_unmatched) {
    p <- p + geom_polygon(data = bg_hex_polys, 
                          aes(x = x, y = y, group = id),
                          fill = "lightgrey", color = NA,
                          linewidth = 0, alpha = 0.3)
  }
  
  # Add matched spots with cell type colors
  p <- p + geom_polygon(data = hex_polys, 
                        aes(x = x, y = y, group = id, fill = dominant_type),
                        color = NA, linewidth = 0) +
    # Use final_colors named vector here
    scale_fill_manual(values = final_colors, name = "Cell Type") +
    coord_fixed(xlim = xlim, ylim = ylim) +
    theme_void() + # Remove standard axes/grid
    theme(
      text = element_text(family = "sans"),
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      
      # Match Python Font Settings
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
      legend.title = element_text(size = 15, face = "bold"),
      legend.text = element_text(size = 14),
      
      # Legend positioning (Right side)
      legend.position = "right",
      legend.key.size = unit(0.8, "cm")
    ) +
    ggtitle("Dominant Cell Type per Spot")
  
  # Add subtitle for unmatched spots if they exist
  if (has_unmatched) {
    n_unmatched <- sum(!matched_mask)
    p <- p + labs(subtitle = sprintf("Grey spots: No count data (%d spots)", n_unmatched)) +
      theme(plot.subtitle = element_text(hjust = 0.5, size = 12, face = "italic", color = "grey40"))
  }
  
  # Save plot
  ggsave(output_path, p, width = 12, height = 10, dpi = 300)
  
  cat(sprintf("Saved dominant type map to: %s\n", output_path))
  
  if (has_unmatched) {
    cat(sprintf("  (Showing %d matched spots + %d unmatched spots as grey background)\n",
                nrow(merged), n_unmatched))
  }
}

# -----------------------------------------------------------------------------
# NEW: Plot Co-occurrence Heatmap
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
# NEW: Plot correlation with ground truth
# -----------------------------------------------------------------------------
plot_correlation_with_ground_truth <- function(proportions, ground_truth_path, output_dir) {
  cat("\n=== Comparing with ground truth ===\n")
  
  # Load ground truth
  gt_df <- fread(ground_truth_path, header = TRUE)
  gt_df <- as.data.frame(gt_df)
  
  # Identify spot ID column
  if ("spot_id" %in% colnames(gt_df)) {
    spot_col <- "spot_id"
  } else if ("barcode" %in% colnames(gt_df)) {
    spot_col <- "barcode"
  } else {
    spot_col <- colnames(gt_df)[1]
  }
  
  rownames(gt_df) <- gt_df[[spot_col]]
  gt_df <- gt_df[, -which(colnames(gt_df) == spot_col), drop = FALSE]
  
  # Get common spots and cell types
  common_spots <- intersect(rownames(proportions), rownames(gt_df))
  
  if (length(common_spots) == 0) {
    warning("No matching spots between predictions and ground truth!")
    return(NULL)
  }
  
  # Normalize column names for matching
  pred_cols <- colnames(proportions)
  gt_cols <- colnames(gt_df)
  
  # Create mapping between prediction and ground truth columns
  col_mapping <- list()
  for (pc in pred_cols) {
    pc_norm <- normalize_name(pc)
    for (gc in gt_cols) {
      gc_norm <- normalize_name(gc)
      if (pc_norm == gc_norm || grepl(pc_norm, gc_norm) || grepl(gc_norm, pc_norm)) {
        col_mapping[[pc]] <- gc
        break
      }
    }
  }
  
  common_types <- names(col_mapping)
  
  if (length(common_types) == 0) {
    warning("No matching cell types between predictions and ground truth!")
    cat("  Prediction columns:", paste(pred_cols, collapse = ", "), "\n")
    cat("  Ground truth columns:", paste(gt_cols, collapse = ", "), "\n")
    return(NULL)
  }
  
  cat("  Matched spots:", length(common_spots), "\n")
  cat("  Matched cell types:", length(common_types), "\n")
  
  # Align data
  pred_aligned <- proportions[common_spots, common_types, drop = FALSE]
  gt_aligned <- gt_df[common_spots, sapply(common_types, function(x) col_mapping[[x]]), drop = FALSE]
  colnames(gt_aligned) <- common_types
  
  # Flatten for correlation
  pred_vec <- as.vector(as.matrix(pred_aligned))
  gt_vec <- as.vector(as.matrix(gt_aligned))
  
  # Calculate metrics
  pearson_corr <- cor(pred_vec, gt_vec, method = "pearson")
  spearman_corr <- cor(pred_vec, gt_vec, method = "spearman")
  rmse <- sqrt(mean((pred_vec - gt_vec)^2))
  mae <- mean(abs(pred_vec - gt_vec))
  
  cat(sprintf("  Pearson correlation: %.4f\n", pearson_corr))
  cat(sprintf("  Spearman correlation: %.4f\n", spearman_corr))
  cat(sprintf("  RMSE: %.4f\n", rmse))
  cat(sprintf("  MAE: %.4f\n", mae))
  
  # Create scatter plot
  plot_df <- data.frame(
    ground_truth = gt_vec,
    predicted = pred_vec
  )
  
  p <- ggplot(plot_df, aes(x = ground_truth, y = predicted)) +
    geom_point(alpha = 0.5, size = 1) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    geom_smooth(method = "lm", se = TRUE, color = "blue", alpha = 0.3) +
    labs(
      title = "RCTD Predictions vs Ground Truth",
      subtitle = sprintf("Pearson r = %.3f, RMSE = %.4f", pearson_corr, rmse),
      x = "Ground Truth Proportion",
      y = "Predicted Proportion"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 11)
    ) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1))
  
  scatter_path <- file.path(output_dir, "ground_truth_correlation.png")
  ggsave(scatter_path, p, width = 8, height = 8, dpi = 300)
  cat(sprintf("  Saved correlation plot to: %s\n", scatter_path))
  
  # Save metrics to CSV
  metrics_df <- data.frame(
    metric = c("pearson_correlation", "spearman_correlation", "rmse", "mae", 
               "n_spots", "n_cell_types"),
    value = c(pearson_corr, spearman_corr, rmse, mae, 
              length(common_spots), length(common_types))
  )
  metrics_path <- file.path(output_dir, "ground_truth_metrics.csv")
  fwrite(metrics_df, metrics_path)
  cat(sprintf("  Saved metrics to: %s\n", metrics_path))
  
  # Per-cell-type correlation
  per_type_corr <- sapply(common_types, function(ct) {
    cor(pred_aligned[[ct]], gt_aligned[[ct]], method = "pearson")
  })
  
  per_type_df <- data.frame(
    cell_type = common_types,
    pearson_correlation = per_type_corr
  )
  per_type_path <- file.path(output_dir, "per_celltype_correlation.csv")
  fwrite(per_type_df, per_type_path)
  cat(sprintf("  Saved per-cell-type correlations to: %s\n", per_type_path))
  
  return(list(
    pearson = pearson_corr,
    spearman = spearman_corr,
    rmse = rmse,
    mae = mae
  ))
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
  
  # Manual transpose flag takes priority
  if (transpose) {
    cat("  Manual transpose requested...\n")
    mat <- t(mat)
  } else {
    # Use smart auto-detection
    mat <- smart_transpose(mat, reference_genes = reference_genes, data_type = data_type)
  }
  
  mode(mat) <- "numeric"
  
  cat("  Final dimensions:", nrow(mat), "genes x", ncol(mat), "cells/spots\n")
  return(mat)
}

# Modified to return both matrix and original df (for coordinate extraction)
load_counts_with_metadata <- function(filepath, transpose = FALSE, reference_genes = NULL, data_type = "Data") {
  cat("Reading:", filepath, "\n")
  
  dt <- fread(filepath, header = TRUE)
  df <- as.data.frame(dt)
  
  # Store original for coordinate extraction
  original_df <- df
  
  rownames(df) <- df[[1]]
  df <- df[, -1, drop = FALSE]
  
  mat <- as.matrix(df)
  
  # Manual transpose flag takes priority
  if (transpose) {
    cat("  Manual transpose requested...\n")
    mat <- t(mat)
  } else {
    # Use smart auto-detection
    mat <- smart_transpose(mat, reference_genes = reference_genes, data_type = data_type)
  }
  
  mode(mat) <- "numeric"
  
  cat("  Final dimensions:", nrow(mat), "genes x", ncol(mat), "cells/spots\n")
  return(list(matrix = mat, original_df = original_df))
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
  
  labels <- as.factor(df[[type_col]])
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

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
cat("\n=== Loading scRNA-seq data ===\n")
sc_counts <- load_counts(args$sc_counts, transpose = args$transpose_sc, 
                          reference_genes = NULL, data_type = "SC")
sc_labels <- load_labels(args$sc_labels)

# Store SC gene names for ST orientation detection
sc_gene_names <- rownames(sc_counts)
cat("  SC genes (first 5):", paste(head(sc_gene_names, 5), collapse = ", "), "\n")

cat("\n=== Loading spatial data ===\n")
st_data <- load_counts_with_metadata(args$st_counts, transpose = args$transpose_st,
                                      reference_genes = sc_gene_names, data_type = "ST")
st_counts <- st_data$matrix
st_original_df <- st_data$original_df

# Detect dataset type
dataset_type <- detect_dataset_type(st_original_df, colnames(st_counts), args$st_coords)

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
# Filter rare cell types (need minimum 5 cells per type for RCTD)
# -----------------------------------------------------------------------------
cat("\n=== Filtering rare cell types ===\n")
min_cells <- 5

cell_type_counts <- table(sc_labels)
rare_types <- names(cell_type_counts[cell_type_counts < min_cells])

if (length(rare_types) > 0) {
  cat("  Removing cell types with <", min_cells, "cells:\n")
  for (rt in rare_types) {
    cat("    -", rt, "(", cell_type_counts[rt], "cells)\n")
  }
  
  # Filter out cells belonging to rare types
  keep_cells <- names(sc_labels)[!sc_labels %in% rare_types]
  sc_counts <- sc_counts[, keep_cells, drop = FALSE]
  sc_labels <- sc_labels[keep_cells]
  sc_labels <- droplevels(sc_labels)  # Remove unused factor levels
  
  cat("  Remaining cells:", length(sc_labels), "\n")
  cat("  Remaining cell types:", length(unique(sc_labels)), "\n")
}

# -----------------------------------------------------------------------------
# Create Reference
# -----------------------------------------------------------------------------
cat("\n=== Creating RCTD Reference ===\n")

gene_vars <- apply(sc_counts, 1, var)
sc_counts <- sc_counts[gene_vars > 0, , drop = FALSE]
cat("  Genes after variance filter:", nrow(sc_counts), "\n")

reference <- Reference(
  counts = sc_counts,
  cell_types = sc_labels,
  n_max_cells = 10000
)
cat("  Reference created successfully\n")

# -----------------------------------------------------------------------------
# Create SpatialRNA object (handles multiple coordinate sources)
# -----------------------------------------------------------------------------
cat("\n=== Creating SpatialRNA object ===\n")

# Initialize variables for full coordinate tracking
coords_for_plots <- NULL
coords_full <- NULL
matched_mask <- NULL

if (args$use_dummy_coords) {
  # Option 1: Generate dummy coordinates
  cat("  Generating dummy coordinates\n")
  n_spots <- ncol(st_counts)
  n_side <- ceiling(sqrt(n_spots))
  coords <- data.frame(
    x = rep(1:n_side, length.out = n_spots),
    y = rep(1:n_side, each = n_side)[1:n_spots]
  )
  rownames(coords) <- colnames(st_counts)
  
  # Create coords_for_plots with barcode column for spatial visualization
  coords_for_plots <- data.frame(
    barcode = colnames(st_counts),
    x = coords$x,
    y = coords$y
  )
  
} else if (!is.null(args$st_coords) && args$st_coords == "auto") {
  # Option 2: Auto-extract coordinates from count matrix
  cat("  Attempting to auto-extract coordinates from count matrix\n")
  
  coords_extracted <- extract_coordinates_from_counts(st_original_df, colnames(st_counts))
  
  if (!is.null(coords_extracted)) {
    cat("  Successfully extracted coordinates\n")
    
    coords_full <- coords_extracted
    matched_mask <- rep(TRUE, nrow(coords_extracted))
    
    # Create coordinate matrix for RCTD
    coords <- as.data.frame(coords_extracted[, c("x", "y")])
    rownames(coords) <- coords_extracted$barcode
    
    coords_for_plots <- coords_extracted
  } else {
    # Fall back to dummy coordinates
    cat("  Could not extract coordinates, using dummy coordinates\n")
    n_spots <- ncol(st_counts)
    n_side <- ceiling(sqrt(n_spots))
    coords <- data.frame(
      x = rep(1:n_side, length.out = n_spots),
      y = rep(1:n_side, each = n_side)[1:n_spots]
    )
    rownames(coords) <- colnames(st_counts)
    
    coords_for_plots <- data.frame(
      barcode = colnames(st_counts),
      x = coords$x,
      y = coords$y
    )
  }
  
} else if (!is.null(args$st_coords)) {
  # Option 3: Load coordinates from external file
  coords_loaded <- load_coordinates(args$st_coords)
  
  # Store ALL coordinates for background plotting
  coords_full <- coords_loaded
  
  # Get spot barcodes from count matrix
  spot_barcodes <- colnames(st_counts)
  
  # Create matched mask - TRUE for spots that have count data
  matched_mask <- coords_loaded$barcode %in% spot_barcodes
  
  # Get matched coordinates
  coords_matched <- coords_loaded[matched_mask, ]
  
  if (nrow(coords_matched) == 0) {
    stop("No matching barcodes between coordinates and count matrix!")
  }
  
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
  
  # Create coordinate matrix for RCTD
  coords <- as.data.frame(coords_matched[, c("x", "y")])
  rownames(coords) <- coords_matched$barcode
  
  # Save matched coords for plotting
  coords_for_plots <- coords_matched
  
  # Subset st_counts to matched spots only
  st_counts <- st_counts[, coords_matched$barcode, drop = FALSE]
  
} else {
  # Option 4: Try to extract coordinates, fall back to dummy
  cat("  No coordinate source specified, attempting auto-detection\n")
  
  coords_extracted <- extract_coordinates_from_counts(st_original_df, colnames(st_counts))
  
  if (!is.null(coords_extracted)) {
    cat("  Successfully extracted coordinates from count matrix\n")
    
    coords_full <- coords_extracted
    matched_mask <- rep(TRUE, nrow(coords_extracted))
    
    coords <- as.data.frame(coords_extracted[, c("x", "y")])
    rownames(coords) <- coords_extracted$barcode
    
    coords_for_plots <- coords_extracted
  } else {
    cat("  Using dummy coordinates\n")
    n_spots <- ncol(st_counts)
    n_side <- ceiling(sqrt(n_spots))
    coords <- data.frame(
      x = rep(1:n_side, length.out = n_spots),
      y = rep(1:n_side, each = n_side)[1:n_spots]
    )
    rownames(coords) <- colnames(st_counts)
    
    coords_for_plots <- data.frame(
      barcode = colnames(st_counts),
      x = coords$x,
      y = coords$y
    )
  }
}

common_genes <- intersect(rownames(sc_counts), rownames(st_counts))
cat("  Common genes:", length(common_genes), "\n")

if (length(common_genes) < 10) {
  cat("\nGenes in reference:", paste(head(rownames(sc_counts), 10), collapse = ", "), "\n")
  cat("Genes in spatial:", paste(head(rownames(st_counts), 10), collapse = ", "), "\n")
  stop("Too few common genes!")
}

st_counts_filtered <- st_counts[common_genes, , drop = FALSE]

st_counts_filtered[st_counts_filtered < 0] <- 0
st_counts_filtered <- round(st_counts_filtered)
mode(st_counts_filtered) <- "integer"

spatialRNA <- SpatialRNA(
  coords = coords,
  counts = st_counts_filtered,
  nUMI = colSums(st_counts_filtered)
)
cat("  SpatialRNA object created\n")

# -----------------------------------------------------------------------------
# Run RCTD
# -----------------------------------------------------------------------------
cat("\n=== Running RCTD ===\n")
cat("  Mode:", args$doublet_mode, "\n")
cat("  Cores:", args$n_cores, "\n")

myRCTD <- create.RCTD(
  spatialRNA = spatialRNA,
  reference = reference,
  max_cores = args$n_cores,
  CELL_MIN_INSTANCE = 5
)

myRCTD <- run.RCTD(
  myRCTD,
  doublet_mode = args$doublet_mode
)

cat("  RCTD completed!\n")

# -----------------------------------------------------------------------------
# Extract Results
# -----------------------------------------------------------------------------
cat("\n=== Extracting results ===\n")

if (args$doublet_mode == "full") {
  weights <- myRCTD@results$weights
  
  # Convert sparse matrix to regular matrix
  weights_mat <- as.matrix(weights)
  
  # Normalize to proportions
  row_sums <- rowSums(weights_mat)
  row_sums[row_sums == 0] <- 1
  proportions <- sweep(weights_mat, 1, row_sums, "/")
  
  # Convert to data frame
  proportions <- as.data.frame(proportions)
  
} else {
  # For doublet or multi mode
  weights <- myRCTD@results$weights
  weights_mat <- as.matrix(weights)
  row_sums <- rowSums(weights_mat)
  row_sums[row_sums == 0] <- 1
  proportions <- as.data.frame(sweep(weights_mat, 1, row_sums, "/"))
}

proportions[is.na(proportions)] <- 0

cat("  Proportions matrix:", nrow(proportions), "spots x", ncol(proportions), "cell types\n")

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

# -----------------------------------------------------------------------------
# Generate Heatmap (original output)
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

nonzero_cols <- colSums(plot_data) > 0
plot_data <- plot_data[, nonzero_cols, drop = FALSE]

# For heatmap, also reorder rows to show diagonal pattern
png(args$output_plot, width = 3600, height = 2400, res = 300)
pheatmap(
  as.matrix(plot_data),
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  show_rownames = (nrow(plot_data) <= 50),
  show_colnames = TRUE,
  main = paste("RCTD Predicted Proportions (", args$doublet_mode, " mode)", sep = ""),
  color = colorRampPalette(c("white", "blue", "red"))(100),
  fontsize = 8
)
dev.off()
cat("  Saved:", args$output_plot, "\n")

# -----------------------------------------------------------------------------
# Generate Spatial Plots (with full spot display)
# -----------------------------------------------------------------------------
if (!is.null(coords_for_plots)) {
  cat("\n=== Generating spatial visualizations ===\n")
  cat(sprintf("  Hexagon orientation: %d degrees\n", args$hex_orientation))
  
  intensity_path <- file.path(output_dir, "spatial_intensity_maps.png")
  dominant_path <- file.path(output_dir, "spatial_dominant_type.png")
  cooccurrence_path <- file.path(output_dir, "cooccurrence_heatmap.png")
  
  # Pass full coordinates and matched mask for background spot display
  plot_spatial_intensity_maps(proportions, coords_for_plots, intensity_path,
                               coords_full = coords_full, matched_mask = matched_mask,
                               hex_angle = args$hex_orientation)
  plot_spatial_dominant_type(proportions, coords_for_plots, dominant_path,
                              coords_full = coords_full, matched_mask = matched_mask,
                              hex_angle = args$hex_orientation)
  plot_cooccurrence_heatmap(proportions, cooccurrence_path)
} else {
  cat("\n=== Skipping spatial visualizations (no coordinates available) ===\n")
  
  # Still generate co-occurrence heatmap (doesn't need coordinates)
  cat("\n=== Generating co-occurrence heatmap ===\n")
  cooccurrence_path <- file.path(output_dir, "cooccurrence_heatmap.png")
  plot_cooccurrence_heatmap(proportions, cooccurrence_path)
}

# -----------------------------------------------------------------------------
# Compare with ground truth if provided
# -----------------------------------------------------------------------------
if (!is.null(args$ground_truth)) {
  gt_metrics <- plot_correlation_with_ground_truth(proportions, args$ground_truth, output_dir)
}

# -----------------------------------------------------------------------------
# Summary Statistics
# -----------------------------------------------------------------------------
cat("\n=== Summary Statistics ===\n")
cat("Average proportions per cell type:\n")
print(round(colMeans(proportions), 4))

cat("\nMax proportion per cell type:\n")
print(round(apply(proportions, 2, max), 4))

cat("\n=== COMPLETED SUCCESSFULLY! ===\n")