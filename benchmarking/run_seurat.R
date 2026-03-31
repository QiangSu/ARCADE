#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(argparse)
  library(Seurat)
  library(ggplot2)
  library(pheatmap)
  library(dplyr)
  library(reshape2)
  library(gridExtra)
  library(RColorBrewer)
  library(viridis)
  library(FNN)  # For nearest neighbor distance calculation
  library(data.table)
})

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
parser <- ArgumentParser(description = "Run Seurat v3/v4 Label Transfer for Spatial Deconvolution")

# Inputs
parser$add_argument("--sc_counts", required=TRUE, help="Path to single-cell counts CSV")
parser$add_argument("--sc_labels", required=TRUE, help="Path to single-cell labels CSV")
parser$add_argument("--st_counts", required=TRUE, help="Path to spatial counts CSV")

# Outputs
parser$add_argument("--output_csv", required=TRUE, help="Path to save predicted proportions")
parser$add_argument("--output_plot", required=TRUE, help="Path to save heatmap")
parser$add_argument("--output_corr_plot", required=TRUE, help="Path to save correlation plot")

# Parameters
parser$add_argument("--n_hvg", type="integer", default=2000, help="Number of HVGs")
parser$add_argument("--dims", type="integer", default=30, help="PCA dimensions")
parser$add_argument("--ground_truth", default=NULL, help="Path to ground truth (Optional, used for ordering)")

# Spatial visualization arguments
parser$add_argument("--st_coords", default=NULL, 
                    help="Path to spatial coordinates CSV, or 'auto' to extract from counts. If omitted with simulated data, will auto-detect.")
parser$add_argument("--use_dummy_coords", action="store_true", default=FALSE, 
                    help="Generate dummy grid coordinates if no coordinates available")
parser$add_argument("--transpose_sc", action="store_true", default=FALSE, 
                    help="Transpose scRNA-seq counts (if cells are rows)")
parser$add_argument("--transpose_st", action="store_true", default=FALSE, 
                    help="Transpose spatial counts (if spots are rows)")
parser$add_argument("--hex_orientation", type="integer", default=0,
                    help="Hexagon orientation angle in degrees (0 = flat-top, 30 = pointy-top)")

args <- parser$parse_args()

print("============================================================")
print("RUNNING SEURAT LABEL TRANSFER (DECONVOLUTION)")
print("============================================================")

# =============================================================================
# HELPER FUNCTIONS FOR COORDINATE EXTRACTION
# =============================================================================

#' Extract coordinates from spot names (e.g., "spot_5_10" -> x=5, y=10)
extract_coords_from_names <- function(spot_names) {
  # Try pattern: spot_X_Y or spotX_Y
  pattern <- "^spot_?(\\d+)_(\\d+)$"
  
  matches <- regmatches(spot_names, regexec(pattern, spot_names, ignore.case = TRUE))
  
  # Check if most spots match the pattern
  valid_matches <- sapply(matches, length) == 3
  
  if (sum(valid_matches) < length(spot_names) * 0.5) {
    # Less than 50% match - try alternative patterns
    # Try pattern: X_Y (just numbers)
    pattern2 <- "^(\\d+)_(\\d+)$"
    matches2 <- regmatches(spot_names, regexec(pattern2, spot_names))
    valid_matches2 <- sapply(matches2, length) == 3
    
    if (sum(valid_matches2) >= length(spot_names) * 0.5) {
      matches <- matches2
      valid_matches <- valid_matches2
    } else {
      return(NULL)
    }
  }
  
  # Extract coordinates
  coords <- data.frame(
    barcode = spot_names,
    x = NA_real_,
    y = NA_real_,
    stringsAsFactors = FALSE
  )
  
  for (i in which(valid_matches)) {
    coords$x[i] <- as.numeric(matches[[i]][2])
    coords$y[i] <- as.numeric(matches[[i]][3])
  }
  
  # Remove rows with NA coordinates
  coords <- coords[complete.cases(coords), ]
  
  if (nrow(coords) == 0) {
    return(NULL)
  }
  
  cat(sprintf("  Extracted coordinates from %d/%d spot names\n", 
              nrow(coords), length(spot_names)))
  
  return(coords)
}

#' Extract coordinates from row/col columns in count matrix
extract_coords_from_columns <- function(counts_df) {
  colnames_lower <- tolower(colnames(counts_df))
  
  # Look for row/col or x/y columns
  row_idx <- which(colnames_lower %in% c("row", "array_row", "y_coord", "y"))
  col_idx <- which(colnames_lower %in% c("col", "column", "array_col", "x_coord", "x"))
  
  if (length(row_idx) == 0 || length(col_idx) == 0) {
    return(NULL)
  }
  
  row_col <- colnames(counts_df)[row_idx[1]]
  col_col <- colnames(counts_df)[col_idx[1]]
  
  cat(sprintf("  Found coordinate columns: '%s' (row/y) and '%s' (col/x)\n", 
              row_col, col_col))
  
  # Extract coordinates
  coords <- data.frame(
    barcode = rownames(counts_df),
    x = as.numeric(counts_df[[col_col]]),
    y = as.numeric(counts_df[[row_col]]),
    stringsAsFactors = FALSE
  )
  
  # Remove coordinate columns from counts
  coord_cols <- c(row_col, col_col)
  gene_cols <- setdiff(colnames(counts_df), coord_cols)
  counts_clean <- counts_df[, gene_cols, drop = FALSE]
  
  cat(sprintf("  Extracted coordinates for %d spots\n", nrow(coords)))
  cat(sprintf("  Remaining gene columns: %d\n", ncol(counts_clean)))
  
  return(list(
    counts = counts_clean,
    coords = coords
  ))
}

#' Load spatial counts with automatic coordinate detection
load_spatial_counts_with_coords <- function(path, transpose = FALSE, reference_genes = NULL) {
  cat("Loading spatial counts from:", path, "\n")
  
  # Read as data frame first to check for coordinate columns
  counts_df <- read.csv(path, row.names = 1, check.names = FALSE)
  
  if (transpose) {
    cat("  Manual transpose requested...\n")
    counts_df <- as.data.frame(t(counts_df))
  }
  
  extracted_coords <- NULL
  
  # Method 1: Try to extract from row/col columns
  result <- extract_coords_from_columns(counts_df)
  if (!is.null(result)) {
    cat("  Successfully extracted coordinates from row/col columns\n")
    counts_df <- result$counts
    extracted_coords <- result$coords
  } else {
    # Method 2: Try to extract from spot names
    coords_from_names <- extract_coords_from_names(rownames(counts_df))
    if (!is.null(coords_from_names)) {
      cat("  Successfully extracted coordinates from spot names\n")
      extracted_coords <- coords_from_names
    }
  }
  
  # Convert to matrix and use smart transpose WITH reference genes
  counts_mat <- as.matrix(counts_df)
  counts_mat <- smart_transpose(counts_mat, reference_genes = reference_genes, data_type = "ST")
  
  return(list(
    counts = counts_mat,
    coords = extracted_coords
  ))
}

# =============================================================================
# SMART TRANSPOSE FUNCTION
# =============================================================================

#' Smart auto-transpose that uses gene name matching instead of dimension heuristics
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
      cat("    -> Rows are genes (correct for Seurat)\n")
      return(mat)
    } else if (col_gene_overlap > 100 && col_gene_overlap > row_gene_overlap) {
      cat("    -> Columns are genes, transposing to genes x samples...\n")
      return(t(mat))
    }
  }
  
  # Heuristic 2: Check for barcode/spot patterns in names
  barcode_patterns <- c(
    "^[ACGT]{16}-\\d+$",      # Visium barcode
    "^spot_\\d+_\\d+$",        # Simulated
    "^spot_?\\d+x\\d+$",       # Simulated
    "^\\d+_\\d+$",             # Grid
    "^cell_\\d+$",             # cell
    "^Cell\\d+$"               # Cell
  )
  
  count_barcode_matches <- function(names) {
    if (is.null(names)) return(0)
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
  
  # For Seurat: genes should be rows, samples/spots should be columns
  if (row_barcode_matches > col_barcode_matches && row_barcode_matches >= 5) {
    cat("    -> Rows are barcodes/spots, transposing to genes x samples...\n")
    return(t(mat))
  } else if (col_barcode_matches > row_barcode_matches && col_barcode_matches >= 5) {
    cat("    -> Columns are barcodes/spots (correct for Seurat)\n")
    return(mat)
  }
  
  # Heuristic 3: Check for common gene name patterns
  gene_patterns <- c(
    "^[A-Z][A-Z0-9]+$",       # Standard gene
    "^[A-Z]+\\d+$",           # Gene with number
    "^MT-",                    # Mito
    "^RP[SL]\\d+",            # Ribosomal
    "^LINC\\d+"               # Linc
  )
  
  count_gene_matches <- function(names) {
    if (is.null(names)) return(0)
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
    cat("    -> Rows are genes (correct for Seurat)\n")
    return(mat)
  } else if (col_gene_matches > row_gene_matches && col_gene_matches >= 20) {
    cat("    -> Columns are genes, transposing to genes x samples...\n")
    return(t(mat))
  }
  
  # Heuristic 4: Fall back
  cat("    -> Could not determine orientation from names\n")
  cat("    -> Using dimension heuristic (more names = samples)\n")
  
  if (nrow(mat) > ncol(mat)) {
    cat("    -> More rows than columns, assuming rows are samples, transposing...\n")
    return(t(mat))
  } else {
    cat("    -> More columns than rows, assuming columns are samples (correct for Seurat)\n")
    return(mat)
  }
}

# =============================================================================
# HELPER FUNCTIONS FOR SPATIAL VISUALIZATION
# =============================================================================

#' Calculate radius for perfect tessellation
#' @param orientation Rotation angle (30 = pointy-top, 0 = flat-top)
calculate_hex_radius <- function(coords, orientation = 0) {
  if (nrow(coords) < 2) {
    return(100)  # Default radius
  }
  
  # Get x and y coordinates
  xy_matrix <- as.matrix(coords[, c("x", "y")])
  
  # Find nearest neighbor distances
  nn <- get.knn(xy_matrix, k = 1)
  median_dist <- median(nn$nn.dist)
  
  # Calculate radius based on hexagon orientation
  # For dense packing: nearest_neighbor = sqrt(3) * radius
  # This formula applies generally for hex grids regardless of orientation 
  # when defining radius as distance from center to vertex
  radius <- median_dist / sqrt(3)
  
  return(radius * 1.0)
}

#' Create hexagon vertices for a single point
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

#' Load Coordinates Function (handles multiple formats)
load_coordinates <- function(filepath) {
  cat("Loading spatial coordinates from:", filepath, "\n")
  
  coords_df <- fread(filepath, header = FALSE)
  
  # Handle different coordinate file formats
  if (ncol(coords_df) == 3) {
    colnames(coords_df) <- c("barcode", "x", "y")
  } else if (ncol(coords_df) >= 5) {
    colnames(coords_df) <- c("barcode", paste0("col", 1:(ncol(coords_df)-3)), "x", "y")
  } else {
    stop("Coordinate file must have at least 3 columns (barcode, x, y)")
  }
  
  # Keep only barcode, x, y
  coords <- coords_df[, c("barcode", "x", "y")]
  coords$x <- as.numeric(coords$x)
  coords$y <- as.numeric(coords$y)
  
  cat("  Loaded", nrow(coords), "spot coordinates\n")
  
  return(as.data.frame(coords))
}

#' Plot spatial intensity maps - Grid of cell type proportion heatmaps
plot_spatial_intensity_maps <- function(proportions, coords, output_path, 
                                         hex_radius = NULL,
                                         coords_full = NULL, matched_mask = NULL,
                                         hex_angle = 0) {
  
  # Merge proportions with coordinates - PRESERVE COUNT MATRIX ORDER
  prop_df <- as.data.frame(proportions)
  prop_df$barcode <- rownames(prop_df)
  
  # Rotate coordinates 90 degrees clockwise for display consistency: (x, y) -> (y, -x)
  if (!is.null(coords_full)) {
    max_x_ref <- max(coords_full$x, na.rm = TRUE)
    coords_full_rotated <- coords_full
    coords_full_rotated$x <- coords_full$y
    coords_full_rotated$y <- max_x_ref - coords_full$x
    coords_full <- coords_full_rotated
    
    coords_rotated <- coords
    coords_rotated$x <- coords$y
    coords_rotated$y <- max_x_ref - coords$x
    coords <- coords_rotated
  } else {
    max_x_ref <- max(coords$x, na.rm = TRUE)
    coords_rotated <- coords
    coords_rotated$x <- coords$y
    coords_rotated$y <- max_x_ref - coords$x
    coords <- coords_rotated
  }
  
  # Match coordinates
  coord_idx <- match(prop_df$barcode, coords$barcode)
  valid_idx <- !is.na(coord_idx)
  
  merged <- data.frame(
    barcode = prop_df$barcode[valid_idx],
    x = coords$x[coord_idx[valid_idx]],
    y = coords$y[coord_idx[valid_idx]]
  )
  for (col in setdiff(colnames(prop_df), "barcode")) {
    merged[[col]] <- prop_df[[col]][valid_idx]
  }
  
  if (nrow(merged) == 0) {
    warning("No matching barcodes between proportions and coordinates!")
    return(NULL)
  }
  
  # Calculate hex radius
  if (is.null(hex_radius)) {
    if (!is.null(coords_full)) {
      hex_radius <- calculate_hex_radius(coords_full, orientation = hex_angle)
    } else {
      hex_radius <- calculate_hex_radius(merged, orientation = hex_angle)
    }
  }
  
  # Get cell type columns
  cell_types <- setdiff(colnames(prop_df), "barcode")
  
  # Create hexagon polygons
  hex_polys <- create_all_hexagons(merged, hex_radius, rotation = hex_angle)
  
  # Create background hexagons
  bg_hex_polys <- NULL
  has_unmatched <- FALSE
  if (!is.null(coords_full) && !is.null(matched_mask)) {
    bg_hex_polys <- create_background_hexagons(coords_full, matched_mask, hex_radius, rotation = hex_angle)
    has_unmatched <- !is.null(bg_hex_polys) && nrow(bg_hex_polys) > 0
  }
  
  # Calculate axis limits
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
  
  # Create plots
  plot_list <- lapply(cell_types, function(ct) {
    hex_data <- hex_polys
    hex_data$value <- merged[[ct]][hex_data$spot_idx]
    
    p <- ggplot()
    
    if (has_unmatched) {
      p <- p + geom_polygon(data = bg_hex_polys, 
                            aes(x = x, y = y, group = id),
                            fill = "lightgrey", color = NA, 
                            linewidth = 0, alpha = 0.3)
    }
    
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
  
  png(output_path, width = n_cols * 5, height = n_rows * 5, units = "in", res = 300)
  grid.arrange(grobs = plot_list, ncol = n_cols)
  dev.off()
  
  cat(sprintf("Saved intensity maps to: %s\n", output_path))
}

#' Plot spatial dominant type map
plot_spatial_dominant_type <- function(proportions, coords, output_path, 
                                        hex_radius = NULL,
                                        coords_full = NULL, matched_mask = NULL,
                                        hex_angle = 0) {
  
  prop_df <- as.data.frame(proportions)
  prop_df$barcode <- rownames(prop_df)
  
  # Rotate coordinates 90 degrees clockwise: (x, y) -> (y, -x)
  if (!is.null(coords_full)) {
    max_x_ref <- max(coords_full$x, na.rm = TRUE)
    coords_full_rotated <- coords_full
    coords_full_rotated$x <- coords_full$y
    coords_full_rotated$y <- max_x_ref - coords_full$x
    coords_full <- coords_full_rotated
    
    coords_rotated <- coords
    coords_rotated$x <- coords$y
    coords_rotated$y <- max_x_ref - coords$x
    coords <- coords_rotated
  } else {
    max_x_ref <- max(coords$x, na.rm = TRUE)
    coords_rotated <- coords
    coords_rotated$x <- coords$y
    coords_rotated$y <- max_x_ref - coords$x
    coords <- coords_rotated
  }
  
  # Match coordinates
  coord_idx <- match(prop_df$barcode, coords$barcode)
  valid_idx <- !is.na(coord_idx)
  
  merged <- data.frame(
    barcode = prop_df$barcode[valid_idx],
    x = coords$x[coord_idx[valid_idx]],
    y = coords$y[coord_idx[valid_idx]]
  )
  for (col in setdiff(colnames(prop_df), "barcode")) {
    merged[[col]] <- prop_df[[col]][valid_idx]
  }
  
  if (nrow(merged) == 0) {
    warning("No matching barcodes between proportions and coordinates!")
    return(NULL)
  }
  
  # Calculate hex radius
  if (is.null(hex_radius)) {
    if (!is.null(coords_full)) {
      hex_radius <- calculate_hex_radius(coords_full, orientation = hex_angle)
    } else {
      hex_radius <- calculate_hex_radius(merged, orientation = hex_angle)
    }
  }
  
  # Determine dominant types
  cell_types <- setdiff(colnames(prop_df), "barcode")
  prop_matrix <- as.matrix(merged[, cell_types])
  merged$dominant_type <- cell_types[apply(prop_matrix, 1, which.max)]
  merged$max_proportion <- apply(prop_matrix, 1, max)
  
  # Create hexagon polygons for matched spots
  hex_polys <- create_all_hexagons(merged, hex_radius, rotation = hex_angle)
  hex_polys$dominant_type <- merged$dominant_type[hex_polys$spot_idx]
  
  # Create background hexagons for unmatched spots
  bg_hex_polys <- NULL
  has_unmatched <- FALSE
  if (!is.null(coords_full) && !is.null(matched_mask)) {
    bg_hex_polys <- create_background_hexagons(coords_full, matched_mask, hex_radius, rotation = hex_angle)
    has_unmatched <- !is.null(bg_hex_polys) && nrow(bg_hex_polys) > 0
  }
  
  # Calculate axis limits
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

  # 3. Determine types present in the dataset (alphabetical order)
  present_types <- sort(unique(merged$dominant_type))
  
  # 4. Construct the final color vector
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
    used_hex <- as.character(na.omit(type_colors))
    available_hex <- setdiff(tab20_colors, used_hex)
    
    if (length(available_hex) < length(remaining_types)) {
      available_hex <- colorRampPalette(tab20_colors)(length(remaining_types))
    }
    type_colors[remaining_types] <- available_hex[1:length(remaining_types)]
  }
  
  # Prepare plotting data
  # Merge background spots into main data for unified legend handling
  plot_data <- hex_polys
  plot_data$fill_group <- plot_data$dominant_type
  plot_data$alpha_val <- 1.0
  
  legend_breaks <- present_types
  
  if (has_unmatched) {
    bg_data <- bg_hex_polys
    bg_data$dominant_type <- "No count data" 
    bg_data$fill_group <- "No count data"
    bg_data$alpha_val <- 0.5 # Grey spots have transparency
    
    # Ensure columns match for rbind (only keep necessary columns)
    cols_to_keep <- c("x", "y", "id", "fill_group", "alpha_val")
    plot_data <- plot_data[, cols_to_keep]
    bg_data <- bg_data[, cols_to_keep]
    
    plot_data <- rbind(plot_data, bg_data)
    
    # Add grey to color map for "No count data"
    type_colors <- c(type_colors, "No count data" = "lightgrey")
    
    # Add "No count data" to breaks to ensure it appears at bottom of legend
    legend_breaks <- c(legend_breaks, "No count data")
  }
  
  # Convert fill_group to factor
  plot_data$fill_group <- factor(plot_data$fill_group, levels = legend_breaks)
  
  # Create plot
  p <- ggplot(plot_data, aes(x = x, y = y, group = id, fill = fill_group, alpha = alpha_val)) +
    geom_polygon(color = NA, linewidth = 0) +
    scale_fill_manual(
      values = type_colors, 
      name = "Cell Type",
      breaks = legend_breaks
    ) +
    scale_alpha_identity() +
    coord_fixed(xlim = xlim, ylim = ylim) +
    theme_void() +
    theme(
      text = element_text(family = "sans"),
      panel.grid = element_blank(),
      axis.text = element_blank(),
      axis.title = element_blank(),
      axis.ticks = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
      legend.title = element_text(size = 15, face = "bold"),
      legend.text = element_text(size = 14),
      legend.position = "right",
      legend.key.size = unit(0.8, "cm")
    ) +
    ggtitle("Dominant Cell Type per Spot")
  
  if (has_unmatched) {
    n_unmatched <- sum(!matched_mask)
    p <- p + labs(subtitle = sprintf("(%d matched spots + %d spots with no data)", 
                                     nrow(merged), n_unmatched)) +
      theme(plot.subtitle = element_text(hjust = 0.5, size = 12, face = "italic", color = "grey40"))
  }
  
  ggsave(output_path, p, width = 12, height = 10, dpi = 300)
  cat(sprintf("Saved dominant type map to: %s\n", output_path))
}

#' Plot cell type co-occurrence (correlation) heatmap
plot_cooccurrence_heatmap <- function(proportions, output_path) {
  prop_matrix <- as.matrix(proportions)
  corr_matrix <- cor(prop_matrix, method = "pearson")
  corr_matrix[is.na(corr_matrix)] <- 0
  
  png(output_path, width = 10, height = 8, units = "in", res = 300)
  pheatmap(
    corr_matrix,
    cluster_rows = TRUE, cluster_cols = TRUE,
    display_numbers = TRUE, number_format = "%.2f", number_color = "black",
    fontsize_number = 8,
    color = colorRampPalette(c("#2166AC", "white", "#B2182B"))(100),
    breaks = seq(-1, 1, length.out = 101),
    main = "Cell Type Co-occurrence (Correlation)",
    border_color = "white", cellwidth = 25, cellheight = 25
  )
  dev.off()
  cat(sprintf("Saved co-occurrence heatmap to: %s\n", output_path))
}

#' Helper: Load Data
load_counts <- function(path, type_name, transpose = FALSE) {
  print(paste("Loading", type_name, "from:", path))
  counts <- read.csv(path, row.names = 1, check.names = FALSE)
  if (transpose) {
    print("  Transposing matrix...")
    counts <- t(counts)
  }
  return(as.matrix(counts))
}

# ==============================================================================
# LOAD DATA
# ==============================================================================

# 1. Load Single Cell FIRST
print("=== Loading Single Cell Data ===")
sc_mat <- load_counts(args$sc_counts, "Single Cell", transpose = args$transpose_sc)
sc_meta <- read.csv(args$sc_labels, row.names = 1)

if (!args$transpose_sc) {
  if (ncol(sc_mat) == nrow(sc_meta)) {
    print("  Orientation: Genes x Cells (Correct)")
  } else if (nrow(sc_mat) == nrow(sc_meta)) {
    print("  Orientation: Cells x Genes (Transposing)")
    sc_mat <- t(sc_mat)
  } else {
    print("  Metadata dimensions don't match directly, using smart detection...")
    sc_mat <- smart_transpose(sc_mat, reference_genes = NULL, data_type = "SC")
    if (nrow(sc_mat) == nrow(sc_meta)) {
      sc_mat <- t(sc_mat)
    }
  }
}

sc_gene_names <- rownames(sc_mat)
cat(sprintf("  SC dimensions: %d genes x %d cells\n", nrow(sc_mat), ncol(sc_mat)))

valid_cols <- c('cell_type', 'celltype', 'CellType', 'cell_types', 'labels', 'cluster')
label_col <- colnames(sc_meta)[colnames(sc_meta) %in% valid_cols][1]
if (is.na(label_col)) label_col <- colnames(sc_meta)[1]
original_labels <- unique(sc_meta[[label_col]])

sc_obj <- CreateSeuratObject(counts = sc_mat, meta.data = sc_meta)
Idents(sc_obj) <- sc_meta[[label_col]]

# 2. Load Spatial Data
print("\n=== Loading Spatial Data ===")

coord_mode <- "none"
if (!is.null(args$st_coords)) {
  if (tolower(args$st_coords) == "auto") {
    coord_mode <- "auto"
  } else if (file.exists(args$st_coords)) {
    coord_mode <- "file"
  } else {
    warning(paste("Coordinate file not found:", args$st_coords, "- trying auto-detection"))
    coord_mode <- "auto"
  }
} else {
  coord_mode <- "auto"
}

st_mat <- NULL
embedded_coords <- NULL
coords_for_plots <- NULL
coords_full <- NULL
matched_mask <- NULL

if (coord_mode == "auto") {
  result <- load_spatial_counts_with_coords(args$st_counts, 
                                             transpose = args$transpose_st,
                                             reference_genes = sc_gene_names)
  st_mat <- result$counts
  embedded_coords <- result$coords
  
  if (!is.null(embedded_coords)) {
    cat("  Using coordinates extracted from count matrix\n")
    coords_for_plots <- embedded_coords
    coords_full <- embedded_coords
    matched_mask <- rep(TRUE, nrow(embedded_coords))
  } else if (args$use_dummy_coords) {
    cat("  No coordinates found, will generate dummy coordinates\n")
  }
  
} else if (coord_mode == "file") {
  st_mat <- load_counts(args$st_counts, "Spatial", transpose = args$transpose_st)
  if (!args$transpose_st) {
    st_mat <- smart_transpose(st_mat, reference_genes = sc_gene_names, data_type = "ST")
  }
  
  coords_loaded <- load_coordinates(args$st_coords)
  coords_full <- coords_loaded
  
  spot_barcodes <- colnames(st_mat)
  matched_mask <- coords_loaded$barcode %in% spot_barcodes
  coords_matched <- coords_loaded[matched_mask, ]
  
  if (nrow(coords_matched) > 0) {
    coords_for_plots <- coords_matched
  }
  
  cat(sprintf("  Matched spots: %d\n", sum(matched_mask)))
} else {
  st_mat <- load_counts(args$st_counts, "Spatial", transpose = args$transpose_st)
  if (!args$transpose_st) {
    st_mat <- smart_transpose(st_mat, reference_genes = sc_gene_names, data_type = "ST")
  }
}

if (is.null(coords_for_plots) && args$use_dummy_coords) {
  cat("  Generating dummy grid coordinates\n")
  n_spots <- ncol(st_mat)
  n_side <- ceiling(sqrt(n_spots))
  
  coords_for_plots <- data.frame(
    barcode = colnames(st_mat),
    x = rep(1:n_side, length.out = n_spots),
    y = rep(1:n_side, each = n_side)[1:n_spots],
    stringsAsFactors = FALSE
  )
  coords_full <- coords_for_plots
  matched_mask <- rep(TRUE, n_spots)
}

st_obj <- CreateSeuratObject(counts = st_mat)
cat(sprintf("  ST dimensions: %d genes x %d spots\n", nrow(st_mat), ncol(st_mat)))

# ==============================================================================
# PREPROCESSING
# ==============================================================================
print("Preprocessing Reference...")
sc_obj <- NormalizeData(sc_obj, verbose = FALSE)
sc_obj <- FindVariableFeatures(sc_obj, selection.method = "vst", nfeatures = args$n_hvg, verbose = FALSE)
sc_obj <- ScaleData(sc_obj, verbose = FALSE)
sc_obj <- RunPCA(sc_obj, features = VariableFeatures(object = sc_obj), verbose = FALSE)

print("Preprocessing Query (Spatial)...")
st_obj <- NormalizeData(st_obj, verbose = FALSE)
st_obj <- FindVariableFeatures(st_obj, selection.method = "vst", nfeatures = args$n_hvg, verbose = FALSE)
st_obj <- ScaleData(st_obj, verbose = FALSE)

# ==============================================================================
# FIND ANCHORS & TRANSFER LABELS
# ==============================================================================
print("Finding Transfer Anchors...")
n_spots <- ncol(st_obj)
dims_use <- 1:min(args$dims, n_spots-1)
k_filter_use <- min(200, n_spots)
k_score_use <- min(30, n_spots - 1)
if (k_score_use < 2) k_score_use <- 2

anchors <- FindTransferAnchors(
  reference = sc_obj, query = st_obj, features = VariableFeatures(object = sc_obj),
  reference.assay = "RNA", query.assay = "RNA", reduction = "cca",
  dims = dims_use, k.filter = k_filter_use, k.score = k_score_use, verbose = TRUE
)

print("Transferring Data...")
k_weight_use <- min(50, n_spots - 1)
if (k_weight_use < 2) k_weight_use <- 2 

predictions <- TransferData(
  anchorset = anchors, refdata = Idents(sc_obj), dims = dims_use,
  weight.reduction = "cca", k.weight = k_weight_use, verbose = TRUE
)

# ==============================================================================
# NAME CORRECTION & FORMATTING
# ==============================================================================
print("Formatting output names...")
score_cols <- grep("prediction.score", colnames(predictions), value = TRUE)
score_cols <- score_cols[score_cols != "prediction.score.max"]
prop_df <- predictions[, score_cols]

current_names <- gsub("prediction.score.", "", colnames(prop_df))
mapping_dict <- list()
for (real_name in original_labels) {
    safe_name <- make.names(real_name)
    mapping_dict[[safe_name]] <- real_name
}
new_names <- sapply(current_names, function(x) {
    if (x %in% names(mapping_dict)) return(mapping_dict[[x]])
    else return(x)
})
colnames(prop_df) <- new_names
prop_df <- prop_df / rowSums(prop_df)

# ==============================================================================
# REORDERING COLUMNS
# ==============================================================================
gt <- NULL
if (!is.null(args$ground_truth)) {
    print(paste("Loading ground truth from:", args$ground_truth))
    gt <- read.csv(args$ground_truth, row.names = 1, check.names = FALSE)
    common_types <- intersect(colnames(prop_df), colnames(gt))
    if (length(common_types) > 0) {
        extra_cols <- setdiff(colnames(prop_df), colnames(gt))
        final_order <- c(colnames(gt)[colnames(gt) %in% colnames(prop_df)], extra_cols)
        prop_df <- prop_df[, final_order, drop=FALSE]
    } else {
        prop_df <- prop_df[, order(colnames(prop_df)), drop=FALSE]
    }
} else {
    prop_df <- prop_df[, order(colnames(prop_df)), drop=FALSE]
}

# ==============================================================================
# OUTPUTS
# ==============================================================================
dir.create(dirname(args$output_csv), recursive = TRUE, showWarnings = FALSE)
write.csv(prop_df, args$output_csv, quote = FALSE)

print("Generating heatmap...")
tryCatch({
    png(args$output_plot, width = 1200, height = 800, res = 150)
    pheatmap(as.matrix(prop_df), cluster_rows = FALSE, cluster_cols = FALSE, 
             main = "Seurat Predicted Proportions", fontsize_row = 8, fontsize_col = 10, angle_col = 45)
    dev.off()
}, error = function(e) {
    print(paste("Error generating heatmap:", e$message))
})

# ==============================================================================
# SPATIAL VISUALIZATIONS
# ==============================================================================
if (!is.null(coords_for_plots)) {
  cat("\n=== Generating spatial visualizations ===\n")
  cat(sprintf("  Hexagon orientation: %d degrees\n", args$hex_orientation))
  
  output_dir <- dirname(args$output_csv)
  intensity_path <- file.path(output_dir, "spatial_intensity_maps.png")
  dominant_path <- file.path(output_dir, "spatial_dominant_type.png")
  cooccurrence_path <- file.path(output_dir, "cooccurrence_heatmap.png")
  
  # Pass full coordinates, matched mask, and orientation angle
  plot_spatial_intensity_maps(prop_df, coords_for_plots, intensity_path,
                               coords_full = coords_full, matched_mask = matched_mask,
                               hex_angle = args$hex_orientation)
  plot_spatial_dominant_type(prop_df, coords_for_plots, dominant_path,
                              coords_full = coords_full, matched_mask = matched_mask,
                              hex_angle = args$hex_orientation)
  plot_cooccurrence_heatmap(prop_df, cooccurrence_path)
} else {
  cat("\n=== Skipping spatial visualizations (no coordinates available) ===\n")
  output_dir <- dirname(args$output_csv)
  cooccurrence_path <- file.path(output_dir, "cooccurrence_heatmap.png")
  plot_cooccurrence_heatmap(prop_df, cooccurrence_path)
}

# ==============================================================================
# GROUND TRUTH METRICS
# ==============================================================================
if (!is.null(gt)) {
  print("Calculating final metrics...")
  common_spots <- intersect(rownames(prop_df), rownames(gt))
  common_types <- intersect(colnames(prop_df), colnames(gt))
  
  if (length(common_spots) > 0 && length(common_types) > 0) {
    pred_sub <- prop_df[common_spots, common_types]
    gt_sub <- gt[common_spots, common_types]
    
    flat_pred <- as.vector(as.matrix(pred_sub))
    flat_gt <- as.vector(as.matrix(gt_sub))
    
    cor_val <- cor(flat_pred, flat_gt)
    print(paste("OVERALL PEARSON CORRELATION:", round(cor_val, 4)))
    
    df_metrics <- data.frame(Metric = "Pearson", Value = cor_val)
    write.csv(df_metrics, paste0(args$output_csv, "_seurat_metrics.csv"), quote=FALSE)
    
    p_df <- data.frame(Truth = flat_gt, Prediction = flat_pred)
    png(args$output_corr_plot, width = 800, height = 800, res = 120)
    p <- ggplot(p_df, aes(x=Truth, y=Prediction)) +
      geom_point(alpha=0.5, color="blue") +
      geom_abline(slope=1, intercept=0, color="red", linetype="dashed") +
      theme_minimal() +
      labs(title=paste("Seurat vs Ground Truth (Corr:", round(cor_val, 3), ")"),
           x="Ground Truth Proportion", y="Predicted Proportion")
    print(p)
    dev.off()
  }
}

cat("\n============================================================\n")
cat("SEURAT LABEL TRANSFER COMPLETE\n")
cat("============================================================\n")
print("Done.")