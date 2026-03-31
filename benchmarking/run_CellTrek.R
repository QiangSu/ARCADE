#!/usr/bin/env Rscript

# ==============================================================================
# CellTrek Spatial Deconvolution Pipeline with Cell State Tracking (FIXED v3)
# ==============================================================================
# Key fixes over v2:
#   - celltrek() fix: adds required "type" column ("cell"/"spot") to co-embedded object
#   - KNN fallback completely rewritten: for each ST spot, finds K nearest SC 
#     neighbors in PCA space and counts their cell types → proportions
#   - This ensures ALL spots get proportions, not just a few
#   - Fixed "Distance between spots is: 0" by adding proper image coordinates
# ==============================================================================

suppressPackageStartupMessages({
  library(argparse)
  library(Seurat)
  library(SeuratObject)
  library(CellTrek)
  library(randomForest)
  library(ggplot2)
  library(pheatmap)
  library(dplyr)
  library(tidyr)
  library(reshape2)
  library(gridExtra)
  library(RColorBrewer)
  library(viridis)
  library(FNN)
  library(data.table)
  library(dbscan)
  library(uwot)
})

# ==============================================================================
# CRITICAL FIX: Patch GetAssayData for Seurat 5.4.0 compatibility
# ==============================================================================

.original_GetAssayData <- NULL

patch_seurat_for_celltrek <- function() {
  if (packageVersion("SeuratObject") >= "5.0.0") {
    cat("Patching Seurat API for CellTrek compatibility (slot -> layer)...\n")
    
    .original_GetAssayData <<- getFromNamespace("GetAssayData", "SeuratObject")
    
    patched_GetAssayData <- function(object, slot = NULL, layer = NULL, ...) {
      if (!is.null(slot) && is.null(layer)) {
        layer <- slot
        slot <- NULL
      }
      if (!is.null(layer)) {
        return(.original_GetAssayData(object, layer = layer, ...))
      } else {
        return(.original_GetAssayData(object, ...))
      }
    }
    
    tryCatch({
      assignInNamespace("GetAssayData", patched_GetAssayData, ns = "SeuratObject")
      cat("  GetAssayData patched successfully\n")
    }, error = function(e) {
      cat(sprintf("  Warning: Could not patch namespace: %s\n", e$message))
      assign("GetAssayData", patched_GetAssayData, envir = .GlobalEnv)
    })
    
    tryCatch({
      original_SetAssayData <- getFromNamespace("SetAssayData", "SeuratObject")
      patched_SetAssayData <- function(object, slot = NULL, layer = NULL, new.data, ...) {
        if (!is.null(slot) && is.null(layer)) {
          layer <- slot
          slot <- NULL
        }
        if (!is.null(layer)) {
          return(original_SetAssayData(object, layer = layer, new.data = new.data, ...))
        } else {
          return(original_SetAssayData(object, new.data = new.data, ...))
        }
      }
      assignInNamespace("SetAssayData", patched_SetAssayData, ns = "SeuratObject")
      cat("  SetAssayData patched successfully\n")
    }, error = function(e) {
      cat(sprintf("  Note: SetAssayData patch skipped: %s\n", e$message))
    })
  }
}

patch_seurat_for_celltrek()

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
parser <- ArgumentParser(description = "CellTrek Spatial Deconvolution with Cell State Tracking")

parser$add_argument("--sc_counts", required = TRUE)
parser$add_argument("--sc_labels", required = TRUE)
parser$add_argument("--st_counts", required = TRUE)
parser$add_argument("--st_coords", default = NULL)
parser$add_argument("--output_dir", default = "./celltrek_output")
parser$add_argument("--output_csv", default = NULL)
parser$add_argument("--celltype_col", default = "cell_type")
parser$add_argument("--n_hvg", type = "integer", default = 3000)
parser$add_argument("--dims", type = "integer", default = 30)
parser$add_argument("--dist_thresh", type = "double", default = 0.55)
parser$add_argument("--intp_pnt", type = "integer", default = 5000)
parser$add_argument("--spot_n", type = "integer", default = 10)
parser$add_argument("--repel_r", type = "integer", default = 20)
parser$add_argument("--repel_iter", type = "integer", default = 20)
parser$add_argument("--ground_truth", default = NULL)
parser$add_argument("--transpose_sc", action = "store_true", default = FALSE)
parser$add_argument("--transpose_st", action = "store_true", default = FALSE)
parser$add_argument("--hex_orientation", type = "integer", default = 0)
parser$add_argument("--state_dims", type = "integer", default = 10)
parser$add_argument("--state_resolution", type = "double", default = 0.5)
parser$add_argument("--knn_k", type = "integer", default = 50,
                    help = "Number of SC neighbors per spot for KNN fallback deconvolution")
parser$add_argument("--min_prop", type = "double", default = 0,
                    help = "Minimum proportion threshold for intensity maps. Values below this are colored grey. (default: 1e-6)")

args <- parser$parse_args()

dir.create(args$output_dir, recursive = TRUE, showWarnings = FALSE)

if (is.null(args$output_csv)) {
  args$output_csv <- file.path(args$output_dir, "proportions.csv")
}

cat("============================================================\n")
cat("CELLTREK SPATIAL DECONVOLUTION WITH CELL STATE TRACKING (v3)\n")
cat("============================================================\n")
cat(sprintf("Output directory: %s\n", args$output_dir))
cat(sprintf("Seurat version: %s\n", as.character(packageVersion("Seurat"))))
cat(sprintf("SeuratObject version: %s\n", as.character(packageVersion("SeuratObject"))))
cat(sprintf("CellTrek version: %s\n", as.character(packageVersion("CellTrek"))))

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

load_matrix <- function(filepath) {
  if (grepl("\\.rds$", filepath, ignore.case = TRUE)) {
    return(readRDS(filepath))
  } else if (grepl("\\.csv$", filepath, ignore.case = TRUE)) {
    mat <- read.csv(filepath, row.names = 1, check.names = FALSE)
    return(as.matrix(mat))
  } else {
    stop("Unsupported file format. Use CSV or RDS.")
  }
}

safe_get_assay_data <- function(object, layer_name = "counts", assay = NULL) {
  if (!is.null(assay)) {
    DefaultAssay(object) <- assay
  }
  tryCatch({
    return(GetAssayData(object, layer = layer_name))
  }, error = function(e1) {
    tryCatch({
      return(GetAssayData(object, slot = layer_name))
    }, error = function(e2) {
      assay_obj <- object[[DefaultAssay(object)]]
      if (inherits(assay_obj, "Assay")) {
        if (layer_name == "counts") return(assay_obj@counts)
        if (layer_name == "data") return(assay_obj@data)
        if (layer_name == "scale.data") return(assay_obj@scale.data)
      }
      if (inherits(assay_obj, "Assay5")) {
        return(LayerData(assay_obj, layer = layer_name))
      }
      stop(sprintf("Cannot access %s from assay", layer_name))
    })
  })
}

create_celltrek_seurat <- function(counts_matrix, meta_data = NULL, assay_name = "RNA") {
  cat("  Creating CellTrek-compatible Seurat object...\n")
  
  if (!inherits(counts_matrix, "dgCMatrix")) {
    counts_matrix <- as(as.matrix(counts_matrix), "dgCMatrix")
  }
  
  obj <- CreateSeuratObject(counts = counts_matrix, assay = assay_name)
  
  if (packageVersion("Seurat") >= "5.0.0") {
    current_class <- class(obj[[assay_name]])[1]
    cat(sprintf("    Initial assay class: %s\n", current_class))
    
    if (current_class == "Assay5") {
      cat("    Converting to legacy Assay for CellTrek...\n")
      
      counts_data <- tryCatch({
        LayerData(obj[[assay_name]], layer = "counts")
      }, error = function(e) {
        counts_matrix
      })
      
      if (!inherits(counts_data, "dgCMatrix")) {
        counts_data <- as(as.matrix(counts_data), "dgCMatrix")
      }
      
      legacy_assay <- CreateAssayObject(counts = counts_data)
      
      if (inherits(legacy_assay, "Assay5")) {
        cat("    CreateAssayObject returned Assay5, building manually...\n")
        legacy_assay <- new(
          Class = "Assay",
          counts = counts_data,
          data = counts_data,
          scale.data = new("matrix"),
          key = paste0(tolower(substr(assay_name, 1, 3)), "_"),
          var.features = character(0),
          meta.features = data.frame(row.names = rownames(counts_data)),
          misc = list()
        )
      }
      
      obj[[assay_name]] <- legacy_assay
      cat(sprintf("    Converted. New class: %s\n", class(obj[[assay_name]])[1]))
    }
  }
  
  if (!is.null(meta_data)) {
    common_cells <- intersect(colnames(obj), rownames(meta_data))
    if (length(common_cells) > 0) {
      for (col_name in colnames(meta_data)) {
        vals <- rep(NA, ncol(obj))
        names(vals) <- colnames(obj)
        idx <- match(common_cells, rownames(meta_data))
        vals[common_cells] <- as.character(meta_data[[col_name]][idx])
        obj@meta.data[[col_name]] <- vals
      }
    }
  }
  
  DefaultAssay(obj) <- assay_name
  cat(sprintf("    Final: %d genes x %d cells, class: %s\n",
              nrow(obj), ncol(obj), class(obj[[assay_name]])[1]))
  return(obj)
}

verify_celltrek_compatible <- function(obj, obj_name = "object") {
  assay_class <- class(obj[["RNA"]])[1]
  cat(sprintf("  %s: RNA assay class = %s\n", obj_name, assay_class))
  ok <- TRUE
  if (assay_class == "Assay5") {
    cat(sprintf("  %s: WARNING - still Assay5\n", obj_name))
    ok <- FALSE
  }
  tryCatch({
    cts <- safe_get_assay_data(obj, "counts")
    cat(sprintf("  %s: counts accessible (%d x %d) ✓\n", obj_name, nrow(cts), ncol(cts)))
  }, error = function(e) {
    cat(sprintf("  %s: counts access FAILED: %s\n", obj_name, e$message))
    ok <<- FALSE
  })
  return(ok)
}

smart_transpose <- function(mat, reference_genes = NULL, data_type = "Data") {
  cat(sprintf("  %s format detection:\n", data_type))
  cat(sprintf("    Input dimensions: %d x %d\n", nrow(mat), ncol(mat)))
  
  row_names <- rownames(mat)
  col_names <- colnames(mat)
  
  if (!is.null(reference_genes) && length(reference_genes) > 0) {
    row_gene_overlap <- length(intersect(row_names, reference_genes))
    col_gene_overlap <- length(intersect(col_names, reference_genes))
    cat(sprintf("    Reference gene overlap - Rows: %d, Cols: %d\n",
                row_gene_overlap, col_gene_overlap))
    if (row_gene_overlap > 100 && row_gene_overlap > col_gene_overlap) {
      cat("    -> Rows are genes (correct)\n")
      return(mat)
    } else if (col_gene_overlap > 100 && col_gene_overlap > row_gene_overlap) {
      cat("    -> Columns are genes, transposing...\n")
      return(t(mat))
    }
  }
  
  barcode_patterns <- c("^[ACGT]{16}-\\d+$", "^spot_\\d+_\\d+$", "^\\d+_\\d+$",
                         "^cell_\\d+$", "^Cell\\d+$")
  count_barcode_matches <- function(names_vec) {
    if (is.null(names_vec)) return(0)
    test_names <- head(names_vec, 50)
    sum(sapply(test_names, function(n) {
      any(sapply(barcode_patterns, function(p) grepl(p, n, ignore.case = TRUE)))
    }))
  }
  
  row_barcode <- count_barcode_matches(row_names)
  col_barcode <- count_barcode_matches(col_names)
  
  if (row_barcode > col_barcode && row_barcode >= 5) {
    cat("    -> Rows are barcodes, transposing...\n")
    return(t(mat))
  } else if (col_barcode > row_barcode && col_barcode >= 5) {
    cat("    -> Columns are barcodes (correct)\n")
    return(mat)
  }
  
  if (nrow(mat) > ncol(mat)) {
    cat("    -> More rows than columns, assuming rows are samples, transposing...\n")
    return(t(mat))
  } else {
    cat("    -> Assuming columns are samples\n")
    return(mat)
  }
}

extract_coords_from_names <- function(spot_names) {
  pattern <- "^spot_?(\\d+)_(\\d+)$"
  matches <- regmatches(spot_names, regexec(pattern, spot_names, ignore.case = TRUE))
  valid_matches <- sapply(matches, length) == 3
  if (sum(valid_matches) < length(spot_names) * 0.5) {
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
  coords <- data.frame(barcode = spot_names, x = NA_real_, y = NA_real_,
                        stringsAsFactors = FALSE)
  for (i in which(valid_matches)) {
    coords$x[i] <- as.numeric(matches[[i]][2])
    coords$y[i] <- as.numeric(matches[[i]][3])
  }
  coords <- coords[complete.cases(coords), ]
  if (nrow(coords) == 0) return(NULL)
  return(coords)
}

load_coordinates <- function(filepath) {
  first_line <- readLines(filepath, n = 1)
  
  # Smarter header detection: only TRUE if standard column names are actually present
  has_header <- any(grepl("barcode|coord|\\bx\\b|\\by\\b|row|col", tolower(first_line)))
  
  coords_df <- fread(filepath, header = has_header)
  
  if (ncol(coords_df) == 6 && is.numeric(coords_df[[2]]) && all(coords_df[[2]] %in% c(0, 1))) {
    cat("    Detected standard 6-column Visium format\n")
    coords <- data.frame(barcode = coords_df[[1]], x = coords_df[[4]], y = coords_df[[3]])
  } else if (ncol(coords_df) >= 3) {
    col_lower <- tolower(colnames(coords_df))
    if ("x" %in% col_lower && "y" %in% col_lower) {
      x_idx <- which(col_lower == "x")[1]
      y_idx <- which(col_lower == "y")[1]
      bc_idx <- if ("barcode" %in% col_lower) which(col_lower == "barcode")[1] else 1
      coords <- data.frame(barcode = coords_df[[bc_idx]], x = coords_df[[x_idx]], y = coords_df[[y_idx]])
    } else {
      coords <- data.frame(barcode = coords_df[[1]], x = coords_df[[2]], y = coords_df[[3]])
    }
  } else {
    stop("Coordinate file must have at least 3 columns")
  }
  coords$x <- as.numeric(coords$x)
  coords$y <- as.numeric(coords$y)
  cat(sprintf("    Extracted %d coordinates. X range: %.1f-%.1f, Y range: %.1f-%.1f\n", 
              nrow(coords), min(coords$x), max(coords$x), min(coords$y), max(coords$y)))
  return(coords)
}

calculate_hex_radius <- function(coords, orientation = 0, override_radius = NULL) {
  # For pointy-top hexagons:
  #   - Same-row center distance = R × sqrt(3)
  #   - Adjacent row vertical distance = R × 1.5
  #   - Every-other-row vertical distance = R × 3
  
  if (!is.null(override_radius)) {
    return(override_radius)
  }
  
  if (nrow(coords) < 2) return(100)
  
  # Group by Y to find same-row X spacing
  xy_matrix <- as.matrix(coords[, c("x", "y")])
  nn <- FNN::get.knn(xy_matrix, k = min(6, nrow(xy_matrix) - 1))
  
  true_nn1 <- apply(nn$nn.dist, 1, function(x) {
    v <- x[x > 1e-5]
    if (length(v) > 0) min(v) else NA
  })
  median_nn_dist <- median(true_nn1, na.rm = TRUE)
  if (is.na(median_nn_dist)) median_nn_dist <- 100
  
  tol <- median_nn_dist * 0.3
  y_rounded <- round(coords$y / tol) * tol
  
  # Calculate same-row X spacing
  same_row_dists <- c()
  for (yval in unique(y_rounded)) {
    row_x <- sort(coords$x[y_rounded == yval])
    if (length(row_x) >= 2) {
      diffs <- diff(row_x)
      diffs <- diffs[diffs > median_nn_dist * 0.5]
      same_row_dists <- c(same_row_dists, diffs)
    }
  }
  
  if (length(same_row_dists) > 5) {
    col_spacing <- median(same_row_dists)  # This is R × sqrt(3)
    
    # R = col_spacing / sqrt(3)
    radius <- col_spacing / sqrt(3)
    
    cat(sprintf("    Hex radius: %.4f (from col_spacing=%.4f)\n", radius, col_spacing))
    cat(sprintf("    Verification: R=%.4f\n", radius))
    cat(sprintf("      Same-row distance should be R*sqrt(3) = %.4f (actual: %.4f)\n",
                radius * sqrt(3), col_spacing))
    cat(sprintf("      Adjacent-row Y distance should be R*1.5 = %.4f\n", radius * 1.5))
    cat(sprintf("      Every-other-row Y distance should be R*3 = %.4f\n", radius * 3))
    
  } else {
    # Fallback: nearest neighbor is likely the diagonal neighbor
    # For pointy-top hex, diagonal distance = R × sqrt(3) (same as horizontal)
    radius <- median_nn_dist / sqrt(3)
    cat(sprintf("    Hex radius (fallback): %.4f (from nn_dist=%.4f)\n", 
                radius, median_nn_dist))
  }
  
  return(radius)
}

detect_and_fix_hex_grid <- function(coords) {
  cat("  Detecting grid geometry...\n")
  
  # 1. Get robust median nearest neighbor distance
  xy_mat <- as.matrix(coords[, c("x", "y")])
  nn <- FNN::get.knn(xy_mat, k = 4)
  true_nn1 <- apply(nn$nn.dist, 1, function(x) {
    v <- x[x > 1e-5]
    if (length(v) > 0) min(v) else NA
  })
  median_dist <- median(true_nn1, na.rm = TRUE)
  
  if (is.na(median_dist)) return(list(coords = coords, is_hex = FALSE, hex_radius = NULL))
  
  # 2. Group coordinates using a tolerance based on neighbor distance
  tol <- median_dist * 0.1
  y_rounded <- round(coords$y / tol) * tol
  y_vals <- sort(unique(y_rounded))
  
  valid_y_diffs <- diff(y_vals)[diff(y_vals) > tol]
  
  if (length(valid_y_diffs) == 0 || length(y_vals) < 3) {
    cat(sprintf("    Continuous spatial coordinates detected (Radius: %.2f)\n", median_dist / sqrt(3)))
    return(list(coords = coords, is_hex = TRUE, hex_radius = median_dist / sqrt(3)))
  }
  y_spacing <- median(valid_y_diffs)
  
  # FIX: Calculate X-spacing based ONLY on spots in the SAME ROW
  same_row_x_diffs <- c()
  for (yval in y_vals) {
    row_x <- sort(coords$x[abs(coords$y - yval) < tol])
    if (length(row_x) >= 2) {
      same_row_x_diffs <- c(same_row_x_diffs, diff(row_x))
    }
  }
  same_row_x_diffs <- same_row_x_diffs[same_row_x_diffs > tol]
  
  if (length(same_row_x_diffs) == 0) {
    return(list(coords = coords, is_hex = TRUE, hex_radius = median_dist / sqrt(3)))
  }
  x_spacing <- median(same_row_x_diffs)
  
  # Check for existing hex offset
  row_assignments <- match(y_rounded, y_vals)
  even_rows <- which(row_assignments %% 2 == 0)
  odd_rows <- which(row_assignments %% 2 == 1)
  
  if (length(even_rows) > 0 && length(odd_rows) > 0) {
    even_min_x <- tapply(coords$x[even_rows], row_assignments[even_rows], min)
    odd_min_x <- tapply(coords$x[odd_rows], row_assignments[odd_rows], min)
    offset_diff <- abs(median(even_min_x) - median(odd_min_x))
    
    if (offset_diff > x_spacing * 0.2) {
      # ================================================================
      # Native hex offset detected.
      # For pointy-top tessellation the geometry MUST satisfy:
      #   x_spacing (same-row center-to-center) = R * sqrt(3)
      #   y_spacing (adjacent-row center-to-center) = R * 1.5
      #   every-other-row distance = R * 3.0
      #   → R = x_spacing / sqrt(3)
      #
      # If the actual y_spacing doesn't match 1.5*R, we MUST rescale Y
      # so that hexagons tile without vertical gaps.
      # ================================================================
      hex_r <- x_spacing / sqrt(3)
      expected_y <- hex_r * 1.5
      y_mismatch_pct <- abs(y_spacing - expected_y) / y_spacing * 100
      
      cat(sprintf("    Native hex offset detected\n"))
      cat(sprintf("    x_spacing (same-row)  = %.4f\n", x_spacing))
      cat(sprintf("    y_spacing (row-row)   = %.4f\n", y_spacing))
      cat(sprintf("    R = x_spacing/sqrt(3) = %.4f\n", hex_r))
      cat(sprintf("    Expected y_spacing = 1.5*R = %.4f\n", expected_y))
      cat(sprintf("    Y mismatch: %.1f%%\n", y_mismatch_pct))
      
      # Check if y_spacing ≈ 3R (every-other-row pattern, double the expected)
      every_other_ratio <- y_spacing / (hex_r * 3.0)
      cat(sprintf("    y_spacing / (3R) = %.3f (1.0 means every-other-row pattern)\n",
                  every_other_ratio))
      
      coords_fixed <- coords
      
      if (y_mismatch_pct > 5) {
        cat(sprintf("    Correcting Y-spacing: %.4f → %.4f\n", y_spacing, expected_y))
        y_scale <- expected_y / y_spacing
        y_center <- mean(range(coords$y))
        coords_fixed$y <- y_center + (coords$y - y_center) * y_scale
        
        # Verify correction
        y_rounded_new <- round(coords_fixed$y / (expected_y * 0.1)) * (expected_y * 0.1)
        y_vals_new <- sort(unique(y_rounded_new))
        new_y_diffs <- diff(y_vals_new)
        new_y_diffs <- new_y_diffs[new_y_diffs > expected_y * 0.5]
        if (length(new_y_diffs) > 0) {
          cat(sprintf("    After correction: median y_spacing = %.4f (target: %.4f)\n",
                      median(new_y_diffs), expected_y))
        }
      } else {
        cat(sprintf("    Y-spacing OK (within 5%%), no correction needed\n"))
      }
      
      cat(sprintf("    Final: R=%.4f, hex_width=R*sqrt(3)=%.4f, hex_height=2R=%.4f\n",
                  hex_r, hex_r * sqrt(3), hex_r * 2))
      cat(sprintf("    Tiling check: same-row gap = x_spacing - hex_width = %.4f (should be ~0)\n",
                  x_spacing - hex_r * sqrt(3)))
      cat(sprintf("    Tiling check: row overlap = 2R - y_spacing_corrected = %.4f (should be R/2 = %.4f)\n",
                  hex_r * 2 - expected_y, hex_r * 0.5))
      
      return(list(coords = coords_fixed, is_hex = TRUE, hex_radius = hex_r))
    }
  }
  
  # Only shift if it is a rigid, synthetic square grid
  cat("    Synthetic square grid detected — applying pointy-top hex row offset...\n")
  coords_fixed <- coords
  for (i in seq_along(y_vals)) {
    row_mask <- abs(coords$y - y_vals[i]) < tol
    # Shift ODD-indexed rows (i=2,4,6... i.e., Row 1, Row 3, Row 5...)
    # so Row 0 stays put, Row 1 shifts right, Row 2 stays put, etc.
    if (i %% 2 == 0) {
      coords_fixed$x[row_mask] <- coords$x[row_mask] + x_spacing / 2
    }
  }
  
  # Correct Y-spacing for pointy-top: ideal = x_spacing * sqrt(3) / 2
  ideal_y_spacing <- x_spacing * sqrt(3) / 2
  y_error <- abs(y_spacing - ideal_y_spacing) / y_spacing
  if (y_error > 0.05) {
    cat(sprintf("    Correcting Y-spacing: %.1f → %.1f (error was %.1f%%)\n",
                y_spacing, ideal_y_spacing, y_error * 100))
    y_scale <- ideal_y_spacing / y_spacing
    y_center <- mean(range(coords_fixed$y))
    coords_fixed$y <- y_center + (coords_fixed$y - y_center) * y_scale
  }
  
  # Pointy-top circumradius: R = x_spacing / sqrt(3)
  hex_r <- x_spacing / sqrt(3)
  cat(sprintf("    Pointy-top hex: R=%.4f, x_spacing=%.2f\n", hex_r, x_spacing))
  cat(sprintf("    Verification: R*sqrt(3)=%.4f should ≈ x_spacing=%.4f\n",
              hex_r * sqrt(3), x_spacing))
  
  return(list(coords = coords_fixed, is_hex = TRUE, hex_radius = hex_r))
}

create_hexagon <- function(x, y, radius, rotation = 0) {
  # For pointy-top hexagons, the first vertex is at the TOP (90 degrees = pi/2).
  # The 6 vertices are spaced 60 degrees apart starting from pi/2.
  # Additional rotation parameter allows switching to flat-top (rotation=30).
  rot_rad <- rotation * pi / 180
  # Start at pi/2 (top vertex) for pointy-top orientation
  angles <- seq(0, 2 * pi, length.out = 7)[1:6] + pi / 2 + rot_rad
  data.frame(x = x + radius * cos(angles), y = y + radius * sin(angles))
}

create_all_hexagons <- function(coords, radius, rotation = 0) {
  hex_list <- lapply(1:nrow(coords), function(i) {
    hex <- create_hexagon(coords$x[i], coords$y[i], radius, rotation)
    hex$id <- coords$barcode[i]
    hex$spot_idx <- i
    hex
  })
  do.call(rbind, hex_list)
}

create_background_hexagons <- function(coords_full, matched_mask, radius, rotation = 0) {
  unmatched_idx <- which(!matched_mask)
  if (length(unmatched_idx) == 0) return(NULL)
  unmatched_coords <- coords_full[unmatched_idx, ]
  hex_list <- lapply(1:nrow(unmatched_coords), function(i) {
    hex <- create_hexagon(unmatched_coords$x[i], unmatched_coords$y[i], radius, rotation)
    hex$id <- unmatched_coords$barcode[i]
    hex$spot_idx <- unmatched_idx[i]
    hex
  })
  do.call(rbind, hex_list)
}

#' Calculate plot dimensions that match the spatial data aspect ratio
#' @param coords data.frame with x, y columns (after rotation)
#' @param plot_width desired width of the spatial area in inches
#' @param legend_width extra width for the legend in inches
#' @param margin extra margin in inches
#' @return list with width and height in inches
calculate_plot_dimensions <- function(coords, plot_width = 8, legend_width = 2.5, margin = 1.0) {
  x_range <- diff(range(coords$x, na.rm = TRUE))
  y_range <- diff(range(coords$y, na.rm = TRUE))
  
  if (x_range < 1e-6 || y_range < 1e-6) {
    return(list(width = plot_width + legend_width, height = plot_width + margin))
  }
  
  aspect <- y_range / x_range
  plot_height <- plot_width * aspect
  
  # Clamp to reasonable dimensions
  plot_height <- max(4, min(plot_height, 20))
  
  total_width <- plot_width + legend_width
  total_height <- plot_height + margin
  
  return(list(width = total_width, height = total_height))
}

# ==============================================================================
# CELL STATE ANALYSIS FUNCTIONS
# ==============================================================================

extract_cell_states <- function(celltrek_result, sc_obj, celltype_col, n_dims = 10) {
  cat("\n=== Extracting Cell States ===\n")
  
  if (!"pca" %in% Reductions(celltrek_result)) {
    celltrek_result <- NormalizeData(celltrek_result, verbose = FALSE)
    celltrek_result <- FindVariableFeatures(celltrek_result, verbose = FALSE)
    celltrek_result <- ScaleData(celltrek_result, verbose = FALSE)
    celltrek_result <- RunPCA(celltrek_result, verbose = FALSE)
  }
  
  pca_dims <- min(n_dims, ncol(Embeddings(celltrek_result, "pca")))
  pca_embed <- Embeddings(celltrek_result, "pca")[, 1:pca_dims, drop = FALSE]
  
  ct_col <- NULL
  for (candidate in c(celltype_col, "CellType", "celltype", "cell_type", "type", "cluster")) {
    if (candidate %in% colnames(celltrek_result@meta.data)) {
      ct_col <- candidate
      break
    }
  }
  
  if (!is.null(ct_col)) {
    cell_types <- celltrek_result@meta.data[[ct_col]]
  } else {
    cell_types <- rep("Unknown", ncol(celltrek_result))
  }
  
  unique_types <- unique(cell_types[!is.na(cell_types)])
  state_scores <- list()
  
  for (ct in unique_types) {
    ct_mask <- cell_types == ct & !is.na(cell_types)
    if (sum(ct_mask) > 10) {
      ct_pca <- pca_embed[ct_mask, , drop = FALSE]
      state_scores[[ct]] <- data.frame(
        cell_id = rownames(pca_embed)[ct_mask],
        state_score = scale(ct_pca[, 1])[, 1],
        state_pc1 = ct_pca[, 1],
        state_pc2 = if (ncol(ct_pca) > 1) ct_pca[, 2] else 0,
        celltype = ct,
        stringsAsFactors = FALSE
      )
    }
  }
  
  if (length(state_scores) > 0) {
    state_df <- do.call(rbind, state_scores)
    rownames(state_df) <- state_df$cell_id
  } else {
    state_df <- data.frame(
      cell_id = rownames(pca_embed),
      state_score = 0,
      state_pc1 = pca_embed[, 1],
      state_pc2 = if (ncol(pca_embed) > 1) pca_embed[, 2] else 0,
      celltype = cell_types,
      stringsAsFactors = FALSE
    )
    rownames(state_df) <- state_df$cell_id
  }
  
  cat(sprintf("  Extracted states for %d cells across %d cell types\n",
              nrow(state_df), length(unique_types)))
  return(list(states = state_df, pca_embed = pca_embed, cell_types = cell_types))
}

calculate_state_continuum <- function(celltrek_result, celltype_col, resolution = 0.5) {
  cat("\n=== Calculating Cell State Continuum ===\n")
  
  if (!"umap" %in% Reductions(celltrek_result)) {
    if (!"pca" %in% Reductions(celltrek_result)) {
      celltrek_result <- NormalizeData(celltrek_result, verbose = FALSE)
      celltrek_result <- FindVariableFeatures(celltrek_result, verbose = FALSE)
      celltrek_result <- ScaleData(celltrek_result, verbose = FALSE)
      celltrek_result <- RunPCA(celltrek_result, verbose = FALSE)
    }
    n_dims <- min(30, ncol(Embeddings(celltrek_result, "pca")))
    celltrek_result <- RunUMAP(celltrek_result, dims = 1:n_dims, verbose = FALSE)
  }
  
  umap_embed <- Embeddings(celltrek_result, "umap")
  
  ct_col <- NULL
  for (candidate in c(celltype_col, "CellType", "celltype", "cell_type", "type", "cluster")) {
    if (candidate %in% colnames(celltrek_result@meta.data)) {
      ct_col <- candidate
      break
    }
  }
  
  if (!is.null(ct_col)) {
    cell_types <- celltrek_result@meta.data[[ct_col]]
  } else {
    cell_types <- rep("Unknown", ncol(celltrek_result))
  }
  
  unique_types <- unique(cell_types[!is.na(cell_types)])
  state_clusters <- rep(NA_character_, ncol(celltrek_result))
  names(state_clusters) <- colnames(celltrek_result)
  
  for (ct in unique_types) {
    ct_cells <- which(cell_types == ct & !is.na(cell_types))
    if (length(ct_cells) > 20) {
      ct_umap <- umap_embed[ct_cells, , drop = FALSE]
      if (length(ct_cells) > 50) {
        k_val <- min(5, nrow(ct_umap) - 1)
        if (k_val >= 1) {
          k_dist <- dbscan::kNNdist(ct_umap, k = k_val)
          eps_val <- quantile(k_dist, 0.9)
          db_result <- dbscan::dbscan(ct_umap, eps = eps_val, minPts = 5)
          sub_clusters <- db_result$cluster
          sub_clusters[sub_clusters == 0] <- max(sub_clusters) + 1
        } else {
          sub_clusters <- rep(1, length(ct_cells))
        }
      } else {
        n_clusters <- max(1, min(3, floor(length(ct_cells) / 10)))
        km_result <- kmeans(ct_umap, centers = n_clusters, nstart = 10)
        sub_clusters <- km_result$cluster
      }
      state_clusters[ct_cells] <- paste0(ct, "_state", sub_clusters)
    } else if (length(ct_cells) > 0) {
      state_clusters[ct_cells] <- paste0(ct, "_state1")
    }
  }
  
  cat(sprintf("  Identified %d cell states across %d cell types\n",
              length(unique(na.omit(state_clusters))), length(unique_types)))
  return(list(state_clusters = state_clusters, umap_embed = umap_embed,
              celltrek_result = celltrek_result))
}

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

plot_spatial_state_maps <- function(mapped_cells, state_data, coords, output_dir,
                                   hex_angle = 0, coords_full = NULL, matched_mask = NULL,
                                   sc_pca = NULL, st_pca = NULL,
                                   sc_celltypes = NULL, knn_k = 50) {
  cat("\n=== Generating Spatial State Maps (Spot-Centric) ===\n")
  
  OVERLAP_FACTOR <- 1.03
  hex_radius <- if (exists("detected_hex_radius", envir = .GlobalEnv) && 
                    !is.null(get("detected_hex_radius", envir = .GlobalEnv))) {
    get("detected_hex_radius", envir = .GlobalEnv) * OVERLAP_FACTOR
  } else {
    calculate_hex_radius(coords, orientation = hex_angle) * OVERLAP_FACTOR
  }
  
  # Rotation for display: 90 degrees left from previous orientation
  max_x_ref <- max(coords$x, na.rm = TRUE)
  max_y_ref <- max(coords$y, na.rm = TRUE)
  coords_rot <- data.frame(
    barcode = coords$barcode,
    x = coords$x,
    y = max_y_ref - coords$y,
    stringsAsFactors = FALSE
  )
  
  # Full tissue background (rotated)
  if (!is.null(coords_full)) {
    max_x_full <- max(coords_full$x, na.rm = TRUE)
    max_y_full <- max(coords_full$y, na.rm = TRUE)
    coords_full_rot <- data.frame(
      barcode = coords_full$barcode,
      x = coords_full$x,
      y = max_y_full - coords_full$y,
      stringsAsFactors = FALSE
    )
  } else {
    coords_full_rot <- coords_rot
  }
  
  # ================================================================
  # ALIGN st_pca rows with coords rows
  # st_pca rownames are "spot_<barcode>", coords$barcode are "<barcode>"
  # We must subset and reorder st_pca to match coords exactly
  # ================================================================
  use_spot_centric <- !is.null(sc_pca) && !is.null(st_pca) && !is.null(sc_celltypes)
  
  if (use_spot_centric) {
    # Strip "spot_" prefix from st_pca rownames to get clean barcodes
    st_pca_barcodes <- gsub("^spot_", "", rownames(st_pca))
    
    # Find which st_pca rows match which coords rows
    st_to_coord_idx <- match(coords$barcode, st_pca_barcodes)
    valid_st_match <- !is.na(st_to_coord_idx)
    
    cat(sprintf("  Aligning ST PCA to coords: %d/%d spots matched\n",
                sum(valid_st_match), nrow(coords)))
    
    if (sum(valid_st_match) == 0) {
      cat("  WARNING: No ST PCA rows match coordinate barcodes, falling back\n")
      use_spot_centric <- FALSE
    } else {
      # Subset st_pca to only matched spots, in coords order
      st_pca_aligned <- st_pca[st_to_coord_idx[valid_st_match], , drop = FALSE]
      
      # Subset coords to only matched spots
      coords_aligned <- coords[valid_st_match, ]
      coords_rot_aligned <- coords_rot[valid_st_match, ]
      
      cat(sprintf("  Aligned: %d spots with both PCA and coordinates\n",
                  nrow(st_pca_aligned)))
    }
  }
  
  # Merge state_data with cell type info
  state_df <- state_data$states
  cell_types <- unique(state_df$celltype[!is.na(state_df$celltype)])
  cat(sprintf("  Cell types with state data: %d\n", length(cell_types)))
  
  plot_list <- list()
  
  for (ct in cell_types) {
    ct_state <- state_df[state_df$celltype == ct & !is.na(state_df$state_score), ]
    if (nrow(ct_state) < 5) next
    
    if (use_spot_centric) {
      # ----- SPOT-CENTRIC: every aligned spot gets a score -----
      
      # Find SC cells of this type
      ct_sc_mask <- sc_celltypes == ct & !is.na(sc_celltypes)
      ct_sc_names <- names(sc_celltypes)[ct_sc_mask]
      
      # Match to state scores
      ct_state_match <- match(ct_sc_names, ct_state$cell_id)
      valid_state <- !is.na(ct_state_match)
      
      if (sum(valid_state) < 5) {
        cat(sprintf("    %s: too few SC cells with state scores (%d), skipping\n",
                    ct, sum(valid_state)))
        next
      }
      
      ct_sc_pca <- sc_pca[ct_sc_mask, , drop = FALSE][valid_state, , drop = FALSE]
      ct_scores <- ct_state$state_score[ct_state_match[valid_state]]
      
      # For each aligned ST spot, find K nearest SC neighbors of this cell type
      k_use <- min(knn_k, nrow(ct_sc_pca))
      nn_ct <- FNN::get.knnx(ct_sc_pca, st_pca_aligned, k = k_use)
      
      # Weighted average state score per spot
      n_aligned <- nrow(st_pca_aligned)
      spot_scores <- numeric(n_aligned)
      
      for (i in seq_len(n_aligned)) {
        nn_idx <- nn_ct$nn.index[i, ]
        nn_dist <- nn_ct$nn.dist[i, ]
        nn_scores <- ct_scores[nn_idx]
        
        nn_dist_safe <- pmax(nn_dist, 1e-10)
        weights <- 1 / nn_dist_safe
        weights <- weights / sum(weights)
        
        spot_scores[i] <- sum(weights * nn_scores)
      }
      
      # Now expand back to ALL coords rows (fill unmatched with 0)
      full_scores <- rep(0, nrow(coords))
      full_scores[valid_st_match] <- spot_scores
      
      spot_state_full <- data.frame(
        barcode = coords_rot$barcode,
        x = coords_rot$x,
        y = coords_rot$y,
        mean_state = full_scores,
        n_cells = ifelse(valid_st_match, k_use, 0),
        stringsAsFactors = FALSE
      )
      
      spots_with_signal <- sum(valid_st_match)
      
    } else {
      # ----- FALLBACK: aggregate from mapped_cells -----
      
      spot_coords_mat <- as.matrix(coords[, c("x", "y")])
      ct_mapped <- mapped_cells[mapped_cells$celltype == ct & 
                                  !is.na(mapped_cells$coord_x), ]
      
      if (nrow(ct_mapped) < 5) next
      
      cell_coords_mat <- as.matrix(ct_mapped[, c("coord_x", "coord_y")])
      nn_spot <- FNN::get.knnx(spot_coords_mat, cell_coords_mat, k = 1)
      ct_mapped$nearest_spot <- coords$barcode[nn_spot$nn.index[, 1]]
      
      ct_mapped_state <- merge(ct_mapped, ct_state[, c("cell_id", "state_score")],
                                by.x = "cell_id", by.y = "cell_id", all.x = TRUE)
      
      spot_state <- ct_mapped_state %>%
        filter(!is.na(state_score)) %>%
        group_by(nearest_spot) %>%
        summarise(mean_state = mean(state_score, na.rm = TRUE),
                  n_cells = n(), .groups = "drop")
      
      spot_state_full <- merge(
        coords_rot,
        spot_state,
        by.x = "barcode", by.y = "nearest_spot",
        all.x = TRUE
      )
      spot_state_full$mean_state[is.na(spot_state_full$mean_state)] <- 0
      spot_state_full$n_cells[is.na(spot_state_full$n_cells)] <- 0
      
      spots_with_signal <- sum(spot_state_full$n_cells > 0)
    }
    
    # Create hexagons for ALL spots
    hex_polys <- create_all_hexagons(spot_state_full, hex_radius, rotation = hex_angle)
    hex_polys$value <- spot_state_full$mean_state[hex_polys$spot_idx]
    
    max_abs <- max(abs(spot_state_full$mean_state), na.rm = TRUE)
    if (max_abs < 1e-8) max_abs <- 1
    
    p <- ggplot() +
      geom_polygon(data = hex_polys, 
                   aes(x = x, y = y, group = id, fill = value),
                   color = NA, linewidth = 0) +
      scale_fill_gradient2(low = "#3b4cc0",      # Coolwarm Blue
                           mid = "#dddddd",      # Coolwarm neutral grey/white
                           high = "#b40426",     # Coolwarm Red
                           midpoint = 0,
                           limits = c(-max_abs, max_abs),
                           name = "State\nScore") +
      coord_fixed() + theme_void() +
      theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
            legend.position = "right", legend.key.size = unit(0.4, "cm"),
            legend.title = element_text(size = 9), 
            legend.text = element_text(size = 8)) +
      ggtitle(paste0(ct, " — State Score (",
                     spots_with_signal, "/", nrow(spot_state_full), " spots)"))
    
    plot_list[[ct]] <- p
    ct_safe <- gsub("[^A-Za-z0-9]", "_", ct)
    dims <- calculate_plot_dimensions(coords_rot)
    ggsave(file.path(output_dir, paste0("spatial_state_", ct_safe, ".png")),
           p, width = dims$width, height = dims$height, dpi = 300)
  }
  
  if (length(plot_list) > 0) {
    n_types <- length(plot_list)
    n_cols <- min(4, n_types)
    n_rows <- ceiling(n_types / n_cols)
    cell_dims <- calculate_plot_dimensions(coords_rot, plot_width = 4, legend_width = 1, margin = 0.5)
    png(file.path(output_dir, "spatial_state_all_types.png"),
        width = n_cols * cell_dims$width, height = n_rows * cell_dims$height, 
        units = "in", res = 300)
    grid.arrange(grobs = plot_list, ncol = n_cols)
    dev.off()
    cat(sprintf("  Saved state maps for %d cell types\n", length(plot_list)))
  }
  return(plot_list)
}

plot_continuum_maps <- function(celltrek_result, state_continuum, mapped_cells,
                                output_dir, celltype_col) {
  cat("\n=== Generating Continuum Maps ===\n")
  
  ct_col <- NULL
  for (candidate in c(celltype_col, "CellType", "celltype", "cell_type")) {
    if (candidate %in% colnames(celltrek_result@meta.data)) {
      ct_col <- candidate
      break
    }
  }
  
  if (!is.null(ct_col)) {
    cell_types_vec <- celltrek_result@meta.data[[ct_col]]
  } else {
    cell_types_vec <- rep("Unknown", ncol(celltrek_result))
  }
  
  umap_df <- data.frame(
    UMAP1 = state_continuum$umap_embed[, 1],
    UMAP2 = state_continuum$umap_embed[, 2],
    celltype = cell_types_vec,
    state_cluster = state_continuum$state_clusters,
    stringsAsFactors = FALSE
  )
  
  unique_types <- sort(unique(umap_df$celltype[!is.na(umap_df$celltype)]))
  n_types <- length(unique_types)
  if (n_types <= 20) {
    type_colors <- scales::hue_pal()(n_types)
  } else {
    type_colors <- colorRampPalette(brewer.pal(12, "Set3"))(n_types)
  }
  names(type_colors) <- unique_types
  
  p1 <- ggplot(umap_df[!is.na(umap_df$celltype), ],
               aes(x = UMAP1, y = UMAP2, color = celltype)) +
    geom_point(size = 0.5, alpha = 0.6) +
    scale_color_manual(values = type_colors, name = "Cell Type") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
          legend.position = "right") +
    guides(color = guide_legend(ncol = 1, override.aes = list(size = 3))) +
    ggtitle("Cell Type Continuum (UMAP)")
  ggsave(file.path(output_dir, "continuum_celltype_umap.png"),
         p1, width = 15, height = 10, dpi = 300)
  
  umap_df_state <- umap_df[!is.na(umap_df$state_cluster), ]
  if (nrow(umap_df_state) > 0) {
    p2 <- ggplot(umap_df_state, aes(x = UMAP1, y = UMAP2, color = state_cluster)) +
      geom_point(size = 0.5, alpha = 0.6) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
            legend.position = "right", legend.text = element_text(size = 6)) +
      guides(color = guide_legend(ncol = 2, override.aes = list(size = 2))) +
      ggtitle("Cell State Clusters (UMAP)")
    ggsave(file.path(output_dir, "continuum_state_clusters.png"),
           p2, width = 12, height = 10, dpi = 300)
  }
  
  if (nrow(mapped_cells) > 0) {
    common_cells <- intersect(mapped_cells$cell_id, rownames(state_continuum$umap_embed))
    if (length(common_cells) > 0) {
      spatial_continuum <- data.frame(
        cell_id = common_cells,
        coord_x = mapped_cells$coord_x[match(common_cells, mapped_cells$cell_id)],
        coord_y = mapped_cells$coord_y[match(common_cells, mapped_cells$cell_id)],
        UMAP1 = state_continuum$umap_embed[common_cells, 1],
        UMAP2 = state_continuum$umap_embed[common_cells, 2],
        stringsAsFactors = FALSE
      )
      spatial_continuum <- spatial_continuum[!is.na(spatial_continuum$coord_x), ]
      
      if (nrow(spatial_continuum) > 0) {
        max_x <- max(spatial_continuum$coord_x, na.rm = TRUE)
        max_y <- max(spatial_continuum$coord_y, na.rm = TRUE)
        spatial_continuum$x_rot <- spatial_continuum$coord_x
        spatial_continuum$y_rot <- max_y - spatial_continuum$coord_y
        
        p3 <- ggplot(spatial_continuum, aes(x = x_rot, y = y_rot, color = UMAP1)) +
          geom_point(size = 1, alpha = 0.7) +
          scale_color_viridis(option = "viridis", name = "Continuum\n(UMAP1)") +
          coord_fixed() + theme_void() +
          theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
                legend.position = "right") +
          ggtitle("Spatial Cell State Continuum")
        cont_dims <- calculate_plot_dimensions(
          data.frame(x = spatial_continuum$x_rot, y = spatial_continuum$y_rot))
        ggsave(file.path(output_dir, "continuum_spatial_umap1.png"),
               p3, width = cont_dims$width, height = cont_dims$height, dpi = 300)
        
        p4 <- ggplot(spatial_continuum, aes(x = x_rot, y = y_rot, color = UMAP2)) +
          geom_point(size = 1, alpha = 0.7) +
          scale_color_viridis(option = "magma", name = "Continuum\n(UMAP2)") +
          coord_fixed() + theme_void() +
          theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
                legend.position = "right") +
          ggtitle("Spatial Cell State Continuum (Secondary Axis)")
        ggsave(file.path(output_dir, "continuum_spatial_umap2.png"),
               p4, width = cont_dims$width, height = cont_dims$height, dpi = 300)
      }
    }
  }
  
  for (ct in unique_types) {
    ct_df <- umap_df[umap_df$celltype == ct & !is.na(umap_df$celltype), ]
    if (nrow(ct_df) < 20) next
    p_ct <- ggplot(ct_df, aes(x = UMAP1, y = UMAP2)) +
      geom_point(size = 0.8, alpha = 0.5, color = "grey50") +
      stat_density_2d(aes(fill = after_stat(level)), geom = "polygon", alpha = 0.5) +
      scale_fill_viridis(option = "plasma", name = "Density") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
      ggtitle(paste0(ct, " State Continuum"))
    ct_safe <- gsub("[^A-Za-z0-9]", "_", ct)
    ggsave(file.path(output_dir, paste0("continuum_", ct_safe, ".png")),
           p_ct, width = 8, height = 6, dpi = 300)
  }
  cat(sprintf("  Saved continuum plots for %d cell types\n", length(unique_types)))
}

plot_spatial_intensity_maps <- function(proportions, coords, output_path, 
                                         hex_radius = NULL,
                                         coords_full = NULL, matched_mask = NULL,
                                         hex_angle = 0, min_prop = 0) {
  
  # Merge proportions with coordinates - PRESERVE COUNT MATRIX ORDER
  prop_df <- as.data.frame(proportions)
  prop_df$barcode <- rownames(prop_df)
  
  # Rotate coordinates 90 degrees clockwise for display consistency: (x, y) -> (y, -x)
  if (!is.null(coords_full)) {
    max_x_ref <- max(coords_full$x, na.rm = TRUE)
    max_y_ref <- max(coords_full$y, na.rm = TRUE)
    coords_full_rotated <- coords_full
    coords_full_rotated$x <- coords_full$x
    coords_full_rotated$y <- max_y_ref - coords_full$y
    coords_full <- coords_full_rotated
    
    coords_rotated <- coords
    coords_rotated$x <- coords$x
    coords_rotated$y <- max_y_ref - coords$y
    coords <- coords_rotated
  } else {
    max_x_ref <- max(coords$x, na.rm = TRUE)
    max_y_ref <- max(coords$y, na.rm = TRUE)
    coords_rotated <- coords
    coords_rotated$x <- coords$x
    coords_rotated$y <- max_y_ref - coords$y
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
    OVERLAP_FACTOR <- 1.03
    if (!is.null(coords_full)) {
      hex_radius <- calculate_hex_radius(coords_full, orientation = hex_angle) * OVERLAP_FACTOR
    } else {
      hex_radius <- calculate_hex_radius(merged, orientation = hex_angle) * OVERLAP_FACTOR
    }
    cat(sprintf("  Auto-calculated hex_radius with overlap: %.4f\n", hex_radius))
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
                            # CHANGED: match Seurat script style
                            fill = "#F0F0F0", color = NA, 
                            linewidth = 0, alpha = 0.5)
    }
    
    p <- p + geom_polygon(data = hex_data, 
                          aes(x = x, y = y, group = id, fill = value),
                          # CHANGED: color = NA, linewidth = 0 (match Seurat script)
                          color = NA,
                          linewidth = 0) +
      scale_fill_viridis(option = "plasma",
                         limits = c(min_prop, max(hex_data$value, na.rm = TRUE) + 1e-8), 
                         na.value = "#F0F0F0", 
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
  
  cell_dims <- calculate_plot_dimensions(coords, plot_width = 4, legend_width = 1, margin = 0.5)
  png(output_path, width = n_cols * cell_dims$width, height = n_rows * cell_dims$height, 
      units = "in", res = 300)
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
    max_y_ref <- max(coords_full$y, na.rm = TRUE)
    coords_full_rotated <- coords_full
    coords_full_rotated$x <- coords_full$x
    coords_full_rotated$y <- max_y_ref - coords_full$y
    coords_full <- coords_full_rotated
    
    coords_rotated <- coords
    coords_rotated$x <- coords$x
    coords_rotated$y <- max_y_ref - coords$y
    coords <- coords_rotated
  } else {
    max_x_ref <- max(coords$x, na.rm = TRUE)
    max_y_ref <- max(coords$y, na.rm = TRUE)
    coords_rotated <- coords
    coords_rotated$x <- coords$x
    coords_rotated$y <- max_y_ref - coords$y
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

  # 3. Determine types present in the dataset
  present_types <- sort(unique(merged$dominant_type))
  
  # 4. Construct the final color vector
  type_colors <- setNames(rep(NA, length(present_types)), present_types)
  
  # First pass: Assign specific colors
  for (ct in names(specific_mapping)) {
    if (ct %in% present_types) {
      type_colors[ct] <- specific_mapping[ct]
    }
  }
  
  # Second pass: Assign remaining types
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
  plot_data <- hex_polys
  plot_data$fill_group <- plot_data$dominant_type
  plot_data$alpha_val <- 1.0
  
  legend_breaks <- present_types
  
  if (has_unmatched) {
    bg_data <- bg_hex_polys
    bg_data$dominant_type <- "No count data" 
    bg_data$fill_group <- "No count data"
    bg_data$alpha_val <- 0.5 
    
    cols_to_keep <- c("x", "y", "id", "fill_group", "alpha_val")
    plot_data <- plot_data[, cols_to_keep]
    bg_data <- bg_data[, cols_to_keep]
    
    plot_data <- rbind(plot_data, bg_data)
    
    type_colors <- c(type_colors, "No count data" = "lightgrey")
    legend_breaks <- c(legend_breaks, "No count data")
  }
  
  plot_data$fill_group <- factor(plot_data$fill_group, levels = legend_breaks)
  
  p <- ggplot(plot_data, aes(x = x, y = y, group = id, fill = fill_group, alpha = alpha_val)) +
    # CHANGED: color = NA, linewidth = 0 (match Seurat script)
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
  
  dims <- calculate_plot_dimensions(coords, plot_width = 8, legend_width = 3.5, margin = 1.0)
  ggsave(output_path, p, width = dims$width, height = dims$height, dpi = 300)
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

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

# 1. LOAD SINGLE-CELL DATA
cat("\n=== Loading Single-Cell Data ===\n")
sc_mat <- load_matrix(args$sc_counts)
sc_meta <- read.csv(args$sc_labels, row.names = 1)

if (args$transpose_sc) {
  cat("  Manual transpose requested...\n")
  sc_mat <- t(sc_mat)
}

if (ncol(sc_mat) == nrow(sc_meta)) {
  cat("  Orientation: Genes x Cells (Correct)\n")
} else if (nrow(sc_mat) == nrow(sc_meta)) {
  cat("  Orientation: Cells x Genes (Transposing)\n")
  sc_mat <- t(sc_mat)
} else {
  sc_mat <- smart_transpose(sc_mat, reference_genes = NULL, data_type = "SC")
}

sc_gene_names <- rownames(sc_mat)
cat(sprintf("  SC dimensions: %d genes x %d cells\n", nrow(sc_mat), ncol(sc_mat)))

valid_cols <- c(args$celltype_col, 'cell_type', 'celltype', 'CellType', 'labels', 'cluster')
label_col <- colnames(sc_meta)[colnames(sc_meta) %in% valid_cols][1]
if (is.na(label_col)) label_col <- colnames(sc_meta)[1]
cat(sprintf("  Using cell type column: %s\n", label_col))

original_labels <- unique(sc_meta[[label_col]])
cat(sprintf("  Found %d cell types: %s\n", length(original_labels),
            paste(head(original_labels, 5), collapse = ", ")))

sc_obj <- create_celltrek_seurat(sc_mat, meta_data = sc_meta, assay_name = "RNA")
Idents(sc_obj) <- sc_obj@meta.data[[label_col]]
verify_celltrek_compatible(sc_obj, "SC object")

# 2. LOAD SPATIAL DATA
cat("\n=== Loading Spatial Data ===\n")
st_mat <- load_matrix(args$st_counts)

if (args$transpose_st) {
  cat("  Manual transpose requested...\n")
  st_mat <- t(st_mat)
} else {
  st_mat <- smart_transpose(st_mat, reference_genes = sc_gene_names, data_type = "ST")
}

cat(sprintf("  ST dimensions: %d genes x %d spots\n", nrow(st_mat), ncol(st_mat)))

coords_for_plots <- NULL
coords_full <- NULL
matched_mask <- NULL

if (!is.null(args$st_coords) && file.exists(args$st_coords)) {
  coords_full <- load_coordinates(args$st_coords)
  matched_mask <- coords_full$barcode %in% colnames(st_mat)
  coords_for_plots <- coords_full[matched_mask, ]
  cat(sprintf("  Matched coordinates for %d/%d spots\n", sum(matched_mask), nrow(coords_full)))
  cat("\n=== Hex Grid Detection and Correction ===\n")

if (!is.null(coords_full)) {
  cat("  Running hex grid detection on FULL tissue coordinates (single pass)...\n")
  hex_result_full <- detect_and_fix_hex_grid(coords_full)
  coords_full <- hex_result_full$coords
  detected_hex_radius <- hex_result_full$hex_radius
  
  # Re-derive matched subset FROM the corrected full coords
  matched_mask <- coords_full$barcode %in% colnames(st_mat)
  coords_for_plots <- coords_full[matched_mask, ]
  
  cat(sprintf("  Full coords: %d spots, Matched: %d spots\n",
              nrow(coords_full), nrow(coords_for_plots)))
  cat(sprintf("  Hex radius: %.4f\n", detected_hex_radius))
} else {
  cat("  Running hex grid detection on matched spots...\n")
  hex_result <- detect_and_fix_hex_grid(coords_for_plots)
  coords_for_plots <- hex_result$coords
  coords_full <- coords_for_plots
  matched_mask <- rep(TRUE, nrow(coords_full))
  detected_hex_radius <- hex_result$hex_radius
}

cat(sprintf("  Detected hex radius: %s\n", 
            ifelse(is.null(detected_hex_radius), "NULL (will auto-calculate)", 
                   sprintf("%.4f", detected_hex_radius))))

# Diagnostic: verify the final coordinate geometry
cat("\n  === Post-correction Geometry Verification ===\n")
if (!is.null(detected_hex_radius)) {
  R <- detected_hex_radius
  cat(sprintf("    R = %.4f\n", R))
  cat(sprintf("    Hex width  (vertex-to-vertex horizontal) = R*sqrt(3) = %.4f\n", R * sqrt(3)))
  cat(sprintf("    Hex height (vertex-to-vertex vertical)   = 2R        = %.4f\n", R * 2))
  cat(sprintf("    Same-row center distance should be       = R*sqrt(3) = %.4f\n", R * sqrt(3)))
  cat(sprintf("    Adjacent-row center distance should be   = 1.5*R     = %.4f\n", R * 1.5))
  cat(sprintf("    Every-other-row center distance should be= 3*R       = %.4f\n", R * 3.0))
  
  # Measure actual distances in corrected coords
  tol_verify <- R * 0.3
  y_round_v <- round(coords_for_plots$y / tol_verify) * tol_verify
  y_vals_v <- sort(unique(y_round_v))
  y_diffs_v <- diff(y_vals_v)
  y_diffs_v <- y_diffs_v[y_diffs_v > tol_verify * 0.5]
  if (length(y_diffs_v) > 0) {
    cat(sprintf("    Actual adjacent-row Y distance (median)  = %.4f\n", median(y_diffs_v)))
    cat(sprintf("    Ratio to 1.5R = %.3f (should be ~1.0)\n", median(y_diffs_v) / (R * 1.5)))
  }
  
  same_row_x_v <- c()
  for (yv in y_vals_v) {
    rx <- sort(coords_for_plots$x[abs(y_round_v - yv) < tol_verify * 0.5])
    if (length(rx) >= 2) same_row_x_v <- c(same_row_x_v, diff(rx))
  }
  same_row_x_v <- same_row_x_v[same_row_x_v > R]
  if (length(same_row_x_v) > 0) {
    cat(sprintf("    Actual same-row X distance (median)      = %.4f\n", median(same_row_x_v)))
    cat(sprintf("    Ratio to R*sqrt(3) = %.3f (should be ~1.0)\n", median(same_row_x_v) / (R * sqrt(3))))
  }
}
cat(sprintf("  Detected hex radius: %s\n", 
            ifelse(is.null(detected_hex_radius), "NULL (will auto-calculate)", 
                   sprintf("%.4f", detected_hex_radius))))
} else {
  coords_extracted <- extract_coords_from_names(colnames(st_mat))
  if (!is.null(coords_extracted)) {
    coords_full <- coords_extracted
    matched_mask <- rep(TRUE, nrow(coords_full))
    coords_for_plots <- coords_full
  } else {
    n_spots <- ncol(st_mat)
    n_side <- ceiling(sqrt(n_spots))
    coords_full <- data.frame(
      barcode = colnames(st_mat),
      x = rep(1:n_side, length.out = n_spots),
      y = rep(1:n_side, each = n_side)[1:n_spots]
    )
    matched_mask <- rep(TRUE, n_spots)
    coords_for_plots <- coords_full
    cat("  Generated grid coordinates\n")
  }
}

st_obj <- create_celltrek_seurat(st_mat, assay_name = "RNA")

coord_match <- match(colnames(st_obj), coords_for_plots$barcode)
st_obj$coord_x <- coords_for_plots$x[coord_match]
st_obj$coord_y <- coords_for_plots$y[coord_match]
st_obj$coord_x[is.na(st_obj$coord_x)] <- 0
st_obj$coord_y[is.na(st_obj$coord_y)] <- 0

verify_celltrek_compatible(st_obj, "ST object")

# 3. PREPROCESSING
cat("\n=== Preprocessing ===\n")

sc_obj <- NormalizeData(sc_obj, verbose = FALSE)
sc_obj <- FindVariableFeatures(sc_obj, selection.method = "vst",
                               nfeatures = args$n_hvg, verbose = FALSE)
sc_obj <- ScaleData(sc_obj, verbose = FALSE)
sc_obj <- RunPCA(sc_obj, npcs = args$dims, verbose = FALSE)
sc_obj <- RunUMAP(sc_obj, dims = 1:args$dims, verbose = FALSE)

st_obj <- NormalizeData(st_obj, verbose = FALSE)
st_obj <- FindVariableFeatures(st_obj, selection.method = "vst",
                               nfeatures = args$n_hvg, verbose = FALSE)
st_obj <- ScaleData(st_obj, verbose = FALSE)
n_pcs_st <- min(args$dims, ncol(st_obj) - 1)
st_obj <- RunPCA(st_obj, npcs = n_pcs_st, verbose = FALSE)
st_obj <- RunUMAP(st_obj, dims = 1:n_pcs_st, verbose = FALSE)

# 4. RUN CELLTREK
cat("\n=== Running CellTrek Co-embedding ===\n")

celltrek_int <- NULL
celltrek_success <- FALSE
traint_assay_name <- "traint"

# ---- ATTEMPT 1: Standard CellTrek traint ----
tryCatch({
  cat("  Attempt 1: Standard CellTrek traint()...\n")
  celltrek_int <- CellTrek::traint(
    st_data = st_obj, sc_data = sc_obj,
    sc_assay = "RNA", st_assay = "RNA",
    norm = "LogNormalize", nfeatures = args$n_hvg, npcs = args$dims
  )
  cat("  traint() succeeded!\n")
  celltrek_success <- TRUE
  
  if ("traint" %in% Assays(celltrek_int)) {
    traint_assay_name <- "traint"
  } else if ("integrated" %in% Assays(celltrek_int)) {
    traint_assay_name <- "integrated"
  } else {
    traint_assay_name <- setdiff(Assays(celltrek_int), "RNA")[1]
    if (is.na(traint_assay_name)) traint_assay_name <- "RNA"
  }
  cat(sprintf("  Using assay: %s\n", traint_assay_name))
  
}, error = function(e) {
  cat(sprintf("  traint() failed: %s\n", e$message))
})

# ---- ATTEMPT 2: Manual co-embedding with CellTrek-required metadata ----
if (!celltrek_success) {
  tryCatch({
    cat("\n  Attempt 2: Manual co-embedding with CellTrek metadata...\n")
    
    common_genes <- intersect(rownames(sc_obj), rownames(st_obj))
    cat(sprintf("  Common genes: %d\n", length(common_genes)))
    if (length(common_genes) < 100) stop("Too few common genes")
    
    sc_counts_common <- safe_get_assay_data(sc_obj, "counts")[common_genes, ]
    st_counts_common <- safe_get_assay_data(st_obj, "counts")[common_genes, ]
    
    if (!inherits(sc_counts_common, "dgCMatrix"))
      sc_counts_common <- as(as.matrix(sc_counts_common), "dgCMatrix")
    if (!inherits(st_counts_common, "dgCMatrix"))
      st_counts_common <- as(as.matrix(st_counts_common), "dgCMatrix")
    
    # Add prefixes
    colnames(sc_counts_common) <- paste0("cell_", colnames(sc_counts_common))
    colnames(st_counts_common) <- paste0("spot_", colnames(st_counts_common))
    
    merged_counts <- cbind(sc_counts_common, st_counts_common)
    cat(sprintf("  Merged matrix: %d genes x %d cells\n",
                nrow(merged_counts), ncol(merged_counts)))
    
    celltrek_int <- create_celltrek_seurat(merged_counts, assay_name = "RNA")
    
    sc_cell_names <- paste0("cell_", colnames(sc_obj))
    st_cell_names <- paste0("spot_", colnames(st_obj))
    
    # ================================================================
    # CRITICAL FIX: Add "type" column that CellTrek::celltrek() requires
    # CellTrek expects: "cell" for SC, "spot" for ST
    # ================================================================
    celltrek_int$type <- ifelse(
      grepl("^cell_", colnames(celltrek_int)), "cell", "spot"
    )
    cat(sprintf("  Added 'type' column: %d cells, %d spots\n",
                sum(celltrek_int$type == "cell"),
                sum(celltrek_int$type == "spot")))
    
    # Also add data_type for our own use
    celltrek_int$data_type <- ifelse(
      grepl("^cell_", colnames(celltrek_int)), "SC", "ST"
    )
    
    # Cell type labels
    celltrek_int@meta.data[[label_col]] <- NA_character_
    sc_match <- match(sc_cell_names, colnames(celltrek_int))
    valid_sc <- !is.na(sc_match)
    celltrek_int@meta.data[[label_col]][sc_match[valid_sc]] <- sc_obj@meta.data[[label_col]][valid_sc]
    
    # ================================================================
    # CRITICAL FIX: Add spatial coordinates that CellTrek expects
    # CellTrek uses coord_x / coord_y from ST data for spatial charting
    # The "Distance between spots is: 0" error came from missing coordinates
    # ================================================================
    celltrek_int$coord_x <- NA_real_
    celltrek_int$coord_y <- NA_real_
    st_match <- match(st_cell_names, colnames(celltrek_int))
    valid_st <- !is.na(st_match)
    celltrek_int$coord_x[st_match[valid_st]] <- st_obj$coord_x[valid_st]
    celltrek_int$coord_y[st_match[valid_st]] <- st_obj$coord_y[valid_st]
    
    cat(sprintf("  ST cells with coordinates: %d/%d\n",
                sum(!is.na(celltrek_int$coord_x)), ncol(celltrek_int)))
    
    # Process
    celltrek_int <- NormalizeData(celltrek_int, verbose = FALSE)
    celltrek_int <- FindVariableFeatures(celltrek_int, nfeatures = args$n_hvg, verbose = FALSE)
    celltrek_int <- ScaleData(celltrek_int, verbose = FALSE)
    celltrek_int <- RunPCA(celltrek_int, npcs = args$dims, verbose = FALSE)
    celltrek_int <- RunUMAP(celltrek_int, dims = 1:args$dims, verbose = FALSE)
    
    traint_assay_name <- "RNA"
    celltrek_success <- TRUE
    
    cat("  Manual co-embedding completed successfully\n")
    cat(sprintf("  Object: %d genes x %d cells\n", nrow(celltrek_int), ncol(celltrek_int)))
    
  }, error = function(e) {
    cat(sprintf("  Manual co-embedding failed: %s\n", e$message))
  })
}

if (!celltrek_success || is.null(celltrek_int)) {
  stop("All co-embedding attempts failed.")
}

# ---- CELLTREK CHARTING ----
cat("\n=== Running CellTrek Spatial Charting ===\n")

celltrek_result <- NULL
charting_success <- FALSE

tryCatch({
  cat(sprintf("  Calling celltrek() with int_assay='%s'...\n", traint_assay_name))
  
  celltrek_result <- CellTrek::celltrek(
    st_sc_int = celltrek_int,
    int_assay = traint_assay_name,
    sc_data = sc_obj,
    sc_assay = "RNA",
    reduction = "pca",
    intp = TRUE,
    intp_pnt = args$intp_pnt,
    intp_lin = FALSE,
    nPCs = args$dims,
    ntree = 1000,
    dist_thresh = args$dist_thresh,
    top_spot = 1,
    spot_n = args$spot_n,
    repel_r = args$repel_r,
    repel_iter = args$repel_iter,
    keep_model = TRUE
  )
  
  cat(sprintf("  celltrek() succeeded! Mapped %d cells\n", ncol(celltrek_result)))
  charting_success <- TRUE
  
}, error = function(e) {
  cat(sprintf("  celltrek() failed: %s\n", e$message))
  cat("  Using KNN-based deconvolution fallback\n")
})

# ==============================================================================
# FALLBACK: KNN-based deconvolution (COMPLETELY REWRITTEN)
# ==============================================================================
# Instead of mapping each SC cell → nearest spot (which concentrates cells),
# we map each ST spot → its K nearest SC neighbors in PCA space,
# then count cell types among those neighbors to get proportions.
# This guarantees EVERY spot gets a proportion vector.
# ==============================================================================

if (!charting_success || is.null(celltrek_result)) {
  cat("\n=== Fallback: KNN-based Deconvolution (Spot-Centric) ===\n")
  
  all_cells <- colnames(celltrek_int)
  
  # Identify SC and ST cells
  if (any(grepl("^cell_", all_cells)) && any(grepl("^spot_", all_cells))) {
    sc_cells <- grep("^cell_", all_cells, value = TRUE)
    st_cells <- grep("^spot_", all_cells, value = TRUE)
  } else if ("data_type" %in% colnames(celltrek_int@meta.data)) {
    sc_cells <- all_cells[celltrek_int$data_type == "SC"]
    st_cells <- all_cells[celltrek_int$data_type == "ST"]
  } else {
    stop("Cannot identify SC/ST cells in co-embedded object")
  }
  
  cat(sprintf("  SC cells: %d, ST cells: %d\n", length(sc_cells), length(st_cells)))
  
  # Get PCA embeddings
  pca_embed <- Embeddings(celltrek_int, "pca")
  sc_pca <- pca_embed[sc_cells, , drop = FALSE]
  st_pca <- pca_embed[st_cells, , drop = FALSE]
  
  # Get cell type labels for SC cells
  sc_clean <- gsub("^(cell|spot|SC|ST)_", "", sc_cells)
  sc_ct_idx <- match(sc_clean, rownames(sc_meta))
  sc_celltypes <- rep(NA_character_, length(sc_cells))
  sc_celltypes[!is.na(sc_ct_idx)] <- sc_meta[[label_col]][sc_ct_idx[!is.na(sc_ct_idx)]]
  names(sc_celltypes) <- sc_cells
  
  # Remove SC cells without cell type annotation
  valid_sc <- !is.na(sc_celltypes)
  sc_pca_valid <- sc_pca[valid_sc, , drop = FALSE]
  sc_celltypes_valid <- sc_celltypes[valid_sc]
  
  cat(sprintf("  SC cells with valid labels: %d/%d\n",
              sum(valid_sc), length(sc_cells)))
  
  # ================================================================
  # SPOT-CENTRIC KNN: For each ST spot, find K nearest SC neighbors
  # ================================================================
  knn_k <- min(args$knn_k, nrow(sc_pca_valid))
  cat(sprintf("  Finding %d nearest SC neighbors per ST spot in PCA space...\n", knn_k))
  
  # Query: ST spots, Reference: SC cells
  nn_result <- get.knnx(sc_pca_valid, st_pca, k = knn_k)
  
  all_types <- sort(unique(sc_celltypes_valid))
  cat(sprintf("  Cell types: %d\n", length(all_types)))
  
  # Build proportion matrix: one row per ST spot
  # Use inverse-distance weighting for smoother proportions
  n_spots <- length(st_cells)
  prop_matrix <- matrix(0, nrow = n_spots, ncol = length(all_types),
                        dimnames = list(st_cells, all_types))
  
  cat("  Computing weighted cell type proportions per spot...\n")
  for (i in seq_len(n_spots)) {
    nn_idx <- nn_result$nn.index[i, ]
    nn_dist <- nn_result$nn.dist[i, ]
    nn_types <- sc_celltypes_valid[nn_idx]
    
    # Inverse distance weights (with floor to avoid division by zero)
    nn_dist_safe <- pmax(nn_dist, 1e-10)
    weights <- 1 / nn_dist_safe
    weights <- weights / sum(weights)
    
    # Weighted type counts
    for (j in seq_along(nn_idx)) {
      ct <- nn_types[j]
      if (!is.na(ct)) {
        prop_matrix[i, ct] <- prop_matrix[i, ct] + weights[j]
      }
    }
  }
  
  # Normalize rows to sum to 1
  row_sums <- rowSums(prop_matrix)
  row_sums[row_sums == 0] <- 1
  prop_matrix <- prop_matrix / row_sums
  
  # Map ST cell names back to original spot barcodes
  st_clean_names <- gsub("^spot_", "", st_cells)
  rownames(prop_matrix) <- st_clean_names
  
  cat(sprintf("  Proportion matrix: %d spots x %d types\n",
              nrow(prop_matrix), ncol(prop_matrix)))
  cat(sprintf("  Spots with non-zero proportions: %d/%d (%.1f%%)\n",
              sum(rowSums(prop_matrix) > 0), nrow(prop_matrix),
              100 * sum(rowSums(prop_matrix) > 0) / nrow(prop_matrix)))
  
  # ================================================================
  # Also create mapped_cells for state visualization
  # For each SC cell, assign it to its nearest ST spot's coordinates
  # ================================================================
  cat("  Creating mapped cell coordinates for visualization...\n")
  
  # Get ST coordinates
  st_clean_for_coords <- gsub("^spot_", "", st_cells)
  st_coord_idx <- match(st_clean_for_coords, coords_for_plots$barcode)
  st_x <- coords_for_plots$x[st_coord_idx]
  st_y <- coords_for_plots$y[st_coord_idx]
  valid_st_coords <- !is.na(st_x) & !is.na(st_y)
  
  # For visualization: map SC cells to spatial coordinates via KNN
  # Use only ST cells with valid coordinates
  st_pca_with_coords <- st_pca[valid_st_coords, , drop = FALSE]
  st_x_with_coords <- st_x[valid_st_coords]
  st_y_with_coords <- st_y[valid_st_coords]
  
  k_map <- min(5, sum(valid_st_coords))
  nn_map <- get.knnx(st_pca_with_coords, sc_pca_valid, k = k_map)
  
  sc_mapped_x <- numeric(nrow(sc_pca_valid))
  sc_mapped_y <- numeric(nrow(sc_pca_valid))
  
  for (i in seq_len(nrow(sc_pca_valid))) {
    nn_idx <- nn_map$nn.index[i, ]
    nn_dist <- nn_map$nn.dist[i, ]
    nn_x <- st_x_with_coords[nn_idx]
    nn_y <- st_y_with_coords[nn_idx]
    
    if (min(nn_dist) < 1e-10) {
      best <- which.min(nn_dist)
      sc_mapped_x[i] <- nn_x[best]
      sc_mapped_y[i] <- nn_y[best]
    } else {
      w <- 1 / nn_dist
      w <- w / sum(w)
      sc_mapped_x[i] <- sum(w * nn_x)
      sc_mapped_y[i] <- sum(w * nn_y)
    }
  }
  
  # Add jitter to avoid all cells at exact spot centers
  # Jitter radius = ~10% of median inter-spot distance
  if (nrow(coords_for_plots) >= 2) {
    nn_spots <- get.knn(as.matrix(coords_for_plots[, c("x", "y")]), k = 1)
    jitter_radius <- median(nn_spots$nn.dist) * 0.15
  } else {
    jitter_radius <- 1
  }
  sc_mapped_x <- sc_mapped_x + rnorm(length(sc_mapped_x), 0, jitter_radius)
  sc_mapped_y <- sc_mapped_y + rnorm(length(sc_mapped_y), 0, jitter_radius)
  
  mapped_cells <- data.frame(
    cell_id = names(sc_celltypes_valid),
    coord_x = sc_mapped_x,
    coord_y = sc_mapped_y,
    celltype = sc_celltypes_valid,
    stringsAsFactors = FALSE
  )
  mapped_cells <- mapped_cells[!is.na(mapped_cells$coord_x) &
                                 !is.na(mapped_cells$celltype), ]
  
  cat(sprintf("  Mapped %d SC cells to spatial coordinates for visualization\n",
              nrow(mapped_cells)))
  
  # Use co-embedded object as result
  celltrek_result <- celltrek_int
  celltrek_result$coord_x <- NA_real_
  celltrek_result$coord_y <- NA_real_
  mc_match <- match(mapped_cells$cell_id, colnames(celltrek_result))
  celltrek_result$coord_x[mc_match[!is.na(mc_match)]] <- mapped_cells$coord_x[!is.na(mc_match)]
  celltrek_result$coord_y[mc_match[!is.na(mc_match)]] <- mapped_cells$coord_y[!is.na(mc_match)]
  
  # Also set ST cell coordinates
  st_match2 <- match(st_cells, colnames(celltrek_result))
  valid_st2 <- !is.na(st_match2) & valid_st_coords
  celltrek_result$coord_x[st_match2[valid_st2]] <- st_x[valid_st2]
  celltrek_result$coord_y[st_match2[valid_st2]] <- st_y[valid_st2]
  
  charting_success <- TRUE
  
  # Flag that we used KNN fallback — proportions are already computed above
  used_knn_fallback <- TRUE
  assign("saved_sc_pca_valid", sc_pca_valid, envir = .GlobalEnv)
  assign("saved_st_pca", st_pca, envir = .GlobalEnv)
  assign("saved_sc_celltypes_valid", sc_celltypes_valid, envir = .GlobalEnv)
  cat("  Saved PCA embeddings for state map visualization\n")
} else {
  used_knn_fallback <- FALSE
}

# If native CellTrek charting succeeded, extract mapped cells
if (charting_success && !exists("mapped_cells", inherits = FALSE)) {
  ct_col_found <- NULL
  for (candidate in c(label_col, "CellType", "celltype", "cell_type")) {
    if (candidate %in% colnames(celltrek_result@meta.data)) {
      ct_col_found <- candidate
      break
    }
  }
  if (!is.null(ct_col_found) &&
      "coord_x" %in% colnames(celltrek_result@meta.data) &&
      "coord_y" %in% colnames(celltrek_result@meta.data)) {
    mapped_cells <- data.frame(
      cell_id = colnames(celltrek_result),
      coord_x = celltrek_result$coord_x,
      coord_y = celltrek_result$coord_y,
      celltype = celltrek_result@meta.data[[ct_col_found]],
      stringsAsFactors = FALSE
    )
    mapped_cells <- mapped_cells[!is.na(mapped_cells$coord_x) &
                                   !is.na(mapped_cells$coord_y) &
                                   !is.na(mapped_cells$celltype), ]
  } else {
    mapped_cells <- data.frame(cell_id = character(0), coord_x = numeric(0),
                               coord_y = numeric(0), celltype = character(0))
  }
}

cat(sprintf("  Total mapped cells: %d\n", nrow(mapped_cells)))

# 5. CALCULATE CELL TYPE PROPORTIONS PER SPOT
cat("\n=== Calculating Cell Type Proportions ===\n")

if (exists("used_knn_fallback") && used_knn_fallback && exists("prop_matrix")) {
  # ================================================================
  # KNN fallback already computed proportions directly
  # Just need to align with coords_for_plots barcodes
  # ================================================================
  cat("  Using KNN-computed proportions (spot-centric)...\n")
  
  # Match prop_matrix rows to coords_for_plots barcodes
  prop_match <- match(coords_for_plots$barcode, rownames(prop_matrix))
  
  all_types <- colnames(prop_matrix)
  final_prop <- matrix(0, nrow = nrow(coords_for_plots), ncol = length(all_types),
                       dimnames = list(coords_for_plots$barcode, all_types))
  
  valid_prop <- !is.na(prop_match)
  final_prop[valid_prop, ] <- prop_matrix[prop_match[valid_prop], ]
  
  prop_df <- as.data.frame(final_prop)
  prop_df <- prop_df[, order(colnames(prop_df))]
  
  spots_with_cells <- sum(rowSums(as.matrix(prop_df)) > 0)
  cat(sprintf("  Spots with proportions: %d/%d (%.1f%%)\n",
              spots_with_cells, nrow(prop_df),
              100 * spots_with_cells / nrow(prop_df)))
  
} else if (nrow(mapped_cells) > 0) {
  # Native CellTrek charting succeeded — count cells per spot
  cat("  Computing proportions from CellTrek-mapped cells...\n")
  
  spot_coords_mat <- as.matrix(coords_for_plots[, c("x", "y")])
  cell_coords_mat <- as.matrix(mapped_cells[, c("coord_x", "coord_y")])
  
  nn_spot <- get.knnx(spot_coords_mat, cell_coords_mat, k = 1)
  mapped_cells$nearest_spot <- coords_for_plots$barcode[nn_spot$nn.index[, 1]]
  mapped_cells$dist_to_spot <- nn_spot$nn.dist[, 1]
  
  all_types <- sort(unique(mapped_cells$celltype))
  prop_matrix <- matrix(0, nrow = nrow(coords_for_plots), ncol = length(all_types),
                        dimnames = list(coords_for_plots$barcode, all_types))
  
  for (spot in coords_for_plots$barcode) {
    spot_cells <- mapped_cells[mapped_cells$nearest_spot == spot, ]
    if (nrow(spot_cells) > 0) {
      cell_counts <- table(spot_cells$celltype)
      for (ct in names(cell_counts)) {
        prop_matrix[spot, ct] <- cell_counts[ct]
      }
    }
  }
  
  row_sums <- rowSums(prop_matrix)
  row_sums[row_sums == 0] <- 1
  prop_matrix <- prop_matrix / row_sums
  
  prop_df <- as.data.frame(prop_matrix)
  prop_df <- prop_df[, order(colnames(prop_df))]
  
  spots_with_cells <- sum(rowSums(as.matrix(prop_df)) > 0)
  cat(sprintf("  Spots with mapped cells: %d/%d (%.1f%%)\n",
              spots_with_cells, nrow(prop_df),
              100 * spots_with_cells / nrow(prop_df)))
  
} else {
  cat("  WARNING: No mapped cells, using uniform proportions\n")
  prop_df <- data.frame(matrix(1 / length(original_labels),
                               nrow = nrow(coords_for_plots),
                               ncol = length(original_labels)))
  colnames(prop_df) <- sort(original_labels)
  rownames(prop_df) <- coords_for_plots$barcode
}

cat(sprintf("  Final proportions: %d spots x %d cell types\n",
            nrow(prop_df), ncol(prop_df)))

# Verify barcode matching
cat("\n  === Barcode Matching Verification ===\n")
cat(sprintf("    prop_df rownames (first 3): %s\n",
            paste(head(rownames(prop_df), 3), collapse = ", ")))
cat(sprintf("    coords_for_plots barcodes (first 3): %s\n",
            paste(head(coords_for_plots$barcode, 3), collapse = ", ")))
n_match <- sum(rownames(prop_df) %in% coords_for_plots$barcode)
cat(sprintf("    Matched: %d/%d (%.1f%%)\n", n_match, nrow(prop_df),
            100 * n_match / nrow(prop_df)))

# Quick diagnostic: print proportion summary
cat("\n  Proportion summary per cell type:\n")
for (ct in colnames(prop_df)) {
  vals <- prop_df[[ct]]
  cat(sprintf("    %-45s: mean=%.4f, max=%.4f, nonzero=%d\n",
              ct, mean(vals), max(vals), sum(vals > 0)))
}

# 6. CELL STATE ANALYSIS
cat("\n=== Cell State Analysis ===\n")

state_data <- extract_cell_states(celltrek_result, sc_obj, label_col,
                                  n_dims = args$state_dims)

state_continuum <- calculate_state_continuum(celltrek_result, label_col,
                                             resolution = args$state_resolution)

if (!is.null(state_continuum$celltrek_result)) {
  celltrek_result <- state_continuum$celltrek_result
}

if (nrow(mapped_cells) > 0) {
  state_match <- match(mapped_cells$cell_id, names(state_continuum$state_clusters))
  mapped_cells$state_cluster <- state_continuum$state_clusters[state_match]
}

# 7. SAVE RESULTS
cat("\n=== Saving Results ===\n")

write.csv(prop_df, args$output_csv, quote = FALSE)
cat(sprintf("  Saved proportions to: %s\n", args$output_csv))

write.csv(mapped_cells, file.path(args$output_dir, "mapped_cell_coordinates.csv"),
          row.names = FALSE)
cat(sprintf("  Saved mapped cells to: %s\n",
            file.path(args$output_dir, "mapped_cell_coordinates.csv")))

write.csv(state_data$states, file.path(args$output_dir, "cell_states.csv"),
          row.names = FALSE)

state_cluster_df <- data.frame(
  cell_id = names(state_continuum$state_clusters),
  state_cluster = state_continuum$state_clusters,
  stringsAsFactors = FALSE
)
write.csv(state_cluster_df, file.path(args$output_dir, "state_clusters.csv"),
          row.names = FALSE)

saveRDS(celltrek_result, file.path(args$output_dir, "celltrek_result.rds"))

# 8. VISUALIZATIONS
cat("\n=== Generating Visualizations ===\n")

# Use detected radius if available, otherwise NULL triggers auto-calculation
OVERLAP_FACTOR <- 1.03
raw_hex_radius <- if (exists("detected_hex_radius") && !is.null(detected_hex_radius)) {
  detected_hex_radius
} else {
  NULL
}
plot_hex_radius <- if (!is.null(raw_hex_radius)) raw_hex_radius * OVERLAP_FACTOR else NULL

cat(sprintf("  Plot hex radius: raw=%.4f, with overlap=%.4f (factor=%.2f)\n",
            ifelse(is.null(raw_hex_radius), NA, raw_hex_radius),
            ifelse(is.null(plot_hex_radius), NA, plot_hex_radius),
            OVERLAP_FACTOR))

intensity_path <- file.path(args$output_dir, "spatial_intensity_maps.png")
plot_spatial_intensity_maps(prop_df, coords_for_plots, intensity_path,
                           hex_radius = plot_hex_radius,
                           hex_angle = args$hex_orientation,
                           coords_full = coords_full, matched_mask = matched_mask,
                           min_prop = args$min_prop)

dominant_path <- file.path(args$output_dir, "spatial_dominant_type.png")
plot_spatial_dominant_type(prop_df, coords_for_plots, dominant_path,
                          hex_radius = plot_hex_radius,
                          hex_angle = args$hex_orientation,
                          coords_full = coords_full, matched_mask = matched_mask)

cooccurrence_path <- file.path(args$output_dir, "cooccurrence_heatmap.png")
plot_cooccurrence_heatmap(prop_df, cooccurrence_path)

heatmap_path <- file.path(args$output_dir, "proportion_heatmap.png")
tryCatch({
  png(heatmap_path, width = 1200, height = 800, res = 150)
  pheatmap(as.matrix(prop_df), cluster_rows = FALSE, cluster_cols = FALSE,
           main = "CellTrek Cell Type Proportions",
           fontsize_row = 6, fontsize_col = 10, angle_col = 45)
  dev.off()
  cat(sprintf("  Saved proportion heatmap to: %s\n", heatmap_path))
}, error = function(e) {
  cat(sprintf("  Proportion heatmap failed: %s\n", e$message))
})

if (nrow(mapped_cells) > 0) {
  # Retrieve saved PCA data if available (from KNN fallback)
  sc_pca_for_states <- if (exists("saved_sc_pca_valid", envir = .GlobalEnv)) {
    get("saved_sc_pca_valid", envir = .GlobalEnv)
  } else NULL
  
  st_pca_for_states <- if (exists("saved_st_pca", envir = .GlobalEnv)) {
    get("saved_st_pca", envir = .GlobalEnv)
  } else NULL
  
  sc_ct_for_states <- if (exists("saved_sc_celltypes_valid", envir = .GlobalEnv)) {
    get("saved_sc_celltypes_valid", envir = .GlobalEnv)
  } else NULL
  
  cat(sprintf("  State maps: sc_pca=%s, st_pca=%s, sc_ct=%s\n",
              ifelse(is.null(sc_pca_for_states), "NULL", 
                     paste(dim(sc_pca_for_states), collapse="x")),
              ifelse(is.null(st_pca_for_states), "NULL",
                     paste(dim(st_pca_for_states), collapse="x")),
              ifelse(is.null(sc_ct_for_states), "NULL",
                     as.character(length(sc_ct_for_states)))))
  
  plot_spatial_state_maps(mapped_cells, state_data, coords_for_plots, args$output_dir,
                          hex_angle = args$hex_orientation,
                          coords_full = coords_full, matched_mask = matched_mask,
                          sc_pca = sc_pca_for_states,
                          st_pca = st_pca_for_states,
                          sc_celltypes = sc_ct_for_states,
                          knn_k = args$knn_k)
}

plot_continuum_maps(celltrek_result, state_continuum, mapped_cells,
                    args$output_dir, label_col)

# Individual cell type maps
cat("  Generating individual cell type maps...\n")

hex_radius <- if (!is.null(plot_hex_radius)) {
  plot_hex_radius
} else {
  calculate_hex_radius(coords_for_plots, orientation = args$hex_orientation) * OVERLAP_FACTOR
}
cat(sprintf("  Individual maps hex_radius (with overlap): %.4f\n", hex_radius))

# Rotate coordinates
max_x_ref <- max(coords_for_plots$x, na.rm = TRUE)
max_y_ref <- max(coords_for_plots$y, na.rm = TRUE)
coords_rot <- coords_for_plots
coords_rot$x_plot <- coords_for_plots$x
coords_rot$y_plot <- max_y_ref - coords_for_plots$y

# Background tissue hexagons (for unmatched spots if any)
bg_hex_polys <- NULL
if (!is.null(coords_full)) {
  max_x_full <- max(coords_full$x, na.rm = TRUE)
  max_y_full <- max(coords_full$y, na.rm = TRUE)
  coords_full_rot <- coords_full
  coords_full_rot$x <- coords_full$x
  coords_full_rot$y <- max_y_full - coords_full$y
  
  if (!is.null(matched_mask)) {
    unmatched_coords <- coords_full_rot[!matched_mask, ]
    if (nrow(unmatched_coords) > 0) {
      bg_hex_polys <- create_all_hexagons(unmatched_coords, hex_radius, 
                                           rotation = args$hex_orientation)
    }
  }
}

# Axis limits from full tissue
if (!is.null(coords_full)) {
  coords_full_rot2 <- data.frame(
    x = coords_full$x,
    y = max(coords_full$y, na.rm = TRUE) - coords_full$y
  )
  x_range <- range(coords_full_rot2$x)
  y_range <- range(coords_full_rot2$y)
  x_expand <- diff(x_range) * 0.05
  y_expand <- diff(y_range) * 0.05
  xlim <- c(x_range[1] - x_expand, x_range[2] + x_expand)
  ylim <- c(y_range[1] - y_expand, y_range[2] + y_expand)
} else {
  xlim <- NULL
  ylim <- NULL
}

# Match proportions to rotated coordinates
prop_coord_idx <- match(rownames(prop_df), coords_rot$barcode)
valid_prop <- !is.na(prop_coord_idx)

spot_plot_df <- data.frame(
  barcode = rownames(prop_df)[valid_prop],
  x = coords_rot$x_plot[prop_coord_idx[valid_prop]],
  y = coords_rot$y_plot[prop_coord_idx[valid_prop]],
  stringsAsFactors = FALSE
)

cat(sprintf("  Spots for individual maps: %d/%d\n", nrow(spot_plot_df), nrow(prop_df)))

# Create hexagons for ALL matched spots
all_spot_hex <- create_all_hexagons(spot_plot_df, hex_radius, 
                                     rotation = args$hex_orientation)

for (ct in colnames(prop_df)) {
  # Get proportion values for this cell type
  ct_values <- prop_df[[ct]][valid_prop]
  
  hex_data <- all_spot_hex
  hex_data$value <- ct_values[hex_data$spot_idx]
  
  p <- ggplot()
  
  # Background (unmatched spots)
  if (!is.null(bg_hex_polys) && nrow(bg_hex_polys) > 0) {
    p <- p + geom_polygon(data = bg_hex_polys,
                          aes(x = x, y = y, group = id),
                          # CHANGED: match Seurat script style
                          fill = "#F0F0F0", color = NA,
                          linewidth = 0, alpha = 0.5)
  }
  
  # All matched spots colored by proportion
  hex_data$value[hex_data$value < args$min_prop] <- NA   # <--- UPDATED THRESHOLD
  
  p <- p + geom_polygon(data = hex_data,
                         aes(x = x, y = y, group = id, fill = value),
                         color = NA,
                         linewidth = 0) +
    scale_fill_viridis(option = "plasma",
                       limits = c(args$min_prop, max(ct_values + 1e-8)),  # <--- UPDATED LIMITS
                       na.value = "#F0F0F0",
                       name = "Proportion") +
    coord_fixed(xlim = xlim, ylim = ylim) +
    theme_void() +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          legend.position = "right",
          legend.key.size = unit(0.5, "cm")) +
    ggtitle(paste0(ct, " — Proportion per Spot"))
  
  ct_safe <- gsub("[^A-Za-z0-9]", "_", ct)
  indiv_dims <- calculate_plot_dimensions(
    data.frame(x = coords_rot$x_plot, y = coords_rot$y_plot))
  ggsave(file.path(args$output_dir, paste0("celltype_", ct_safe, ".png")),
         p, width = indiv_dims$width, height = indiv_dims$height, dpi = 300)
}
cat(sprintf("  Saved %d individual cell type maps\n", ncol(prop_df)))

# 9. GROUND TRUTH COMPARISON
if (!is.null(args$ground_truth) && file.exists(args$ground_truth)) {
  cat("\n=== Comparing with Ground Truth ===\n")
  
  gt <- read.csv(args$ground_truth, row.names = 1, check.names = FALSE)
  common_spots <- intersect(rownames(prop_df), rownames(gt))
  common_types <- intersect(colnames(prop_df), colnames(gt))
  
  cat(sprintf("  Common spots: %d, Common types: %d\n",
              length(common_spots), length(common_types)))
  
  if (length(common_spots) > 0 && length(common_types) > 0) {
    pred_sub <- prop_df[common_spots, common_types]
    gt_sub <- gt[common_spots, common_types]
    
    flat_pred <- as.vector(as.matrix(pred_sub))
    flat_gt <- as.vector(as.matrix(gt_sub))
    
    cor_val <- cor(flat_pred, flat_gt)
    rmse_val <- sqrt(mean((flat_pred - flat_gt)^2))
    mae_val <- mean(abs(flat_pred - flat_gt))
    
    cat(sprintf("\n  OVERALL PEARSON CORRELATION: %.4f\n", cor_val))
    cat(sprintf("  RMSE: %.4f\n", rmse_val))
    cat(sprintf("  MAE: %.4f\n", mae_val))
    
    cat("\n  Per cell-type correlations:\n")
    for (ct in common_types) {
      ct_cor <- cor(pred_sub[[ct]], gt_sub[[ct]])
      cat(sprintf("    %s: r = %.4f\n", ct, ct_cor))
    }
    
    metrics_df <- data.frame(
      Metric = c("Pearson", "RMSE", "MAE"),
      Value = c(cor_val, rmse_val, mae_val)
    )
    write.csv(metrics_df, file.path(args$output_dir, "metrics.csv"),
              row.names = FALSE, quote = FALSE)
    
    corr_plot_path <- file.path(args$output_dir, "correlation_plot.png")
    p_corr_df <- data.frame(Truth = flat_gt, Prediction = flat_pred)
    png(corr_plot_path, width = 800, height = 800, res = 120)
    p <- ggplot(p_corr_df, aes(x = Truth, y = Prediction)) +
      geom_point(alpha = 0.5, color = "blue") +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      theme_minimal() +
      labs(title = paste0("CellTrek vs Ground Truth (r=", round(cor_val, 3), ")"),
           x = "Ground Truth Proportion", y = "Predicted Proportion")
    print(p)
    dev.off()
    cat(sprintf("  Saved correlation plot to: %s\n", corr_plot_path))
  }
}

# 10. SUMMARY
cat("\n")
cat("============================================================\n")
cat("CELLTREK ANALYSIS COMPLETE\n")
cat("============================================================\n")
cat(sprintf("Total cells mapped: %d\n", nrow(mapped_cells)))
cat(sprintf("Total spots: %d\n", nrow(prop_df)))
cat(sprintf("Cell types: %d\n", ncol(prop_df)))
cat(sprintf("Cell states identified: %d\n",
            length(unique(na.omit(state_continuum$state_clusters)))))
if (exists("used_knn_fallback") && used_knn_fallback) {
  cat(sprintf("Deconvolution method: KNN-based (K=%d, spot-centric)\n", args$knn_k))
} else {
  cat("Deconvolution method: CellTrek native charting\n")
}
cat("\nCell type distribution:\n")
if (nrow(mapped_cells) > 0) {
  ct_table <- sort(table(mapped_cells$celltype), decreasing = TRUE)
  print(ct_table)
}
cat(sprintf("\nResults saved to: %s\n", args$output_dir))
cat("============================================================\n")