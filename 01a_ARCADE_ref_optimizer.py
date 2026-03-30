#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated Two-Stage Bayesian Optimization and Final Analysis Pipeline for scRNA-seq.

This script combines a Bayesian optimization stage for parameter discovery with a
final, detailed analysis stage that uses the discovered optimal parameters. It now
supports iterative refinement for low-confidence cells and specific export of
consistent cell populations.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import time
import celltypist
from celltypist import models
import argparse
import random
import re
import anndata
import sys
import matplotlib

# Added import for robust marker aggregation in Stage 2
from collections import defaultdict

# Use 'Agg' backend for non-interactive environments
matplotlib.use('Agg')

MARKER_PRIOR_DB = None  # Will hold the loaded marker database DataFrame
MARKER_PRIOR_DICT = {}  # Dict mapping (species, organ, cell_type) -> set of marker genes

# ==============================================================================
# IMPORTANT: MPS (Marker Prior Score) DEFINITION
# ==============================================================================
# In this script, MPS is calculated as the F1 SCORE:
#
#   F1 = 2 × (Precision × Recall) / (Precision + Recall)
#
# Where:
#   - Precision = |DEGs ∩ Canonical| / |DEGs|
#   - Recall = |DEGs ∩ Canonical| / |Canonical|
#
# The variable names (mean_mps, per_group_mps) are retained for backward
# compatibility, but they now contain F1 Scores rather than pure recall.
# ==============================================================================
def calculate_marker_prior_score(
    adata,
    prior_dict: dict,
    groupby_key: str = 'leiden',
    annotation_key: str = 'ctpt_consensus_prediction',
    n_top_degs: int = 50,
    deg_ranking_method: str = 'original',
    deg_weight_fc: float = 0.4,
    deg_weight_expr: float = 0.3,
    deg_weight_pct: float = 0.3,
    species: str = "human",
    min_cells_per_group: int = 5,
    penalize_unmatched: bool = True,
    similarity_threshold: float = 0.6,
    verbose: bool = False
) -> dict:
    """
    Calculates Marker Prior Score (MPS) as F1 SCORE for each cluster/cell type.
    
    MPS is defined as the F1 Score (harmonic mean of Precision and Recall):
    - Precision: What fraction of top DEGs are canonical markers?
    - Recall: What fraction of canonical markers appear in top DEGs?
    - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    This provides a balanced metric that penalizes both:
    - False positives (DEGs that aren't canonical markers)
    - False negatives (canonical markers not in DEGs)
    
    Returns:
        dict with keys:
            'mean_mps': Mean F1 score across all groups (0-100 scale)
            'per_group_mps': Dict mapping group -> F1 score
            'per_group_precision': Dict mapping group -> precision
            'per_group_recall': Dict mapping group -> recall
            'mean_precision': Mean precision across all groups
            'mean_recall': Mean recall across all groups
            'n_matched_groups': Number of groups with matched cell types
            'n_unmatched_groups': Number of groups without matches
    """
    import scanpy as sc
    
    results = {
        'mean_mps': 0.0,  # This is now F1 Score
        'per_group_mps': {},
        'per_group_precision': {},
        'per_group_recall': {},
        'per_group_n_overlap': {},
        'per_group_n_canonical': {},
        'per_group_n_degs': {},
        'per_group_matched_type': {},
        'mean_precision': 0.0,
        'mean_recall': 0.0,
        'n_matched_groups': 0,
        'n_unmatched_groups': 0
    }
    
    if not prior_dict:
        return results
    
    # Get unique groups
    if groupby_key not in adata.obs.columns:
        print(f"[WARNING] groupby_key '{groupby_key}' not found in adata.obs")
        return results
    
    # Get group counts
    group_counts = adata.obs[groupby_key].value_counts()
    valid_groups = group_counts[group_counts >= min_cells_per_group].index.tolist()
    
    if len(valid_groups) < 2:
        print(f"[WARNING] Need at least 2 valid groups for DEG analysis, found {len(valid_groups)}")
        return results
    
    # Run DEG analysis
    deg_key = f'rank_genes_mps_{groupby_key}'
    try:
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby_key,
            groups=valid_groups,
            reference='rest',
            method='wilcoxon',
            pts=True,
            key_added=deg_key
        )
    except Exception as e:
        print(f"[WARNING] DEG analysis failed: {e}")
        return results
    
    # Get DEG results
    try:
        marker_df = sc.get.rank_genes_groups_df(adata, group=None, key=deg_key)
    except Exception as e:
        print(f"[WARNING] Failed to extract DEG results: {e}")
        return results
    
    # Apply composite ranking if requested
    if deg_ranking_method == 'composite':
        marker_df = apply_composite_deg_ranking(
            marker_df=marker_df,
            deg_weight_fc=deg_weight_fc,
            deg_weight_expr=deg_weight_expr,
            deg_weight_pct=deg_weight_pct
        )
        rank_col = 'composite_score'
    else:
        rank_col = 'logfoldchanges'
    
    # Create matcher for robust cell type matching
    matcher = create_robust_cell_type_matcher(prior_dict, similarity_threshold=similarity_threshold)
    
    # Calculate F1 for each group
    f1_scores = []
    precision_scores = []
    recall_scores = []
    n_matched = 0
    n_unmatched = 0
    
    for group_id in valid_groups:
        group_id_str = str(group_id)
        
        # Get annotation for this group (majority vote)
        if annotation_key in adata.obs.columns:
            group_mask = adata.obs[groupby_key] == group_id
            group_annotations = adata.obs.loc[group_mask, annotation_key]
            if len(group_annotations) > 0:
                annotation = group_annotations.value_counts().index[0]
            else:
                annotation = group_id_str
        else:
            annotation = group_id_str
        
        # Get canonical markers for this cell type
        markers, matched_type, match_method, confidence = get_markers_for_cell_type(
            annotation, matcher, verbose=verbose
        )
        
        # Get top DEGs for this group
        group_df = marker_df[marker_df['group'] == group_id].copy()
        if group_df.empty:
            results['per_group_mps'][group_id_str] = 0.0
            results['per_group_precision'][group_id_str] = 0.0
            results['per_group_recall'][group_id_str] = 0.0
            if penalize_unmatched:
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
            n_unmatched += 1
            continue
        
        # Sort by ranking column
        if rank_col in group_df.columns:
            group_df = group_df.sort_values(rank_col, ascending=False)
        
        # Get top N DEGs
        top_genes = set(group_df.head(n_top_degs)['names'].tolist())
        
        # Standardize gene names
        top_genes_standard = {standardize_gene_name(g, species) for g in top_genes}
        
        # Calculate metrics
        if markers and len(markers) > 0:
            n_matched += 1
            
            # Standardize canonical markers
            markers_standard = {standardize_gene_name(m, species) for m in markers}
            
            # Calculate overlap
            overlap = top_genes_standard.intersection(markers_standard)
            n_overlap = len(overlap)
            
            # Precision: What fraction of DEGs are canonical markers?
            precision = n_overlap / len(top_genes_standard) if top_genes_standard else 0.0
            
            # Recall: What fraction of canonical markers are in DEGs?
            recall = n_overlap / len(markers_standard) if markers_standard else 0.0
            
            # F1 Score (THIS IS NOW THE MPS!)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            # Store results (scale to 0-100 for consistency with original MPS)
            results['per_group_mps'][group_id_str] = f1 * 100
            results['per_group_precision'][group_id_str] = precision * 100
            results['per_group_recall'][group_id_str] = recall * 100
            results['per_group_n_overlap'][group_id_str] = n_overlap
            results['per_group_n_canonical'][group_id_str] = len(markers_standard)
            results['per_group_n_degs'][group_id_str] = len(top_genes_standard)
            results['per_group_matched_type'][group_id_str] = matched_type
            
            f1_scores.append(f1 * 100)
            precision_scores.append(precision * 100)
            recall_scores.append(recall * 100)
            
            if verbose:
                print(f"       [{group_id_str}] '{annotation}' -> '{matched_type}': "
                      f"P={precision*100:.1f}%, R={recall*100:.1f}%, F1={f1*100:.1f}%")
        else:
            # No match found
            n_unmatched += 1
            results['per_group_mps'][group_id_str] = 0.0
            results['per_group_precision'][group_id_str] = 0.0
            results['per_group_recall'][group_id_str] = 0.0
            results['per_group_matched_type'][group_id_str] = None
            
            if penalize_unmatched:
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
            
            if verbose:
                print(f"       [{group_id_str}] '{annotation}' -> NO MATCH (MPS=0)")
    
    # Calculate means
    if f1_scores:
        results['mean_mps'] = np.mean(f1_scores)  # Mean F1 Score
        results['mean_precision'] = np.mean(precision_scores)
        results['mean_recall'] = np.mean(recall_scores)
    
    results['n_matched_groups'] = n_matched
    results['n_unmatched_groups'] = n_unmatched
    
    return results

def load_marker_prior_database(csv_path: str, species_filter: str = "Human", organ_filter: str = None):
    """
    Loads an external marker gene database from a CSV file with standardized names.
    
    Expected CSV columns: species, organ, cell_type, marker_genes, gene_count
    The 'marker_genes' column should contain semicolon-separated gene symbols.
    
    Args:
        csv_path (str): Path to the marker database CSV file.
        species_filter (str): Filter for species (default: "Human").
        organ_filter (str): Optional filter for organ/tissue type.
    
    Returns:
        dict: Mapping from cell_type -> dict with:
            'original_name': original cell type name from DB
            'markers_standard': set of standardized marker genes
            'markers_original': set of original marker genes
            'matching_keys': tuple for fuzzy matching
    """
    global MARKER_PRIOR_DB, MARKER_PRIOR_DICT
    
    print(f"\n--- Loading External Marker Prior Database ---")
    print(f"       -> Path: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"       -> Loaded {len(df)} entries from database")
        
        # Validate required columns
        required_cols = ['species', 'cell_type', 'marker_genes']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Apply species filter
        if species_filter:
            df = df[df['species'].str.lower() == species_filter.lower()]
            print(f"       -> After species filter ('{species_filter}'): {len(df)} entries")
        
        # Apply organ filter if specified
        if organ_filter and 'organ' in df.columns:
            df = df[df['organ'].str.lower().str.contains(organ_filter.lower(), na=False)]
            print(f"       -> After organ filter ('{organ_filter}'): {len(df)} entries")
        
        MARKER_PRIOR_DB = df
        
        # Determine species for gene standardization
        gene_species = "human" if species_filter.lower() == "human" else "mouse"
        
        # Build the marker dictionary with standardization
        marker_dict = {}
        
        for _, row in df.iterrows():
            cell_type_original = str(row['cell_type']).strip()
            markers_str = str(row['marker_genes'])
            
            if pd.isna(markers_str) or markers_str == 'nan':
                continue
            
            # Parse and standardize markers
            markers_original = {m.strip() for m in markers_str.split(';') if m.strip()}
            markers_standard = {standardize_gene_name(m, gene_species) for m in markers_original}
            
            # Create standardized cell type key
            cell_type_key = standardize_celltype_name(cell_type_original)
            matching_keys = create_celltype_matching_key(cell_type_original)
            
            # Also try to get canonical form for better matching
            canonical_form = get_canonical_celltype(cell_type_original)
            
            # Aggregate markers if cell type already exists
            if cell_type_key in marker_dict:
                marker_dict[cell_type_key]['markers_standard'].update(markers_standard)
                marker_dict[cell_type_key]['markers_original'].update(markers_original)
            else:
                marker_dict[cell_type_key] = {
                    'original_name': cell_type_original,
                    'markers_standard': markers_standard,
                    'markers_original': markers_original,
                    'matching_keys': matching_keys,
                    'canonical_form': canonical_form  # NEW: Store canonical form
                }
        
        MARKER_PRIOR_DICT = marker_dict
        
        print(f"       -> Built marker dictionary for {len(MARKER_PRIOR_DICT)} unique cell types")
        
        # Print summary statistics
        total_markers = sum(len(v['markers_standard']) for v in MARKER_PRIOR_DICT.values())
        avg_markers = total_markers / len(MARKER_PRIOR_DICT) if MARKER_PRIOR_DICT else 0
        print(f"       -> Total unique markers: {total_markers}")
        print(f"       -> Average markers per cell type: {avg_markers:.1f}")
        
        return MARKER_PRIOR_DICT
        
    except FileNotFoundError:
        print(f"[WARNING] Marker prior database not found at: {csv_path}")
        print(f"       -> Marker Prior Score (MPS) will be disabled.")
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to load marker prior database: {e}")
        return {}

def standardize_gene_name(gene_name: str, species: str = "human") -> str:
    """
    Standardizes a single gene name based on species convention.
    
    Args:
        gene_name: Raw gene name/symbol
        species: "human" or "mouse"
    
    Returns:
        Standardized gene name:
        - Human: UPPERCASE (CD4, PTPRC, MT-CO1)
        - Mouse: Title case with special handling (Cd4, Ptprc, mt-Co1)
    """
    if not gene_name or not isinstance(gene_name, str):
        return gene_name
    
    gene_name = gene_name.strip()
    
    if species.lower() == "human":
        return gene_name.upper()
    elif species.lower() == "mouse":
        # Mouse genes: Title case, but mitochondrial genes stay lowercase prefix
        if gene_name.lower().startswith('mt-') or gene_name.lower().startswith('mt.'):
            # Mitochondrial: mt-Xxx format
            prefix = gene_name[:3].lower()
            suffix = gene_name[3:].capitalize() if len(gene_name) > 3 else ""
            return prefix + suffix
        else:
            # Standard mouse gene: Titlecase (Cd4, Ptprc)
            return gene_name.capitalize()
    else:
        # Default: uppercase
        return gene_name.upper()


def standardize_gene_names_array(gene_names, species: str = "human") -> list:
    """
    Standardizes an array/list of gene names.
    
    Args:
        gene_names: Iterable of gene names (list, pd.Index, np.array)
        species: "human" or "mouse"
    
    Returns:
        List of standardized gene names
    """
    return [standardize_gene_name(str(g), species) for g in gene_names]

def normalize_to_01(series: pd.Series) -> pd.Series:
    """
    Normalizes a pandas Series to 0-1 range using min-max scaling.
    
    Args:
        series: pandas Series of numeric values
    
    Returns:
        Normalized series with values between 0 and 1
    """
    min_val = series.min()
    max_val = series.max()
    
    if max_val - min_val == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    
    return (series - min_val) / (max_val - min_val)


def apply_composite_deg_ranking(
    marker_df: pd.DataFrame,
    deg_weight_fc: float = 0.4,
    deg_weight_expr: float = 0.3,
    deg_weight_pct: float = 0.3
) -> pd.DataFrame:
    """
    Applies composite ranking to DEG dataframe.
    
    The composite score combines:
    - log2 fold change (differential expression strength)
    - mean expression level (gene expression magnitude)
    - pct difference (cluster specificity: pct.1 - pct.2)
    
    Args:
        marker_df: DataFrame from sc.get.rank_genes_groups_df() with columns:
                   'names', 'scores', 'logfoldchanges', 'pvals', 'pvals_adj', 
                   'pct_nz_group', 'pct_nz_reference', 'group'
        deg_weight_fc: Weight for log2 fold change (default: 0.4)
        deg_weight_expr: Weight for expression level (default: 0.3)
        deg_weight_pct: Weight for pct difference (default: 0.3)
    
    Returns:
        DataFrame with added 'composite_score' column, sorted by composite_score
    """
    # Normalize weights to sum to 1
    weight_sum = deg_weight_fc + deg_weight_expr + deg_weight_pct
    if weight_sum > 0:
        w_fc = deg_weight_fc / weight_sum
        w_expr = deg_weight_expr / weight_sum
        w_pct = deg_weight_pct / weight_sum
    else:
        w_fc, w_expr, w_pct = 1/3, 1/3, 1/3
    
    # Create a copy to avoid modifying original
    df = marker_df.copy()
    
    # Calculate pct_diff (cluster specificity)
    # Scanpy uses 'pct_nz_group' and 'pct_nz_reference' for percentage non-zero
    if 'pct_nz_group' in df.columns and 'pct_nz_reference' in df.columns:
        df['pct_diff'] = df['pct_nz_group'] - df['pct_nz_reference']
    else:
        # Fallback: use scores as proxy if pct columns not available
        df['pct_diff'] = df['scores'] if 'scores' in df.columns else 0
    
    # Get expression level - use 'scores' as proxy for expression strength
    # In Scanpy's rank_genes_groups, higher scores generally indicate stronger markers
    if 'scores' in df.columns:
        df['expr_proxy'] = df['scores']
    else:
        df['expr_proxy'] = df['logfoldchanges'].abs()
    
    # Apply composite ranking per group
    result_dfs = []
    
    for group_name, group_df in df.groupby('group', sort=False):
        group_df = group_df.copy()
        
        # Normalize each component to 0-1 scale within the group
        if 'logfoldchanges' in group_df.columns:
            group_df['norm_log2fc'] = normalize_to_01(group_df['logfoldchanges'].fillna(0))
        else:
            group_df['norm_log2fc'] = 0.5
        
        group_df['norm_expr'] = normalize_to_01(group_df['expr_proxy'].fillna(0))
        group_df['norm_pct_diff'] = normalize_to_01(group_df['pct_diff'].fillna(0))
        
        # Calculate weighted composite score
        group_df['composite_score'] = (
            w_fc * group_df['norm_log2fc'] +
            w_expr * group_df['norm_expr'] +
            w_pct * group_df['norm_pct_diff']
        )
        
        # Sort by composite score (descending)
        group_df = group_df.sort_values('composite_score', ascending=False)
        
        result_dfs.append(group_df)
    
    # Combine all groups
    result_df = pd.concat(result_dfs, ignore_index=True)
    
    return result_df

def create_robust_cell_type_matcher(
    prior_dict: dict,
    similarity_threshold: float = 0.6
) -> dict:
    """
    Creates a mapping structure for robust cell type matching.
    
    Parameters
    ----------
    prior_dict : dict
        Dictionary with cell types as keys and marker genes as values.
    similarity_threshold : float
        Minimum similarity score for fuzzy matching (0-1).
    
    Returns
    -------
    dict
        Matcher object containing normalized mappings and original dict.
    """
    from difflib import SequenceMatcher
    
    def normalize_cell_type(name: str) -> str:
        """Normalize cell type name for matching."""
        if not isinstance(name, str):
            return ""
        # Lowercase
        normalized = name.lower().strip()
        # Remove common prefixes
        prefixes_to_remove = [
            'brain ', 'cerebellum ', 'cortex ', 'cortical ',
            'hippocampal ', 'hippocampus ', 'spinal ',
            'human ', 'mouse ', 'developing ', 'adult ', 'fetal ',
            'mature ', 'immature ', 'early ', 'late ',
        ]
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        # Remove common suffixes
        suffixes_to_remove = [' cell', ' cells', ' neuron', ' neurons']
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
                normalized = normalized[:-len(suffix)]
        # Standardize common terms
        replacements = {
            'oligo ': 'oligodendrocyte ',
            'opc': 'oligodendrocyte precursor',
            'npc': 'neural progenitor',
            'ipc': 'intermediate progenitor',
            'rg': 'radial glia',
            'rgc': 'radial glial cell',
            'astro': 'astrocyte',
            'micro': 'microglia',
            'endo': 'endothelial',
            'peri': 'pericyte',
            'exc ': 'excitatory ',
            'inh ': 'inhibitory ',
            'gaba': 'gabaergic',
            'glut': 'glutamatergic',
        }
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        return normalized.strip()
    
    # Build normalized lookup
    normalized_to_original = {}
    for cell_type in prior_dict.keys():
        norm = normalize_cell_type(cell_type)
        if norm:
            normalized_to_original[norm] = cell_type
    
    return {
        'original_dict': prior_dict,
        'normalized_to_original': normalized_to_original,
        'normalize_func': normalize_cell_type,
        'similarity_threshold': similarity_threshold
    }


def match_cell_type_robust(
    query_cell_type: str,
    matcher: dict,
    verbose: bool = False
) -> tuple:
    """
    Performs robust cell type matching using multiple strategies.
    
    ENHANCED: Now expands abbreviated cell type names before matching.
    
    ...existing docstring...
    """
    from difflib import SequenceMatcher, get_close_matches
    
    prior_dict = matcher['original_dict']
    normalized_to_original = matcher['normalized_to_original']
    normalize_func = matcher['normalize_func']
    threshold = matcher['similarity_threshold']
    
    if not query_cell_type or not isinstance(query_cell_type, str):
        return (None, None, 0.0)
    
    # =========================================================================
    # NEW: EXPAND ABBREVIATIONS FIRST
    # =========================================================================
    query_expanded, was_expanded, query_original = expand_celltype_with_context(
        query_cell_type, verbose=verbose
    )
    
    # Use expanded form for matching
    query_clean = query_expanded.strip()
    query_lower = query_clean.lower()
    query_normalized = normalize_func(query_clean)
    
    # Keep original forms for fallback matching
    query_original_clean = query_cell_type.strip()
    query_original_lower = query_original_clean.lower()
    query_original_normalized = normalize_func(query_original_clean)
    
    if was_expanded and verbose:
        print(f"       [MATCH] Using expanded form: '{query_original}' -> '{query_expanded}'")
    
    # Strategy 1: Exact match
    if query_clean in prior_dict:
        if verbose:
            print(f"       [Match] '{query_clean}' -> EXACT")
        return (query_clean, 'exact', 1.0)
    
    # Strategy 2: Case-insensitive exact match
    for db_type in prior_dict.keys():
        if query_lower == db_type.lower():
            if verbose:
                print(f"       [Match] '{query_clean}' -> '{db_type}' (case-insensitive)")
            return (db_type, 'case_insensitive', 0.99)
    
    # Strategy 3: Normalized exact match
    if query_normalized in normalized_to_original:
        matched = normalized_to_original[query_normalized]
        if verbose:
            print(f"       [Match] '{query_clean}' -> '{matched}' (normalized)")
        return (matched, 'normalized', 0.95)
    
    # Strategy 4: Substring match - query contains database term
    for db_type in prior_dict.keys():
        db_lower = db_type.lower()
        # Check if database type is substring of query
        if len(db_lower) >= 4 and db_lower in query_lower:
            if verbose:
                print(f"       [Match] '{query_clean}' -> '{db_type}' (substring: db in query)")
            return (db_type, 'substring_db_in_query', 0.85)
    
    # Strategy 5: Substring match - database contains query term
    for db_type in prior_dict.keys():
        db_lower = db_type.lower()
        # Check if query is substring of database type
        if len(query_lower) >= 4 and query_lower in db_lower:
            if verbose:
                print(f"       [Match] '{query_clean}' -> '{db_type}' (substring: query in db)")
            return (db_type, 'substring_query_in_db', 0.80)
    
    # Strategy 6: Normalized substring match
    for norm_type, orig_type in normalized_to_original.items():
        if len(query_normalized) >= 4 and query_normalized in norm_type:
            if verbose:
                print(f"       [Match] '{query_clean}' -> '{orig_type}' (normalized substring)")
            return (orig_type, 'normalized_substring', 0.75)
        if len(norm_type) >= 4 and norm_type in query_normalized:
            if verbose:
                print(f"       [Match] '{query_clean}' -> '{orig_type}' (normalized substring)")
            return (orig_type, 'normalized_substring', 0.75)
    
    # Strategy 7: Fuzzy matching on normalized names
    best_match = None
    best_score = 0.0
    
    for norm_type, orig_type in normalized_to_original.items():
        score = SequenceMatcher(None, query_normalized, norm_type).ratio()
        if score > best_score:
            best_score = score
            best_match = orig_type
    
    if best_score >= threshold:
        if verbose:
            print(f"       [Match] '{query_clean}' -> '{best_match}' (fuzzy: {best_score:.2f})")
        return (best_match, 'fuzzy', best_score)
    
    # Strategy 8: Fuzzy matching on original names (fallback)
    for db_type in prior_dict.keys():
        score = SequenceMatcher(None, query_lower, db_type.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = db_type
    
    if best_score >= threshold:
        if verbose:
            print(f"       [Match] '{query_clean}' -> '{best_match}' (fuzzy original: {best_score:.2f})")
        return (best_match, 'fuzzy_original', best_score)
    
    # No match found
    if verbose:
        print(f"       [Match] '{query_clean}' -> NO MATCH (best score: {best_score:.2f})")
    return (None, None, 0.0)


def get_markers_for_cell_type(
    query_cell_type: str,
    matcher: dict,
    verbose: bool = False
) -> tuple:
    """
    Get canonical markers for a cell type using robust matching.
    
    ENHANCED: Now expands abbreviated cell type names before matching.
    
    ...existing docstring...
    """
    # =========================================================================
    # NEW: EXPAND ABBREVIATIONS FIRST
    # =========================================================================
    query_expanded, was_expanded, query_original = expand_celltype_with_context(
        query_cell_type, verbose=verbose
    )
    
    if was_expanded and verbose:
        print(f"       [MARKER] Expanded '{query_original}' -> '{query_expanded}'")
    
    # Try matching with expanded form first
    matched_type, match_method, confidence = match_cell_type_robust(
        query_expanded, matcher, verbose
    )
    
    # If no match with expanded form, try original form as fallback
    if matched_type is None and was_expanded:
        if verbose:
            print(f"       [MARKER] No match for expanded form, trying original '{query_original}'")
        matched_type, match_method, confidence = match_cell_type_robust(
            query_original, matcher, verbose
        )
    
    if matched_type is None:
        return (set(), None, None, 0.0)
    
    # =========================================================================
    # FIX: Extract the marker SET from the dictionary structure
    # Handle case-insensitive key lookup
    # =========================================================================
    original_dict = matcher['original_dict']
    marker_data = None
    matched_key = None
    
    # Try exact match first
    if matched_type in original_dict:
        marker_data = original_dict[matched_type]
        matched_key = matched_type
    else:
        # Try case-insensitive match
        matched_type_lower = matched_type.lower().strip()
        for db_key in original_dict.keys():
            if db_key.lower().strip() == matched_type_lower:
                marker_data = original_dict[db_key]
                matched_key = db_key
                break
        
        # If still not found, try partial match
        if marker_data is None:
            for db_key in original_dict.keys():
                if matched_type_lower in db_key.lower() or db_key.lower() in matched_type_lower:
                    marker_data = original_dict[db_key]
                    matched_key = db_key
                    if verbose:
                        print(f"       [DEBUG] Partial key match: '{matched_type}' -> '{db_key}'")
                    break
    
    if marker_data is None:
        if verbose:
            print(f"       [WARNING] No marker data found for matched type '{matched_type}'")
            print(f"       [DEBUG] Available keys (first 10): {list(original_dict.keys())[:10]}")
        return (set(), matched_type, match_method, confidence)
    
    # Handle different possible structures
    markers = set()
    
    if isinstance(marker_data, set):
        markers = marker_data
    elif isinstance(marker_data, dict):
        # Dictionary structure - try multiple keys
        markers = marker_data.get('markers_standard', set())
        if not markers:
            markers = marker_data.get('markers_original', set())
        if not markers:
            markers = marker_data.get('markers', set())
        if not markers:
            # Maybe the dict values themselves are the markers
            for v in marker_data.values():
                if isinstance(v, (set, list)):
                    markers = set(v)
                    break
        
        # DEBUG: Print what we found
        if verbose and not markers:
            print(f"       [DEBUG] marker_data keys: {marker_data.keys()}")
            print(f"       [DEBUG] marker_data content sample: {dict(list(marker_data.items())[:3])}")
            
    elif isinstance(marker_data, list):
        markers = set(marker_data)
    elif isinstance(marker_data, str):
        # Maybe it's a semicolon-separated string
        markers = {m.strip() for m in marker_data.split(';') if m.strip()}
    
    # Ensure it's a set
    if not isinstance(markers, set):
        markers = set(markers) if markers else set()
    
    if verbose:
        print(f"       [DEBUG] '{query_cell_type}' -> '{matched_key}': {len(markers)} markers found")
        if markers:
            print(f"       [DEBUG] Sample markers: {list(markers)[:5]}")
    
    return (markers, matched_key if matched_key else matched_type, match_method, confidence)

def expand_celltype_abbreviation(cell_type: str, verbose: bool = False) -> str:
    """
    Expands abbreviated cell type names to their full canonical forms.
    
    This function handles:
    1. Direct abbreviation matches (e.g., 'OPC' -> 'oligodendrocyte precursor cell')
    2. Abbreviations with location prefixes (e.g., 'Cortex OPC' -> 'Cortex oligodendrocyte precursor cell')
    3. Compound abbreviations (e.g., 'L5-6 Exc' -> 'layer 5-6 excitatory neuron')
    
    Args:
        cell_type: Raw cell type name (possibly abbreviated)
        verbose: If True, print expansion details
    
    Returns:
        Expanded cell type name, or original if no expansion found
    """
    if not cell_type or not isinstance(cell_type, str):
        return cell_type
    
    original = cell_type
    cell_type_lower = cell_type.strip().lower()
    
    # Strategy 1: Direct exact match (case-insensitive)
    if cell_type_lower in ABBREVIATION_TO_FULL:
        expanded = ABBREVIATION_TO_FULL[cell_type_lower]
        if verbose:
            print(f"       [EXPAND] '{original}' -> '{expanded}' (direct match)")
        return expanded
    
    # Strategy 2: Check if the cell type ENDS with a known abbreviation
    # This handles cases like "Cortex OPC" -> "Cortex oligodendrocyte precursor cell"
    words = cell_type_lower.split()
    
    if len(words) >= 1:
        # Check last word
        last_word = words[-1]
        if last_word in ABBREVIATION_TO_FULL:
            prefix = ' '.join(words[:-1])
            expanded_suffix = ABBREVIATION_TO_FULL[last_word]
            if prefix:
                expanded = f"{prefix} {expanded_suffix}"
            else:
                expanded = expanded_suffix
            if verbose:
                print(f"       [EXPAND] '{original}' -> '{expanded}' (suffix expansion)")
            return expanded
        
        # Check last two words combined (e.g., "L5-6")
        if len(words) >= 2:
            last_two = ' '.join(words[-2:])
            if last_two in ABBREVIATION_TO_FULL:
                prefix = ' '.join(words[:-2])
                expanded_suffix = ABBREVIATION_TO_FULL[last_two]
                if prefix:
                    expanded = f"{prefix} {expanded_suffix}"
                else:
                    expanded = expanded_suffix
                if verbose:
                    print(f"       [EXPAND] '{original}' -> '{expanded}' (two-word suffix expansion)")
                return expanded
    
    # Strategy 3: Check if the cell type STARTS with a known abbreviation
    # This handles cases like "OPC (cortex)" -> "oligodendrocyte precursor cell (cortex)"
    if len(words) >= 1:
        first_word = words[0]
        if first_word in ABBREVIATION_TO_FULL:
            suffix = ' '.join(words[1:])
            expanded_prefix = ABBREVIATION_TO_FULL[first_word]
            if suffix:
                expanded = f"{expanded_prefix} {suffix}"
            else:
                expanded = expanded_prefix
            if verbose:
                print(f"       [EXPAND] '{original}' -> '{expanded}' (prefix expansion)")
            return expanded
    
    # Strategy 4: Token-by-token expansion for compound types
    # This handles cases like "Micro/Macro" or "OPC-like"
    expanded_tokens = []
    made_expansion = False
    
    # Split on common delimiters
    import re
    tokens = re.split(r'([\s/\-_]+)', cell_type_lower)
    
    for token in tokens:
        token_stripped = token.strip()
        if token_stripped in ABBREVIATION_TO_FULL:
            expanded_tokens.append(ABBREVIATION_TO_FULL[token_stripped])
            made_expansion = True
        else:
            expanded_tokens.append(token)
    
    if made_expansion:
        expanded = ''.join(expanded_tokens)
        if verbose:
            print(f"       [EXPAND] '{original}' -> '{expanded}' (token expansion)")
        return expanded
    
    # No expansion found - return original
    return original


def expand_celltype_with_context(cell_type: str, verbose: bool = False) -> tuple:
    """
    Expands cell type abbreviation and returns both original and expanded forms.
    
    Args:
        cell_type: Raw cell type name
        verbose: If True, print expansion details
    
    Returns:
        Tuple of (expanded_name, was_expanded: bool, original_name)
    """
    original = cell_type
    expanded = expand_celltype_abbreviation(cell_type, verbose=verbose)
    was_expanded = (expanded.lower() != original.lower())
    
    return (expanded, was_expanded, original)

def standardize_celltype_name(cell_type: str) -> str:
    """
    Standardizes a cell type name for robust matching.
    
    Transformations:
    1. Strip whitespace
    2. Convert to lowercase
    3. Replace underscores, hyphens with spaces
    4. Remove special characters (parentheses, brackets, etc.)
    5. Collapse multiple spaces to single space
    6. Remove common suffixes like "cell", "cells"
    7. Handle common abbreviations
    
    Args:
        cell_type: Raw cell type name
    
    Returns:
        Standardized cell type name for matching
    """
    if not cell_type or not isinstance(cell_type, str):
        return ""
    
    # Basic cleaning
    name = cell_type.strip().lower()
    
    # Replace separators with spaces
    name = re.sub(r'[_\-]+', ' ', name)
    
    # Remove special characters but keep alphanumeric and spaces
    name = re.sub(r'[^\w\s]', '', name)
    
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Remove trailing "cell" or "cells" for matching purposes
    # But keep for display - this is just for the key
    name = re.sub(r'\s+cells?$', '', name)
    
    # Handle common prefix/suffix variations
    name = re.sub(r'^positive\s+', '', name)  # "positive CD4" -> "CD4"
    name = re.sub(r'\s+positive$', '', name)  # "CD4 positive" -> "CD4"
    
    return name

# ==============================================================================
# --- CELL TYPE SYNONYM DICTIONARY ---
# ==============================================================================

# Comprehensive synonym dictionary for cell type matching
# Format: canonical_name -> list of alternative names/abbreviations
CELLTYPE_SYNONYMS = {
    # === T Cells ===
    't cell': ['t lymphocyte', 'tlymphocyte', 'tcel', 't-cell', 'tcell'],
    'cd4 t cell': [
        'helper t cell', 'th cell', 'cd4+ t cell', 'cd4 positive t cell',
        'cd4 t lymphocyte', 'cd4+ t lymphocyte', 't helper cell',
        'cd4 positive t lymphocyte', 'helper t lymphocyte'
    ],
    'cd8 t cell': [
        'cytotoxic t cell', 'ctl', 'cd8+ t cell', 'cd8 positive t cell',
        'cd8 t lymphocyte', 'cd8+ t lymphocyte', 'cytotoxic t lymphocyte',
        'killer t cell', 'tc cell'
    ],
    'regulatory t cell': [
        'treg', 't reg', 'cd4 cd25 t cell', 'foxp3 t cell',
        'cd4+cd25+ t cell', 'suppressor t cell', 'regulatory t lymphocyte'
    ],
    'naive t cell': [
        'tn', 'naive t', 'naive t lymphocyte', 'tn cell',
        'cd45ra t cell', 'virgin t cell'
    ],
    'memory t cell': [
        'tm', 'memory t', 'memory t lymphocyte', 'tm cell',
        'antigen experienced t cell'
    ],
    'effector t cell': [
        'teff', 'effector t lymphocyte', 'activated t cell',
        'effector memory t cell', 'tem'
    ],
    'central memory t cell': [
        'tcm', 'central memory t lymphocyte', 'tcm cell'
    ],
    'effector memory t cell': [
        'tem', 'effector memory t lymphocyte', 'tem cell'
    ],
    'gamma delta t cell': [
        'gd t cell', 'γδ t cell', 'gamma delta t lymphocyte',
        'gdt cell', 'gammadelta t cell'
    ],
    
    # === B Cells ===
    'b cell': ['b lymphocyte', 'blymphocyte', 'bcel', 'b-cell', 'bcell'],
    'plasma cell': [
        'plasmacyte', 'plasma b cell', 'antibody secreting cell',
        'asc', 'plasma lymphocyte', 'effector b cell'
    ],
    'memory b cell': [
        'bmem', 'memory b', 'memory b lymphocyte', 'bmem cell',
        'antigen experienced b cell'
    ],
    'naive b cell': [
        'naive b lymphocyte', 'virgin b cell', 'mature naive b cell'
    ],
    'plasmablast': [
        'plasma blast', 'proliferating plasma cell', 'pre plasma cell'
    ],
    'germinal center b cell': [
        'gc b cell', 'germinal centre b cell', 'gcb cell'
    ],
    'marginal zone b cell': [
        'mz b cell', 'marginal zone b lymphocyte'
    ],
    'follicular b cell': [
        'fo b cell', 'follicular b lymphocyte'
    ],
    
    # === NK Cells ===
    'nk cell': [
        'natural killer', 'nkcell', 'natural killer cell', 'nk',
        'nk lymphocyte', 'natural killer lymphocyte', 'large granular lymphocyte'
    ],
    'nkt cell': [
        'natural killer t', 'nkt', 'invariant nkt', 'inkt cell',
        'natural killer t cell', 'inkt', 'type i nkt cell'
    ],
    'cd56bright nk cell': [
        'cd56 bright nk cell', 'cd56hi nk cell', 'immunoregulatory nk cell'
    ],
    'cd56dim nk cell': [
        'cd56 dim nk cell', 'cd56lo nk cell', 'cytotoxic nk cell'
    ],
    
    # === Myeloid Cells ===
    'monocyte': [
        'mono', 'monocytes', 'blood monocyte', 'peripheral monocyte'
    ],
    'classical monocyte': [
        'cd14 monocyte', 'cd14+ monocyte', 'cd14 positive monocyte',
        'cd14++cd16- monocyte', 'inflammatory monocyte'
    ],
    'non classical monocyte': [
        'cd16 monocyte', 'cd16+ monocyte', 'cd14+cd16+ monocyte',
        'patrolling monocyte', 'cd14dimcd16+ monocyte'
    ],
    'intermediate monocyte': [
        'cd14+cd16+ monocyte', 'cd14++cd16+ monocyte',
        'transitional monocyte'
    ],
    'macrophage': [
        'macro', 'mf', 'mφ', 'tissue macrophage', 'resident macrophage'
    ],
    'm1 macrophage': [
        'classically activated macrophage', 'inflammatory macrophage',
        'pro inflammatory macrophage', 'm1 mf'
    ],
    'm2 macrophage': [
        'alternatively activated macrophage', 'anti inflammatory macrophage',
        'tissue repair macrophage', 'm2 mf'
    ],
    'dendritic cell': [
        'dc', 'dendriticcell', 'antigen presenting cell', 'apc'
    ],
    'conventional dendritic cell': [
        'cdc', 'classical dendritic cell', 'myeloid dendritic cell'
    ],
    'plasmacytoid dendritic cell': [
        'pdc', 'plasmacytoid dc', 'interferon producing cell'
    ],
    'neutrophil': [
        'neut', 'pmn', 'polymorphonuclear', 'polymorphonuclear leukocyte',
        'granulocyte', 'neutrophilic granulocyte'
    ],
    'eosinophil': [
        'eos', 'eosinophilic granulocyte'
    ],
    'basophil': [
        'baso', 'basophilic granulocyte'
    ],
    'mast cell': [
        'mastocyte', 'tissue mast cell', 'mucosal mast cell'
    ],
    
    # === Stem/Progenitor Cells ===
    'hematopoietic stem cell': [
        'hsc', 'hspc', 'hematopoietic progenitor cell',
        'hematopoietic stem and progenitor cell', 'blood stem cell'
    ],
    'multipotent progenitor': [
        'mpp', 'multipotent progenitor cell'
    ],
    'common lymphoid progenitor': [
        'clp', 'lymphoid progenitor'
    ],
    'common myeloid progenitor': [
        'cmp', 'myeloid progenitor'
    ],
    'granulocyte monocyte progenitor': [
        'gmp', 'granulocyte macrophage progenitor'
    ],
    'megakaryocyte erythroid progenitor': [
        'mep', 'megakaryocyte erythrocyte progenitor'
    ],
    'progenitor': ['prog', 'progenitor cell'],
    
    # === Stromal Cells ===
    'fibroblast': [
        'fib', 'fibro', 'connective tissue cell', 'stromal fibroblast'
    ],
    'myofibroblast': [
        'activated fibroblast', 'contractile fibroblast'
    ],
    'endothelial cell': [
        'ec', 'endothelial', 'endothelium', 'vascular endothelial cell'
    ],
    'lymphatic endothelial cell': [
        'lec', 'lymphatic endothelium'
    ],
    'vascular endothelial cell': [
        'vec', 'blood endothelial cell', 'bec'
    ],
    'epithelial cell': [
        'epi', 'epithelial', 'epithelium'
    ],
    'smooth muscle cell': [
        'smc', 'smooth muscle', 'vascular smooth muscle cell', 'vsmc'
    ],
    'pericyte': [
        'mural cell', 'perivascular cell'
    ],
    
    # === Adipose Cells ===
    'adipocyte': [
        'fat cell', 'adipose cell', 'lipocyte'
    ],
    'white adipocyte': [
        'white fat cell', 'wat adipocyte'
    ],
    'brown adipocyte': [
        'brown fat cell', 'bat adipocyte', 'thermogenic adipocyte'
    ],
    'beige adipocyte': [
        'brite adipocyte', 'inducible brown adipocyte'
    ],
    'preadipocyte': [
        'pre adipocyte', 'preadipocytes', 'adipocyte precursor',
        'adipose progenitor cell', 'adipose precursor'
    ],
    'adipose stem cell': [
        'asc', 'adipose derived stem cell', 'adsc',
        'adipose derived stromal cell'
    ],
    'adipose progenitor cell': [
        'adipocyte progenitor', 'fat progenitor', 'adipogenic progenitor'
    ],
    
    # === Epithelial Subtypes ===
    'keratinocyte': [
        'skin epithelial cell', 'epidermal cell'
    ],
    'basal cell': [
        'basal epithelial cell', 'basal keratinocyte'
    ],
    'luminal cell': [
        'luminal epithelial cell'
    ],
    'alveolar cell': [
        'pneumocyte', 'alveolar epithelial cell'
    ],
    'type i alveolar cell': [
        'at1', 'type 1 pneumocyte', 'alveolar type 1 cell'
    ],
    'type ii alveolar cell': [
        'at2', 'type 2 pneumocyte', 'alveolar type 2 cell',
        'surfactant producing cell'
    ],
    'enterocyte': [
        'intestinal epithelial cell', 'absorptive enterocyte'
    ],
    'goblet cell': [
        'mucus secreting cell', 'mucous cell'
    ],
    'paneth cell': [
        'crypt cell', 'antimicrobial peptide secreting cell'
    ],
    
    # === Neural Cells ===
    'neuron': [
        'nerve cell', 'neuronal cell'
    ],
    'astrocyte': [
        'astroglia', 'astroglial cell'
    ],
    'oligodendrocyte': [
        'oligo', 'myelinating cell'
    ],
    'microglia': [
        'brain macrophage', 'resident brain macrophage'
    ],
    'schwann cell': [
        'peripheral glial cell', 'neurilemma cell'
    ],
    
    # === Other Specialized Cells ===
    'hepatocyte': [
        'liver cell', 'parenchymal liver cell'
    ],
    'stellate cell': [
        'hepatic stellate cell', 'ito cell', 'fat storing cell'
    ],
    'kupffer cell': [
        'liver macrophage', 'hepatic macrophage'
    ],
    'cardiomyocyte': [
        'cardiac muscle cell', 'heart muscle cell'
    ],
    'skeletal muscle cell': [
        'myocyte', 'muscle fiber', 'striated muscle cell'
    ],
    'satellite cell': [
        'muscle stem cell', 'muscle progenitor'
    ],
    'osteoblast': [
        'bone forming cell'
    ],
    'osteoclast': [
        'bone resorbing cell'
    ],
    'chondrocyte': [
        'cartilage cell'
    ],
    'erythrocyte': [
        'red blood cell', 'rbc'
    ],
    'platelet': [
        'thrombocyte'
    ],
    'megakaryocyte': [
        'platelet precursor', 'mk'
    ],
}

# ==============================================================================
# --- CELL TYPE ABBREVIATION EXPANSION DICTIONARY ---
# ==============================================================================
# Maps common abbreviated cell type names to their full canonical forms
# This is the REVERSE lookup - abbreviation -> full name
# Used to expand short names BEFORE matching against marker databases

CELLTYPE_ABBREVIATION_EXPANSIONS = {
    # === Neural/Brain Cell Types ===
    'l5-6': 'layer 5-6 excitatory neuron',
    'l5/6': 'layer 5-6 excitatory neuron',
    'l56': 'layer 5-6 excitatory neuron',
    'l5-6 exc': 'layer 5-6 excitatory neuron',
    'l5-6 excitatory': 'layer 5-6 excitatory neuron',
    
    'l6': 'layer 6 excitatory neuron',
    'l6 exc': 'layer 6 excitatory neuron',
    'l6 excitatory': 'layer 6 excitatory neuron',
    
    'l5': 'layer 5 excitatory neuron',
    'l5 exc': 'layer 5 excitatory neuron',
    'l5 excitatory': 'layer 5 excitatory neuron',
    
    'l4': 'layer 4 excitatory neuron',
    'l4 exc': 'layer 4 excitatory neuron',
    
    'l2-3': 'layer 2-3 excitatory neuron',
    'l2/3': 'layer 2-3 excitatory neuron',
    'l23': 'layer 2-3 excitatory neuron',
    'l2-3 exc': 'layer 2-3 excitatory neuron',
    
    # === Glial Cells ===
    'opc': 'oligodendrocyte precursor cell',
    'opcs': 'oligodendrocyte precursor cell',
    'oligo': 'oligodendrocyte',
    'oligos': 'oligodendrocyte',
    'oligodendro': 'oligodendrocyte',
    
    'astro': 'astrocyte',
    'astros': 'astrocyte',
    'astroglial': 'astrocyte',
    
    'micro': 'microglia',
    'micros': 'microglia',
    'microglial': 'microglia',
    
    # === Immune Cells ===
    'macro': 'macrophage',
    'macros': 'macrophage',
    'mf': 'macrophage',
    'mφ': 'macrophage',
    
    'myeloid': 'myeloid cell',
    
    't': 't cell',
    'tcell': 't cell',
    't-cell': 't cell',
    'tlymph': 't lymphocyte',
    
    'b': 'b cell',
    'bcell': 'b cell',
    'b-cell': 'b cell',
    'blymph': 'b lymphocyte',
    
    'nk': 'natural killer cell',
    'nkcell': 'natural killer cell',
    
    'dc': 'dendritic cell',
    'dcs': 'dendritic cell',
    
    'mono': 'monocyte',
    'monos': 'monocyte',
    
    'neut': 'neutrophil',
    'neuts': 'neutrophil',
    'pmn': 'neutrophil',
    
    # === Vascular/Stromal Cells ===
    'pc': 'pericyte',
    'pcs': 'pericyte',
    'pericytes': 'pericyte',
    
    'smc': 'smooth muscle cell',
    'smcs': 'smooth muscle cell',
    'vsmc': 'vascular smooth muscle cell',
    
    'ec': 'endothelial cell',
    'ecs': 'endothelial cell',
    'endo': 'endothelial cell',
    'endos': 'endothelial cell',
    'endothelial': 'endothelial cell',
    
    'vlmc': 'vascular and leptomeningeal cell',
    'vlmcs': 'vascular and leptomeningeal cell',
    'leptomeningeal': 'vascular and leptomeningeal cell',
    
    'fb': 'fibroblast',
    'fbs': 'fibroblast',
    'fibro': 'fibroblast',
    'fibros': 'fibroblast',
    
    # === Blood Cells ===
    'rb': 'red blood cell',
    'rbc': 'red blood cell',
    'rbcs': 'red blood cell',
    'erythrocyte': 'red blood cell',
    'erythro': 'red blood cell',
    
    'plt': 'platelet',
    'thrombocyte': 'platelet',
    
    # === Stem/Progenitor Cells ===
    'hsc': 'hematopoietic stem cell',
    'hspc': 'hematopoietic stem cell',
    
    'nsc': 'neural stem cell',
    'nscs': 'neural stem cell',
    
    'npc': 'neural progenitor cell',
    'npcs': 'neural progenitor cell',
    
    'ipc': 'intermediate progenitor cell',
    'ipcs': 'intermediate progenitor cell',
    
    'rg': 'radial glia',
    'rgc': 'radial glial cell',
    
    # === Neuron Subtypes ===
    'exc': 'excitatory neuron',
    'excitatory': 'excitatory neuron',
    'glut': 'glutamatergic neuron',
    'glutamatergic': 'glutamatergic neuron',
    
    'inh': 'inhibitory neuron',
    'inhibitory': 'inhibitory neuron',
    'gaba': 'gabaergic neuron',
    'gabaergic': 'gabaergic neuron',
    
    'int': 'interneuron',
    'ints': 'interneuron',
    'interneurons': 'interneuron',
    
    'pv': 'parvalbumin interneuron',
    'pv+': 'parvalbumin interneuron',
    'pvalb': 'parvalbumin interneuron',
    
    'sst': 'somatostatin interneuron',
    'sst+': 'somatostatin interneuron',
    
    'vip': 'vip interneuron',
    'vip+': 'vip interneuron',
    
    'da': 'dopaminergic neuron',
    'dopaminergic': 'dopaminergic neuron',
    
    'sero': 'serotonergic neuron',
    'serotonergic': 'serotonergic neuron',
    '5ht': 'serotonergic neuron',
    
    'chol': 'cholinergic neuron',
    'cholinergic': 'cholinergic neuron',
    'ach': 'cholinergic neuron',
    
    # === Other Common Abbreviations ===
    'epi': 'epithelial cell',
    'epithelial': 'epithelial cell',
    
    'mes': 'mesenchymal cell',
    'mesenchymal': 'mesenchymal cell',
    'msc': 'mesenchymal stem cell',
    
    'adipo': 'adipocyte',
    'adipocytes': 'adipocyte',
    
    'hepato': 'hepatocyte',
    'hepatocytes': 'hepatocyte',
    
    'cardio': 'cardiomyocyte',
    'cm': 'cardiomyocyte',
    'cardiomyocytes': 'cardiomyocyte',
}
ABBREVIATION_TO_FULL = {k.lower(): v for k, v in CELLTYPE_ABBREVIATION_EXPANSIONS.items()}
# Build reverse lookup: synonym -> canonical name
SYNONYM_TO_CANONICAL = {}
for canonical, synonyms in CELLTYPE_SYNONYMS.items():
    SYNONYM_TO_CANONICAL[canonical] = canonical  # Map canonical to itself
    for syn in synonyms:
        SYNONYM_TO_CANONICAL[syn] = canonical

def create_celltype_matching_key(cell_type: str) -> tuple:
    """
    Creates multiple matching keys for a cell type to enable fuzzy matching.
    
    Args:
        cell_type: Raw cell type name
    
    Returns:
        Tuple of (standardized_name, token_set, abbreviated_form, canonical_form)
    """
    standardized = standardize_celltype_name(cell_type)
    
    # Create token set for Jaccard similarity
    tokens = set(standardized.split())
    
    # Create abbreviated form (first letter of each word)
    words = standardized.split()
    abbreviated = ''.join(word[0] for word in words if word)
    
    # Look up canonical form
    canonical = SYNONYM_TO_CANONICAL.get(standardized, None)
    
    return (standardized, tokens, abbreviated, canonical)

def calculate_jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculates Jaccard similarity between two sets.
    
    Args:
        set1: First set of tokens
        set2: Second set of tokens
    
    Returns:
        Jaccard similarity score (0-1)
    """
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def calculate_token_overlap_score(tokens1: set, tokens2: set) -> float:
    """
    Calculates overlap score based on shared tokens.
    More lenient than Jaccard - considers partial overlap.
    
    Args:
        tokens1: First set of tokens
        tokens2: Second set of tokens
    
    Returns:
        Overlap score (0-1)
    """
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    min_size = min(len(tokens1), len(tokens2))
    
    return intersection / min_size if min_size > 0 else 0.0


def get_canonical_celltype(cell_type: str) -> str:
    """
    Returns the canonical cell type name for a given input.
    
    Args:
        cell_type: Raw cell type name
    
    Returns:
        Canonical cell type name, or original if no mapping found
    """
    standardized = standardize_celltype_name(cell_type)
    return SYNONYM_TO_CANONICAL.get(standardized, standardized)


def are_celltypes_equivalent(type1: str, type2: str) -> bool:
    """
    Checks if two cell type names refer to the same cell type.
    
    Args:
        type1: First cell type name
        type2: Second cell type name
    
    Returns:
        True if equivalent, False otherwise
    """
    canonical1 = get_canonical_celltype(type1)
    canonical2 = get_canonical_celltype(type2)
    
    return canonical1 == canonical2

def build_gene_name_mapping(adata, species: str = "human") -> dict:
    """
    Builds a bidirectional mapping between original and standardized gene names.
    
    Args:
        adata: AnnData object
        species: "human" or "mouse"
    
    Returns:
        dict with keys:
            'original_to_standard': {original_name: standard_name}
            'standard_to_original': {standard_name: original_name}
            'standard_set': set of all standardized names
    """
    original_names = adata.var_names.tolist()
    standard_names = standardize_gene_names_array(original_names, species)
    
    original_to_standard = dict(zip(original_names, standard_names))
    standard_to_original = {}
    
    # Handle potential duplicates after standardization
    for orig, std in zip(original_names, standard_names):
        if std not in standard_to_original:
            standard_to_original[std] = orig
        # If duplicate, keep first occurrence (or could keep list)
    
    return {
        'original_to_standard': original_to_standard,
        'standard_to_original': standard_to_original,
        'standard_set': set(standard_names)
    }

def get_all_canonical_markers_from_prior(prior_dict: dict) -> set:
    """
    Extracts all unique marker genes from the loaded prior database.
    
    Returns:
        set: All marker genes (uppercase) across all cell types.
    """
    all_markers = set()
    for markers in prior_dict.values():
        all_markers.update(markers)
    return all_markers


def ensure_markers_in_hvg(adata, prior_dict: dict, n_hvg: int, batch_key: str = None,
                          species: str = "human", max_marker_fraction: float = 0.1):
    """
    Selects HVGs while ensuring canonical markers are included if present in the data.
    
    Args:
        adata: AnnData object (after normalization, before HVG filtering)
        prior_dict: Marker prior dictionary (with standardized structure)
        n_hvg: Target number of HVGs
        batch_key: Batch key for HVG calculation
        species: Species for gene standardization
        max_marker_fraction: Maximum fraction of HVGs to reserve for markers (default: 0.1)
        
    Returns:
        list: Final list of HVG gene names (original names) including protected markers
    """
    # Step 1: Standard HVG calculation
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=n_hvg, 
        batch_key=batch_key,
        flavor='seurat_v3'
    )
    
    hvg_df = adata.var[adata.var.highly_variable].copy()
    
    # Determine sort column
    if 'dispersions_norm' in hvg_df.columns:
        sort_col = 'dispersions_norm'
    elif 'variances_norm' in hvg_df.columns:
        sort_col = 'variances_norm'
    else:
        sort_col = hvg_df.columns[0]
    
    hvg_df = hvg_df.sort_values(sort_col, ascending=False)
    current_hvgs = set(hvg_df.index[:n_hvg])
    
    # Step 2: Build gene name mapping
    gene_mapping = build_gene_name_mapping(adata, species)
    standard_to_original = gene_mapping['standard_to_original']
    original_to_standard = gene_mapping['original_to_standard']
    available_genes_standard = gene_mapping['standard_set']
    
    # Step 3: Get all canonical markers from prior (standardized)
    all_canonical_standard = set()
    for data in prior_dict.values():
        all_canonical_standard.update(data['markers_standard'])
    
    # Find canonical markers present in data
    canonical_in_data_standard = all_canonical_standard & available_genes_standard
    canonical_in_data_original = {
        standard_to_original[g] for g in canonical_in_data_standard 
        if g in standard_to_original
    }
    
    # Step 4: Find canonical markers NOT in current HVGs
    missing_canonical = canonical_in_data_original - current_hvgs
    
    if not missing_canonical:
        print(f"       -> All {len(canonical_in_data_original)} available canonical markers already in HVGs")
        return list(current_hvgs)
    
    # Step 5: Determine how many to add
    max_to_add = max(int(n_hvg * max_marker_fraction), 50)
    markers_to_add = list(missing_canonical)[:max_to_add]
    
    print(f"       -> [MPS-HVG] Found {len(missing_canonical)} canonical markers not in initial HVGs")
    print(f"       -> [MPS-HVG] Adding up to {len(markers_to_add)} markers to HVG list")
    
    # Step 6: Remove lowest-variance HVGs to make room
    n_to_remove = len(markers_to_add)
    hvgs_to_keep = list(hvg_df.index[:n_hvg - n_to_remove])
    
    # Combine
    final_hvgs = hvgs_to_keep + markers_to_add
    
    print(f"       -> [MPS-HVG] Final HVG count: {len(final_hvgs)} "
          f"(includes {len(markers_to_add)} protected canonical markers)")
    
    return final_hvgs

def find_matching_cell_type_in_prior(
    query_cell_type: str, 
    prior_dict: dict, 
    min_similarity: float = 0.5,
    min_token_length: int = 4,
    verbose: bool = False
) -> tuple:
    """
    Finds the best matching cell type with STRICT threshold enforcement.
    
    ENHANCED: Now expands abbreviated cell type names before matching.
    
    Matching stages:
    1. Expand abbreviations (e.g., 'OPC' -> 'oligodendrocyte precursor cell')
    2. Exact match on expanded name
    3. Canonical synonym match
    4. Semantic token matching
    5. Whole-phrase substring match
    6. Fuzzy string matching
    
    ...existing docstring...
    """
    if not query_cell_type or not prior_dict:
        return None, "no_match", 0.0, set()
    
    # =========================================================================
    # NEW: EXPAND ABBREVIATIONS FIRST
    # =========================================================================
    query_expanded, was_expanded, query_original = expand_celltype_with_context(
        query_cell_type, verbose=verbose
    )
    
    if was_expanded and verbose:
        print(f"       [MPS] Expanded '{query_original}' -> '{query_expanded}'")
    
    # Use expanded form for all matching
    query_lower = query_expanded.lower().strip()
    query_normalized = normalize_cell_type_name(query_lower)
    
    # Also try matching with the original (unexpanded) form as fallback
    query_original_lower = query_cell_type.lower().strip()
    query_original_normalized = normalize_cell_type_name(query_original_lower)
    
    # Pre-compute normalized DB entries
    db_entries = {}
    for db_type in prior_dict.keys():
        db_lower = db_type.lower().strip()
        db_normalized = normalize_cell_type_name(db_lower)
        db_entries[db_type] = {
            'lower': db_lower,
            'normalized': db_normalized,
            'tokens': set(tokenize_cell_type(db_lower))
        }
    
    def extract_markers(marker_data):
        """Helper to extract markers from prior dict entry."""
        if isinstance(marker_data, dict):
            return marker_data.get('markers_standard', set())
        elif isinstance(marker_data, set):
            return marker_data
        return set()
    
    # ══════════════════════════════════════════════════════════════════
    # STRICT MODE CHECK: If threshold >= 0.99, ONLY exact matches
    # ══════════════════════════════════════════════════════════════════
    if min_similarity >= 0.99:
        for db_type, db_info in db_entries.items():
            if query_normalized == db_info['normalized']:
                return db_type, "exact", 1.0, extract_markers(prior_dict[db_type])
        return None, "no_match", 0.0, set()
    
    best_match = None
    best_score = 0.0
    best_method = "no_match"
    
    query_tokens = set(tokenize_cell_type(query_lower))
    query_tokens = {t for t in query_tokens if len(t) >= min_token_length}
    
    # ══════════════════════════════════════════════════════════════════
    # STAGE 1: Exact match (confidence = 1.0) - Always allowed
    # ══════════════════════════════════════════════════════════════════
    for db_type, db_info in db_entries.items():
        if query_normalized == db_info['normalized']:
            return db_type, "exact", 1.0, extract_markers(prior_dict[db_type])
    
    # ══════════════════════════════════════════════════════════════════
    # STAGE 2: Canonical synonym match (confidence = 0.90-0.95)
    # FIXED: Now respects threshold
    # ══════════════════════════════════════════════════════════════════
    CANONICAL_SYNONYMS = {
        # Cell type synonyms (NOT including location terms!)
        'opc': {
            'synonyms': ['oligodendrocyte precursor cell', 'oligodendrocyte progenitor cell', 
                        'oligodendrocyte precursor', 'ng2 cell', 'ng2+ cell'],
            'confidence': 0.95
        },
        'oligodendrocyte precursor cell': {
            'synonyms': ['opc', 'oligodendrocyte progenitor cell', 'ng2 cell'],
            'confidence': 0.95
        },
        'glioblast': {
            'synonyms': ['radial glia', 'radial glial cell', 'glial precursor'],
            'confidence': 0.90
        },
        'radial glia': {
            'synonyms': ['glioblast', 'radial glial cell', 'glial precursor'],
            'confidence': 0.90
        },
        'neuron': {
            'synonyms': ['neuronal cell', 'nerve cell'],
            'confidence': 0.95
        },
        'astrocyte': {
            'synonyms': ['astroglia', 'astroglial cell'],
            'confidence': 0.95
        },
        'microglia': {
            'synonyms': ['microglial cell', 'brain macrophage'],
            'confidence': 0.90
        },
        'neural crest cell': {
            'synonyms': ['neural crest', 'neural crest cells', 'ncc'],
            'confidence': 0.95
        },
        'neural crest': {
            'synonyms': ['neural crest cell', 'neural crest cells', 'ncc'],
            'confidence': 0.95
        },
    }
    
    # Extract the core cell type from query (remove location prefixes)
    LOCATION_TERMS = {'dorsal', 'ventral', 'lateral', 'medial', 'anterior', 'posterior',
                      'midbrain', 'forebrain', 'hindbrain', 'pons', 'cerebellum', 
                      'cortex', 'hippocampus', 'thalamus', 'hypothalamus', 'striatum'}
    
    query_words = query_normalized.split()
    query_cell_core = ' '.join([w for w in query_words if w not in LOCATION_TERMS]).strip()
    
    for canonical, syn_info in CANONICAL_SYNONYMS.items():
        confidence = syn_info['confidence']
        synonyms = syn_info['synonyms']
        
        # FIXED: Check if confidence meets threshold BEFORE matching
        if confidence < min_similarity:
            continue
            
        # Check if canonical matches query core (not full query with locations)
        if canonical == query_cell_core or canonical in query_cell_core:
            for db_type, db_info in db_entries.items():
                db_norm = db_info['normalized']
                
                # Direct match to canonical
                if db_norm == canonical:
                    if confidence >= min_similarity:
                        return db_type, "canonical_synonym", confidence, extract_markers(prior_dict[db_type])
                
                # Match to synonym
                for syn in synonyms:
                    if db_norm == syn:
                        syn_confidence = confidence - 0.05  # Slightly lower for synonym
                        if syn_confidence >= min_similarity:
                            return db_type, "canonical_synonym", syn_confidence, extract_markers(prior_dict[db_type])
    
    # ══════════════════════════════════════════════════════════════════
    # STAGE 3: Semantic token matching with CELL TYPE tokens only
    # FIXED: Excludes location-only matches
    # ══════════════════════════════════════════════════════════════════
    CORE_CELL_TOKENS = {
        'neuron', 'astrocyte', 'oligodendrocyte', 'microglia', 'glioblast',
        'opc', 'progenitor', 'precursor', 'stem', 'endothelial', 'epithelial',
        'fibroblast', 'macrophage', 'monocyte', 'lymphocyte', 'dendritic',
        'radial', 'glia', 'glial', 'neural', 'crest', 'interneuron',
        'excitatory', 'inhibitory', 'glutamatergic', 'gabaergic', 'cholinergic',
        'dopaminergic', 'serotonergic', 'purkinje', 'granule', 'pyramidal'
    }
    
    query_core_tokens = query_tokens.intersection(CORE_CELL_TOKENS)
    
    if query_core_tokens:  # Only proceed if query has cell-type tokens
        for db_type, db_info in db_entries.items():
            db_tokens = {t for t in db_info['tokens'] if len(t) >= min_token_length}
            db_core_tokens = db_tokens.intersection(CORE_CELL_TOKENS)
            
            # FIXED: DB entry must also have cell-type tokens
            if not db_core_tokens:
                continue
            
            core_overlap = query_core_tokens.intersection(db_core_tokens)
            
            if core_overlap:
                # Calculate score based on overlap
                overlap_ratio = len(core_overlap) / max(len(query_core_tokens), len(db_core_tokens))
                all_overlap = query_tokens.intersection(db_tokens)
                union = query_tokens.union(db_tokens)
                jaccard = len(all_overlap) / len(union) if union else 0
                
                # Combined score
                score = (overlap_ratio * 0.6) + (jaccard * 0.4)
                
                # FIXED: Apply threshold check
                if score > best_score and score >= min_similarity:
                    best_score = score
                    best_match = db_type
                    best_method = "token_semantic"
    
    # ══════════════════════════════════════════════════════════════════
    # STAGE 4: Whole-phrase substring match with LENGTH RATIO CHECK
    # FIXED: Prevents short DB entries matching long queries
    # ══════════════════════════════════════════════════════════════════
    MIN_LENGTH_RATIO = 0.4  # DB entry must be at least 40% length of query
    SUBSTRING_BASE_CONFIDENCE = 0.85
    
    if best_match is None and SUBSTRING_BASE_CONFIDENCE >= min_similarity:
        for db_type, db_info in db_entries.items():
            db_norm = db_info['normalized']
            
            # FIXED: Length ratio check prevents "dorsal" matching "dorsal midbrain opc"
            length_ratio = len(db_norm) / len(query_normalized) if query_normalized else 0
            
            if len(db_norm) < 5 or length_ratio < MIN_LENGTH_RATIO:
                continue
            
            # Check if DB entry is a whole-word substring of query
            if db_norm in query_normalized:
                idx = query_normalized.find(db_norm)
                before_ok = (idx == 0) or (not query_normalized[idx-1].isalnum())
                after_idx = idx + len(db_norm)
                after_ok = (after_idx == len(query_normalized)) or (not query_normalized[after_idx].isalnum())
                
                if before_ok and after_ok:
                    # Score based on how much of the query is covered
                    coverage = len(db_norm) / len(query_normalized)
                    score = SUBSTRING_BASE_CONFIDENCE * coverage
                    
                    # FIXED: Apply threshold check
                    if score > best_score and score >= min_similarity:
                        best_score = score
                        best_match = db_type
                        best_method = "whole_phrase_match"
    
    # ══════════════════════════════════════════════════════════════════
    # STAGE 5: Fuzzy string matching with STRICT checks
    # FIXED: Added bidirectional ratio and length checks
    # ══════════════════════════════════════════════════════════════════
    if best_match is None:
        try:
            from rapidfuzz import fuzz
            
            for db_type, db_info in db_entries.items():
                db_norm = db_info['normalized']
                
                # FIXED: Length ratio check
                len_query = len(query_normalized)
                len_db = len(db_norm)
                length_ratio = min(len_query, len_db) / max(len_query, len_db) if max(len_query, len_db) > 0 else 0
                
                # Skip if lengths are too different (prevents "dorsal" matching long strings)
                if length_ratio < 0.3:
                    continue
                
                # FIXED: Use multiple fuzzy metrics and take minimum
                token_set = fuzz.token_set_ratio(query_normalized, db_norm) / 100.0
                token_sort = fuzz.token_sort_ratio(query_normalized, db_norm) / 100.0
                partial = fuzz.partial_ratio(query_normalized, db_norm) / 100.0
                
                # Use weighted combination, penalizing partial matches
                ratio = (token_set * 0.4) + (token_sort * 0.4) + (partial * 0.2)
                
                # Apply length penalty for mismatched lengths
                ratio = ratio * (0.5 + 0.5 * length_ratio)
                
                if ratio > best_score and ratio >= min_similarity:
                    best_score = ratio
                    best_match = db_type
                    best_method = "fuzzy_match"
                    
        except ImportError:
            pass
    
    # ══════════════════════════════════════════════════════════════════
    # Return result
    # ══════════════════════════════════════════════════════════════════
    if best_match:
        return best_match, best_method, best_score, extract_markers(prior_dict[best_match])
    
    return None, "no_match", 0.0, set()

def normalize_cell_type_name(name: str) -> str:
    """Normalize cell type name for matching."""
    import re
    
    # Lowercase
    name = name.lower().strip()
    
    # Expand common abbreviations
    abbreviations = {
        'opc': 'oligodendrocyte precursor cell',
        'npc': 'neural progenitor cell',
        'nsc': 'neural stem cell',
        'rg': 'radial glia',
        'bfcn': 'basal forebrain cholinergic neuron',
    }
    
    for abbr, full in abbreviations.items():
        # Match whole word only
        name = re.sub(rf'\b{abbr}\b', full, name)
    
    # Remove parenthetical content
    name = re.sub(r'\([^)]*\)', '', name)
    
    # Remove extra whitespace
    name = ' '.join(name.split())
    
    return name


def tokenize_cell_type(name: str) -> list:
    """Tokenize cell type name into meaningful tokens."""
    import re
    
    # Lowercase and split on non-alphanumeric
    tokens = re.split(r'[^a-z0-9]+', name.lower())
    
    # Remove empty and very short tokens
    tokens = [t for t in tokens if len(t) >= 2]
    
    # Remove common stopwords
    stopwords = {'cell', 'cells', 'type', 'like', 'the', 'and', 'of', 'in', 'on'}
    tokens = [t for t in tokens if t not in stopwords]
    
    return tokens

def diagnose_celltype_matching(adata, prior_dict: dict, annotation_col: str = 'ctpt_consensus_prediction',
                                output_path: str = None) -> pd.DataFrame:
    """
    Diagnoses cell type matching between annotated cell types and the marker prior database.
    
    Produces a detailed report showing:
    - Which cell types matched and how (exact, synonym, fuzzy, etc.)
    - Which cell types failed to match
    - Suggestions for improving matches
    
    Args:
        adata: AnnData object with cell type annotations
        prior_dict: Marker prior dictionary
        annotation_col: Column containing cell type annotations
        output_path: Optional path to save diagnostics CSV
    
    Returns:
        pd.DataFrame with matching diagnostics
    """
    print(f"\n--- Cell Type Matching Diagnostics ---")
    
    if annotation_col not in adata.obs.columns:
        print(f"[ERROR] Annotation column '{annotation_col}' not found in adata.obs")
        return pd.DataFrame()
    
    unique_types = adata.obs[annotation_col].dropna().unique()
    diagnostics = []
    
    for cell_type in unique_types:
        cell_type_str = str(cell_type)
        n_cells = (adata.obs[annotation_col] == cell_type).sum()
        
        # =====================================================================
        # NEW: Expand abbreviations and show both forms in diagnostics
        # =====================================================================
        cell_type_expanded, was_expanded, _ = expand_celltype_with_context(cell_type_str)
        
        # =====================================================================
        # FIX: Unpack all 4 return values from find_matching_cell_type_in_prior
        # =====================================================================
        matched_key, match_method, match_confidence, matched_markers = find_matching_cell_type_in_prior(
            cell_type_str, prior_dict, verbose=False
        )
        
        # Get standardized forms for comparison
        query_standard = standardize_celltype_name(cell_type_str)
        query_canonical = get_canonical_celltype(cell_type_str)
        
        # =====================================================================
        # FIX: Update logic to use the correct variable names
        # =====================================================================
        if matched_key is not None and matched_markers:
            n_markers = len(matched_markers)
            # Get original name from prior_dict if available
            marker_data = prior_dict.get(matched_key, {})
            if isinstance(marker_data, dict):
                matched_original = marker_data.get('original_name', matched_key)
            else:
                matched_original = matched_key
            status = 'MATCHED'
        else:
            n_markers = 0
            matched_original = None
            match_method = 'NO_MATCH' if match_method is None else match_method
            match_confidence = 0.0
            status = 'UNMATCHED'
        
        diagnostics.append({
            'Query_Cell_Type': cell_type_str,
            'Expanded_Form': cell_type_expanded if was_expanded else 'N/A',  # NEW
            'Was_Expanded': was_expanded,  # NEW
            'N_Cells': n_cells,
            'Standardized_Query': query_standard,
            'Canonical_Form': query_canonical if query_canonical else 'N/A',
            'Match_Status': status,
            'Match_Method': match_method if match_method else 'N/A',
            'Match_Confidence': f"{match_confidence:.2f}" if match_confidence else '0.00',  # NEW
            'Matched_DB_Key': matched_key if matched_key else 'N/A',
            'Matched_DB_Original': matched_original if matched_original else 'N/A',
            'N_Markers_Available': n_markers
        })
    
    df = pd.DataFrame(diagnostics)
    df = df.sort_values(['Match_Status', 'N_Cells'], ascending=[True, False])
    
    # Print summary
    n_matched = (df['Match_Status'] == 'MATCHED').sum()
    n_unmatched = (df['Match_Status'] == 'UNMATCHED').sum()
    cells_matched = df[df['Match_Status'] == 'MATCHED']['N_Cells'].sum()
    cells_unmatched = df[df['Match_Status'] == 'UNMATCHED']['N_Cells'].sum()
    total_cells = cells_matched + cells_unmatched
    
    print(f"       -> Cell types matched: {n_matched}/{len(df)} ({n_matched/len(df)*100:.1f}%)")
    print(f"       -> Cells with matched types: {cells_matched}/{total_cells} ({cells_matched/total_cells*100:.1f}%)")
    
    if n_unmatched > 0:
        print(f"\n       -> Unmatched cell types ({n_unmatched}):")
        unmatched = df[df['Match_Status'] == 'UNMATCHED']
        for _, row in unmatched.iterrows():
            print(f"          - {row['Query_Cell_Type']} ({row['N_Cells']} cells)")
            print(f"            Standardized: '{row['Standardized_Query']}'")
            if row['Canonical_Form'] != 'N/A':
                print(f"            Known canonical: '{row['Canonical_Form']}'")
    
    # Show match methods used
    if n_matched > 0:
        print(f"\n       -> Match methods used:")
        method_counts = df[df['Match_Status'] == 'MATCHED']['Match_Method'].value_counts()
        for method, count in method_counts.items():
            print(f"          - {method}: {count}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n       -> Saved diagnostics to: {output_path}")
    
    return df
# ==============================================================================
# INSERT NEW FUNCTION HERE - AFTER diagnose_celltype_matching
# ==============================================================================

def export_celltype_marker_details(
    adata,
    prior_dict: dict,
    output_dir: str,
    prefix: str,
    groupby_key: str = 'ctpt_consensus_prediction',
    n_top_genes: int = 50,
    deg_ranking_method: str = 'original',
    deg_weight_fc: float = 0.4,
    deg_weight_expr: float = 0.3,
    deg_weight_pct: float = 0.3,
    species: str = "human"
):
    """
    Exports detailed cell type matching and marker gene information.
    
    Outputs:
    1. {prefix}_celltype_matching_summary.csv - Cell type to database matching details
    2. {prefix}_celltype_top_markers.csv - Top N marker genes per cell type (ranked)
    3. {prefix}_celltype_canonical_overlap.csv - Overlap between DEGs and canonical markers
    4. {prefix}_celltype_hvg_genes.csv - Top N HVG genes per cell type
    
    Args:
        adata: AnnData object with annotations and DEG results
        prior_dict: Marker prior dictionary
        output_dir: Output directory path
        prefix: File prefix
        groupby_key: Column containing cell type annotations
        n_top_genes: Number of top marker genes to export per cell type
        deg_ranking_method: 'original' (logFC) or 'composite'
        deg_weight_fc, deg_weight_expr, deg_weight_pct: Weights for composite ranking
        species: Species for gene name standardization
    
    Returns:
        dict: Paths to exported files
    """
    import scanpy as sc
    
    print(f"\n{'='*70}")
    print(f"--- Exporting Cell Type Marker Details ---")
    print(f"{'='*70}")
    print(f"    Output directory: {output_dir}")
    print(f"    Ranking method: {deg_ranking_method}")
    print(f"    Top N genes: {n_top_genes}")
    
    exported_files = {}
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate groupby_key exists
    if groupby_key not in adata.obs.columns:
        print(f"[ERROR] Column '{groupby_key}' not found in adata.obs")
        return exported_files
    
    # Get unique cell types with cells
    celltype_counts = adata.obs[groupby_key].value_counts()
    unique_celltypes = celltype_counts[celltype_counts > 0].index.tolist()
    print(f"    Found {len(unique_celltypes)} unique cell types with cells")
    
    # =========================================================================
    # 1. Run DEG analysis if not already done
    # =========================================================================
    deg_key = f'rank_genes_{groupby_key}'
    
    valid_celltypes = celltype_counts[celltype_counts >= 3].index.tolist()
    
    if len(valid_celltypes) >= 2:
        print(f"\n    [Step 1] Running/retrieving DEG analysis...")
        try:
            sc.tl.rank_genes_groups(
                adata,
                groupby=groupby_key,
                groups=valid_celltypes,
                reference='rest',
                method='wilcoxon',
                pts=True,
                key_added=deg_key
            )
        except Exception as e:
            print(f"    [WARNING] DEG analysis failed: {e}")
            deg_key = None
    else:
        print(f"    [WARNING] Not enough cell types for DEG analysis (need >= 2, have {len(valid_celltypes)})")
        deg_key = None
    
    # =========================================================================
    # 2. Extract and rank DEGs
    # =========================================================================
    marker_records = []
    hvg_records = []
    
    if deg_key and deg_key in adata.uns:
        print(f"\n    [Step 2] Extracting and ranking marker genes...")
        
        try:
            marker_df = sc.get.rank_genes_groups_df(adata, group=None, key=deg_key)
            
            # Apply composite ranking if requested
            if deg_ranking_method == 'composite':
                print(f"       -> Applying composite ranking (FC:{deg_weight_fc}, Expr:{deg_weight_expr}, Pct:{deg_weight_pct})")
                marker_df = apply_composite_deg_ranking(
                    marker_df=marker_df,
                    deg_weight_fc=deg_weight_fc,
                    deg_weight_expr=deg_weight_expr,
                    deg_weight_pct=deg_weight_pct
                )
                rank_col = 'composite_score'
            else:
                rank_col = 'logfoldchanges'
            
            # Process each cell type
            for celltype in unique_celltypes:
                ct_df = marker_df[marker_df['group'] == celltype].copy()
                
                if ct_df.empty:
                    continue
                
                # Sort by ranking column
                if rank_col in ct_df.columns:
                    ct_df = ct_df.sort_values(rank_col, ascending=False)
                
                # Get top N genes
                top_genes = ct_df.head(n_top_genes)
                
                for rank_idx, (_, row) in enumerate(top_genes.iterrows(), 1):
                    record = {
                        'Cell_Type': celltype,
                        'Rank': rank_idx,
                        'Gene': row['names'],
                        'Log2FC': row.get('logfoldchanges', np.nan),
                        'P_Value': row.get('pvals', np.nan),
                        'P_Value_Adj': row.get('pvals_adj', np.nan),
                        'Pct_In_Group': row.get('pct_nz_group', np.nan),
                        'Pct_In_Reference': row.get('pct_nz_reference', np.nan),
                    }
                    
                    if deg_ranking_method == 'composite' and 'composite_score' in row:
                        record['Composite_Score'] = row['composite_score']
                    
                    marker_records.append(record)
                    
        except Exception as e:
            print(f"    [WARNING] Failed to extract DEG results: {e}")
    
    # =========================================================================
    # 3. Cell Type Matching Summary
    # =========================================================================
    print(f"\n    [Step 3] Generating cell type matching summary...")
    
    matching_records = []
    overlap_records = []
    
    # Create matcher for robust matching
    if prior_dict:
        matcher = create_robust_cell_type_matcher(prior_dict, similarity_threshold=0.6)
    else:
        matcher = None
    
    for celltype in unique_celltypes:
        n_cells = int(celltype_counts[celltype])
        
        # Get matching info
        if matcher:
            markers, matched_db, match_method, confidence = get_markers_for_cell_type(
                celltype, matcher, verbose=False
            )
        else:
            markers, matched_db, match_method, confidence = set(), None, None, 0.0
        
        # Get top DEGs for this cell type
        ct_top_genes = []
        if marker_records:
            ct_top_genes = [r['Gene'] for r in marker_records if r['Cell_Type'] == celltype][:n_top_genes]
        
        ct_top_genes_set = set(ct_top_genes)
        
        # Calculate overlap with canonical markers
        if markers:
            overlap_genes = markers.intersection(ct_top_genes_set)
            n_overlap = len(overlap_genes)
            recall = n_overlap / len(markers) if markers else 0.0
        else:
            overlap_genes = set()
            n_overlap = 0
            recall = 0.0
        
        matching_records.append({
            'Cell_Type': celltype,
            'N_Cells': n_cells,
            'Matched_DB_Entry': matched_db if matched_db else 'NO_MATCH',
            'Match_Method': match_method if match_method else 'N/A',
            'Match_Confidence': f"{confidence:.2f}",
            'N_Canonical_Markers': len(markers) if markers else 0,
            'N_DEGs_in_Top': len(ct_top_genes),
            'N_Overlap': n_overlap,
            'Recall': f"{recall:.2%}",
            'Overlapping_Genes': '; '.join(sorted(overlap_genes)) if overlap_genes else ''
        })
        
        # Detailed overlap record
        if markers:
            deg_only = ct_top_genes_set - markers
            canonical_only = markers - ct_top_genes_set
            
            overlap_records.append({
                'Cell_Type': celltype,
                'N_Cells': n_cells,
                'Matched_DB_Entry': matched_db if matched_db else 'NO_MATCH',
                'N_Canonical_Markers': len(markers),
                'N_Top_DEGs': len(ct_top_genes),
                'N_Overlap': n_overlap,
                'Recall_Pct': f"{recall:.2%}",
                'Overlapping_Genes': '; '.join(sorted(overlap_genes)),
                'DEG_Only_Genes': '; '.join(sorted(list(deg_only)[:20])) + ('...' if len(deg_only) > 20 else ''),
                'Canonical_Only_Genes': '; '.join(sorted(list(canonical_only)[:20])) + ('...' if len(canonical_only) > 20 else '')
            })
    
    # =========================================================================
    # 4. HVG per cell type (using mean expression in each cell type)
    # =========================================================================
    print(f"\n    [Step 4] Calculating top expressed genes per cell type...")
    
    try:
        # Use raw data if available
        adata_expr = adata.raw.to_adata() if adata.raw is not None else adata
        
        for celltype in unique_celltypes:
            ct_mask = adata_expr.obs[groupby_key] == celltype
            if ct_mask.sum() < 3:
                continue
            
            ct_adata = adata_expr[ct_mask]
            
            # Calculate mean expression per gene
            if hasattr(ct_adata.X, 'toarray'):
                mean_expr = np.array(ct_adata.X.toarray().mean(axis=0)).flatten()
            else:
                mean_expr = np.array(ct_adata.X.mean(axis=0)).flatten()
            
            # Get top N by expression
            top_indices = np.argsort(mean_expr)[::-1][:n_top_genes]
            
            for rank_idx, gene_idx in enumerate(top_indices, 1):
                hvg_records.append({
                    'Cell_Type': celltype,
                    'Rank': rank_idx,
                    'Gene': adata_expr.var_names[gene_idx],
                    'Mean_Expression': mean_expr[gene_idx]
                })
                
    except Exception as e:
        print(f"    [WARNING] Failed to calculate HVG per cell type: {e}")
    
    # =========================================================================
    # 5. Save all CSVs
    # =========================================================================
    print(f"\n    [Step 5] Saving output files...")
    
    # 5a. Top markers per cell type
    if marker_records:
        markers_df = pd.DataFrame(marker_records)
        markers_path = os.path.join(output_dir, f"{prefix}_celltype_top_markers.csv")
        markers_df.to_csv(markers_path, index=False)
        exported_files['top_markers'] = markers_path
        print(f"       -> Saved: {markers_path}")
    
    # 5b. Cell type matching summary
    if matching_records:
        matching_df = pd.DataFrame(matching_records)
        matching_df = matching_df.sort_values('N_Cells', ascending=False)
        matching_path = os.path.join(output_dir, f"{prefix}_celltype_matching_summary.csv")
        matching_df.to_csv(matching_path, index=False)
        exported_files['matching_summary'] = matching_path
        print(f"       -> Saved: {matching_path}")
    
    # 5c. Canonical marker overlap
    if overlap_records:
        overlap_df = pd.DataFrame(overlap_records)
        overlap_df = overlap_df.sort_values('N_Cells', ascending=False)
        overlap_path = os.path.join(output_dir, f"{prefix}_celltype_canonical_overlap.csv")
        overlap_df.to_csv(overlap_path, index=False)
        exported_files['canonical_overlap'] = overlap_path
        print(f"       -> Saved: {overlap_path}")
    
    # 5d. HVG per cell type
    if hvg_records:
        hvg_df = pd.DataFrame(hvg_records)
        hvg_path = os.path.join(output_dir, f"{prefix}_celltype_hvg_genes.csv")
        hvg_df.to_csv(hvg_path, index=False)
        exported_files['hvg_genes'] = hvg_path
        print(f"       -> Saved: {hvg_path}")
    
    print(f"\n    ✅ Exported {len(exported_files)} files to {output_dir}")
    
    return exported_files

def generate_consistent_cells_dotplot(
    adata_filtered,
    output_dir: str,
    prefix: str,
    annotation_col: str = 'ctpt_consensus_prediction',
    n_top_genes: int = 5,
    figsize: tuple = None,
    dpi: int = 500
):
    """
    Generates a dotplot showing top marker genes for each cell type in the filtered consistent cells.
    Uses EXACT same parameters as the standalone script to match stage_2_final_analysis output.
    
    Args:
        adata_filtered: AnnData object containing filtered consistent cells
        output_dir: Output directory path
        prefix: File prefix
        annotation_col: Column containing cell type annotations
        n_top_genes: Number of top marker genes per cell type to show
        figsize: Figure size (width, height) - auto-calculated if None
        dpi: Figure resolution (default 500 to match Stage 2)
    
    Returns:
        str: Path to saved dotplot, or None if failed
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    import os
    import re
    
    print(f"\n--- Generating Standard Dotplot for Filtered Consistent Cells ---")
    print(f"    Cells: {adata_filtered.n_obs}")
    print(f"    Cell types: {adata_filtered.obs[annotation_col].nunique()}")
    print(f"    Top genes per type: {n_top_genes}")
    
    # Validate inputs
    if annotation_col not in adata_filtered.obs.columns:
        print(f"[ERROR] Column '{annotation_col}' not found in adata.obs")
        return None
    
    # Set scanpy figure parameters to match Stage 2
    sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False, dpi_save=dpi)
    sc.settings.figdir = output_dir
    
    # Clean up categories
    adata_filtered.obs[annotation_col] = adata_filtered.obs[annotation_col].astype('category')
    adata_filtered.obs[annotation_col] = adata_filtered.obs[annotation_col].cat.remove_unused_categories()
    
    # Get valid labels (>1 cell, matching Stage 2 logic)
    celltype_counts = adata_filtered.obs[annotation_col].value_counts()
    valid_celltypes = celltype_counts[celltype_counts > 1].index.tolist()
    
    if len(valid_celltypes) < 2:
        print(f"[WARNING] Need at least 2 cell types for DEG analysis, found {len(valid_celltypes)}")
        return None
    
    print(f"    Valid cell types (>1 cell): {len(valid_celltypes)}")
    
    # Run rank_genes_groups with EXACT Stage 2 parameters
    deg_key = f'wilcoxon_{annotation_col}'
    
    try:
        print("    Running differential expression analysis (wilcoxon, use_raw=True)...")
        sc.tl.rank_genes_groups(
            adata_filtered,
            groupby=annotation_col,
            groups=valid_celltypes,
            method='wilcoxon',
            use_raw=True,  # CRITICAL: Match Stage 2
            key_added=deg_key
        )
    except Exception as e:
        print(f"[ERROR] DEG analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Extract top marker genes with mitochondrial filtering (EXACT Stage 2 logic)
    try:
        marker_df = sc.get.rank_genes_groups_df(adata_filtered, group=None, key=deg_key)
        
        # Filter mitochondrial genes (same as Stage 2)
        is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
        
        filtered_rows = []
        for _, sub in marker_df.groupby('group', sort=False):
            non_mito = sub[~sub['names'].apply(is_mito)].head(n_top_genes)
            filtered_rows.append(non_mito)
        
        top_genes_df = pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame()
        
        # Get unique markers preserving order
        unique_markers = []
        seen = set()
        for gene in top_genes_df['names'].tolist():
            if gene not in seen:
                seen.add(gene)
                unique_markers.append(gene)
        
        print(f"    Selected {len(unique_markers)} unique marker genes (mito-filtered)")
        
        if len(unique_markers) == 0:
            print("[WARNING] No marker genes found for dotplot")
            return None
        
    except Exception as e:
        print(f"[ERROR] Failed to extract marker genes: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Generate dotplot with EXACT Stage 2 parameters
    try:
        print(f"    Generating dotplot...")
        
        # Use exact same plt.rc_context as Stage 2
        with plt.rc_context({
            'font.size': 18, 
            'font.weight': 'bold', 
            'axes.labelweight': 'bold', 
            'axes.titleweight': 'bold'
        }):
            sc.pl.dotplot(
                adata_filtered,
                var_names=unique_markers,
                groupby=annotation_col,
                categories_order=valid_celltypes,
                use_raw=True,  # CRITICAL: Match Stage 2
                dendrogram=False,
                standard_scale='var',
                cmap='Reds',
                show=False,
                save=f"_{prefix}_consistent_cells_filtered_dotplot.png"
            )
        
        # Construct the actual output path (scanpy adds 'dotplot' prefix)
        dotplot_path = os.path.join(output_dir, f"dotplot_{prefix}_consistent_cells_filtered_dotplot.png")
        
        print(f"    ✅ Saved standard dotplot to: {dotplot_path}")
        return dotplot_path
        
    except Exception as e:
        print(f"[ERROR] Failed to generate dotplot: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_consistent_cells_dotplot_by_category(
    adata_filtered,
    output_dir: str,
    prefix: str,
    annotation_col: str = 'ctpt_consensus_prediction',
    n_top_genes: int = 3,
    figsize: tuple = None,
    dpi: int = 500
):
    """
    Generates a CATEGORIZED dotplot with genes grouped by cell type.
    Uses EXACT same parameters as stage_2_final_analysis for uniform formatting.
    
    This creates a dotplot where genes are organized into categories (one per cell type),
    making it easier to see which markers are specific to which cell type.
    
    Args:
        adata_filtered: AnnData object containing filtered consistent cells
        output_dir: Output directory path
        prefix: File prefix
        annotation_col: Column containing cell type annotations
        n_top_genes: Number of top marker genes per cell type category
        figsize: Figure size (width, height) - auto-calculated if None
        dpi: Figure resolution (default 500 to match Stage 2)
    
    Returns:
        str: Path to saved dotplot, or None if failed
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    import os
    import re
    from collections import OrderedDict
    
    print(f"\n--- Generating Categorized Dotplot for Filtered Consistent Cells ---")
    print(f"    Cells: {adata_filtered.n_obs}")
    print(f"    Cell types: {adata_filtered.obs[annotation_col].nunique()}")
    print(f"    Top genes per category: {n_top_genes}")
    
    # Validate inputs
    if annotation_col not in adata_filtered.obs.columns:
        print(f"[ERROR] Column '{annotation_col}' not found in adata.obs")
        return None
    
    # Set scanpy figure parameters to match Stage 2
    sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False, dpi_save=dpi)
    sc.settings.figdir = output_dir
    
    # Clean up categories
    adata_filtered.obs[annotation_col] = adata_filtered.obs[annotation_col].astype('category')
    adata_filtered.obs[annotation_col] = adata_filtered.obs[annotation_col].cat.remove_unused_categories()
    
    # Get valid labels (>1 cell, matching Stage 2 logic)
    celltype_counts = adata_filtered.obs[annotation_col].value_counts()
    valid_celltypes = celltype_counts[celltype_counts > 1].index.tolist()
    
    if len(valid_celltypes) < 2:
        print(f"[WARNING] Need at least 2 cell types for DEG analysis, found {len(valid_celltypes)}")
        return None
    
    print(f"    Valid cell types (>1 cell): {len(valid_celltypes)}")
    
    # Run rank_genes_groups with EXACT Stage 2 parameters
    deg_key = f'wilcoxon_categorized_{annotation_col}'
    
    try:
        print("    Running differential expression analysis (wilcoxon, use_raw=True)...")
        sc.tl.rank_genes_groups(
            adata_filtered,
            groupby=annotation_col,
            groups=valid_celltypes,
            method='wilcoxon',
            use_raw=True,  # CRITICAL: Match Stage 2
            key_added=deg_key
        )
    except Exception as e:
        print(f"[ERROR] DEG analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Build categorized marker dictionary
    try:
        marker_df = sc.get.rank_genes_groups_df(adata_filtered, group=None, key=deg_key)
        
        # Filter mitochondrial genes
        is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
        
        # Build OrderedDict for categorized dotplot
        marker_dict = OrderedDict()
        all_selected_genes = set()
        
        for celltype in valid_celltypes:
            ct_df = marker_df[marker_df['group'] == celltype].copy()
            
            # Filter mito genes
            ct_df = ct_df[~ct_df['names'].apply(is_mito)]
            
            # Sort by score (descending) and get top N
            ct_df = ct_df.sort_values('scores', ascending=False)
            
            # Select top N genes that haven't been selected yet (to avoid duplicates)
            selected_genes = []
            for gene in ct_df['names'].tolist():
                if gene not in all_selected_genes and len(selected_genes) < n_top_genes:
                    selected_genes.append(gene)
                    all_selected_genes.add(gene)
            
            if selected_genes:
                marker_dict[str(celltype)] = selected_genes
        
        print(f"    Built categorized marker dict with {len(marker_dict)} categories")
        print(f"    Total unique genes: {len(all_selected_genes)}")
        
        if len(marker_dict) == 0:
            print("[WARNING] No marker categories found for categorized dotplot")
            return None
        
    except Exception as e:
        print(f"[ERROR] Failed to build marker categories: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Generate categorized dotplot with EXACT Stage 2 parameters
    try:
        print(f"    Generating categorized dotplot...")
        
        # Use exact same plt.rc_context as Stage 2
        with plt.rc_context({
            'font.size': 18, 
            'font.weight': 'bold', 
            'axes.labelweight': 'bold', 
            'axes.titleweight': 'bold'
        }):
            sc.pl.dotplot(
                adata_filtered,
                var_names=marker_dict,  # OrderedDict creates categories
                groupby=annotation_col,
                categories_order=valid_celltypes,
                use_raw=True,  # CRITICAL: Match Stage 2
                dendrogram=False,  # No dendrogram for categorized
                standard_scale='var',
                cmap='Reds',
                show=False,
                save=f"_{prefix}_consistent_cells_filtered_categorized_dotplot.png"
            )
        
        # Construct the actual output path
        dotplot_path = os.path.join(
            output_dir, 
            f"dotplot_{prefix}_consistent_cells_filtered_categorized_dotplot.png"
        )
        
        print(f"    ✅ Saved categorized dotplot to: {dotplot_path}")
        
        # Also export the marker list as CSV (MCS-style format)
        mcs_style_data = []
        for celltype, genes in marker_dict.items():
            for rank, gene in enumerate(genes, 1):
                mcs_style_data.append({
                    'Cell_Type': celltype,
                    'Rank': rank,
                    'Gene': gene
                })
        
        mcs_style_df = pd.DataFrame(mcs_style_data)
        mcs_style_csv = os.path.join(
            output_dir, 
            f"{prefix}_consistent_cells_filtered_categorized_markers.csv"
        )
        mcs_style_df.to_csv(mcs_style_csv, index=False)
        print(f"    ✅ Saved MCS-style marker list to: {mcs_style_csv}")
        
        return dotplot_path
        
    except Exception as e:
        print(f"[ERROR] Failed to generate categorized dotplot: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# ==============================================================================
# NEW FUNCTION: MARKER-BASED AUTOMATIC ANNOTATION WITH SCORING
# ==============================================================================
def annotate_filtered_consistent_cells_by_markers(
    adata_consistent_filtered,
    prior_dict: dict,
    output_dir: str,
    prefix: str,
    annotation_col: str = 'ctpt_consensus_prediction',
    n_top_genes: int = 50,
    min_overlap_score: float = 0.05,
    species: str = "human",
    show_top_n_matches: int = 5
):
    """
    Performs marker-based annotation on the SAME filtered consistent cells
    used in export_consistent_cells, enabling direct UMAP comparison.
    
    This function:
    1. Uses the exact same cells as Br_consistent_cells_Refinement_ThreeWay_filtered_umap.png
    2. Runs DEG analysis on these cells
    3. Matches DEGs against canonical markers from prior database
    4. Assigns marker-based annotations
    5. Generates a UMAP that is directly comparable to the filtered consistent cells UMAP
    
    Parameters
    ----------
    adata_consistent_filtered : AnnData
        The filtered consistent cells AnnData object (same as used for filtered UMAP).
    prior_dict : dict
        Marker prior dictionary from load_marker_prior_database().
    output_dir : str
        Output directory (will create 'marker_based_annotation' subdirectory).
    prefix : str
        File prefix for outputs.
    annotation_col : str
        Column containing current cell type annotations.
    n_top_genes : int
        Number of top DEG markers to consider per cluster.
    min_overlap_score : float
        Minimum overlap score to consider a match valid.
    species : str
        Species for gene name standardization.
    show_top_n_matches : int
        Number of top alternative matches to show in output.
    
    Returns
    -------
    dict
        Contains:
        - 'adata': AnnData with 'marker_based_annotation' column added
        - 'annotation_df': DataFrame with detailed results
        - 'output_paths': dict of output file paths
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*70}")
    print(f"--- MARKER-BASED ANNOTATION FOR FILTERED CONSISTENT CELLS ---")
    print(f"{'='*70}")
    print(f"    Input cells: {adata_consistent_filtered.n_obs}")
    print(f"    Input cell types: {adata_consistent_filtered.obs[annotation_col].nunique()}")
    print(f"    Annotation column: {annotation_col}")
    
    # Create output subdirectory
    marker_anno_dir = os.path.join(output_dir, "marker_based_annotation")
    os.makedirs(marker_anno_dir, exist_ok=True)
    print(f"    Output directory: {marker_anno_dir}")
    
    output_paths = {}
    
    # Work on a copy to avoid modifying the original
    adata_work = adata_consistent_filtered.copy()
    
    # Validate inputs
    if annotation_col not in adata_work.obs.columns:
        print(f"[ERROR] Column '{annotation_col}' not found in adata.obs")
        return None
    
    if not prior_dict:
        print(f"[ERROR] Empty prior dictionary. Cannot perform marker-based annotation.")
        return None
    
    # Get unique cell types
    celltype_counts = adata_work.obs[annotation_col].value_counts()
    valid_celltypes = celltype_counts[celltype_counts >= 3].index.tolist()
    
    print(f"    Valid cell types (>= 3 cells): {len(valid_celltypes)}")
    
    if len(valid_celltypes) < 2:
        print(f"[ERROR] Need at least 2 valid cell types for DEG analysis.")
        return None
    
    # =========================================================================
    # Step 1: Run DEG analysis on filtered consistent cells
    # =========================================================================
    print(f"\n    [Step 1] Running DEG analysis on filtered consistent cells...")
    
    deg_key = f'rank_genes_marker_filtered_{annotation_col}'
    
    try:
        sc.tl.rank_genes_groups(
            adata_work,
            groupby=annotation_col,
            groups=valid_celltypes,
            reference='rest',
            method='wilcoxon',
            pts=True,
            key_added=deg_key
        )
    except Exception as e:
        print(f"[ERROR] DEG analysis failed: {e}")
        return None
    
    # Extract DEG results
    try:
        marker_df = sc.get.rank_genes_groups_df(adata_work, group=None, key=deg_key)
    except Exception as e:
        print(f"[ERROR] Failed to extract DEG results: {e}")
        return None
    
    # =========================================================================
    # Step 2: Match each cell type against marker database
    # =========================================================================
    print(f"\n    [Step 2] Matching cell types against marker database...")
    
    # Create matcher
    matcher = create_robust_cell_type_matcher(prior_dict, similarity_threshold=0.6)
    
    annotation_records = []
    all_matches_records = []
    
    for celltype in valid_celltypes:
        celltype_str = str(celltype)
        n_cells = int(celltype_counts[celltype])
        
        # Get top DEGs for this cell type
        ct_df = marker_df[marker_df['group'] == celltype].copy()
        if ct_df.empty:
            continue
        
        ct_df = ct_df.sort_values('logfoldchanges', ascending=False)
        top_genes = ct_df.head(n_top_genes)['names'].tolist()
        top_genes_set = {standardize_gene_name(g, species) for g in top_genes}
        
        # Calculate overlap scores for ALL cell types in database
        all_scores = []
        
        for db_type, db_data in prior_dict.items():
            if isinstance(db_data, dict):
                db_markers = db_data.get('markers_standard', set())
            elif isinstance(db_data, set):
                db_markers = db_data
            else:
                continue
            
            if not db_markers:
                continue
            
            # Calculate overlap
            overlap = top_genes_set.intersection(db_markers)
            n_overlap = len(overlap)
            
            # Calculate precision, recall, F1
            precision = n_overlap / len(top_genes_set) if top_genes_set else 0.0
            recall = n_overlap / len(db_markers) if db_markers else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            all_scores.append({
                'db_type': db_type,
                'n_overlap': n_overlap,
                'n_db_markers': len(db_markers),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'overlap_genes': overlap
            })
        
        # Sort by F1 score
        all_scores.sort(key=lambda x: x['f1'], reverse=True)
        
        # Get best match
        if all_scores and all_scores[0]['f1'] >= min_overlap_score:
            best_match = all_scores[0]
            marker_annotation = best_match['db_type']
            confidence = best_match['f1']
        else:
            marker_annotation = f"Unknown ({celltype_str})"
            confidence = 0.0
            best_match = None
        
        # Store annotation record
        annotation_records.append({
            'Original_Cell_Type': celltype_str,
            'N_Cells': n_cells,
            'Marker_Based_Annotation': marker_annotation,
            'Confidence_F1': f"{confidence:.3f}",
            'N_Overlap': best_match['n_overlap'] if best_match else 0,
            'N_DB_Markers': best_match['n_db_markers'] if best_match else 0,
            'Precision': f"{best_match['precision']:.3f}" if best_match else '0.000',
            'Recall': f"{best_match['recall']:.3f}" if best_match else '0.000',
            'Top_Overlap_Genes': '; '.join(sorted(list(best_match['overlap_genes'])[:10])) if best_match and best_match['overlap_genes'] else ''
        })
        
        # Store top N matches for detailed report
        for rank, match in enumerate(all_scores[:show_top_n_matches], 1):
            all_matches_records.append({
                'Original_Cell_Type': celltype_str,
                'Match_Rank': rank,
                'Matched_DB_Type': match['db_type'],
                'F1_Score': f"{match['f1']:.3f}",
                'Precision': f"{match['precision']:.3f}",
                'Recall': f"{match['recall']:.3f}",
                'N_Overlap': match['n_overlap'],
                'N_DB_Markers': match['n_db_markers'],
                'Overlap_Genes': '; '.join(sorted(list(match['overlap_genes'])[:15]))
            })
    
    # =========================================================================
    # Step 3: Create mapping and apply to adata
    # =========================================================================
    print(f"\n    [Step 3] Applying marker-based annotations...")
    
    # Create mapping from original to marker-based annotation
    annotation_mapping = {
        rec['Original_Cell_Type']: rec['Marker_Based_Annotation'] 
        for rec in annotation_records
    }
    
    # Apply mapping
    adata_work.obs['marker_based_annotation'] = adata_work.obs[annotation_col].astype(str).map(annotation_mapping)
    
    # Handle unmapped (shouldn't happen, but just in case)
    unmapped_mask = adata_work.obs['marker_based_annotation'].isna()
    if unmapped_mask.any():
        adata_work.obs.loc[unmapped_mask, 'marker_based_annotation'] = \
            adata_work.obs.loc[unmapped_mask, annotation_col].astype(str).apply(lambda x: f"Unknown ({x})")
    
    adata_work.obs['marker_based_annotation'] = adata_work.obs['marker_based_annotation'].astype('category')
    
    print(f"       -> Applied marker-based annotations to {len(adata_work)} cells")
    print(f"       -> Unique marker-based annotations: {adata_work.obs['marker_based_annotation'].nunique()}")
    
    # =========================================================================
    # Step 4: Generate COMPARABLE UMAP (same coordinates as filtered consistent cells)
    # =========================================================================
    print(f"\n    [Step 4] Generating COMPARABLE UMAPs...")
    
    # The adata_work already has X_umap from the filtered consistent cells
    # So we just need to plot with the new annotation
    
    if 'X_umap' not in adata_work.obsm:
        print("       -> Computing UMAP coordinates...")
        if 'X_pca' not in adata_work.obsm:
            sc.pp.pca(adata_work)
        sc.pp.neighbors(adata_work)
        sc.tl.umap(adata_work)
    else:
        print("       -> Using existing UMAP coordinates (same as filtered consistent cells)")
    
    # Generate side-by-side comparison UMAP
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        # Left: Original annotation (matches Br_consistent_cells_..._filtered_umap.png)
        sc.pl.umap(
            adata_work,
            color=annotation_col,
            ax=axes[0],
            show=False,
            palette=sc.pl.palettes.godsnot_102,
            legend_loc='right margin',
            legend_fontsize=7,
            title=f'Original Consensus Annotation\n(Same as filtered_umap.png, n={len(adata_work)})'
        )
        
        # Right: New marker-based annotation
        sc.pl.umap(
            adata_work,
            color='marker_based_annotation',
            ax=axes[1],
            show=False,
            palette=sc.pl.palettes.godsnot_102,
            legend_loc='right margin',
            legend_fontsize=7,
            title=f'Marker-Based Annotation\n({adata_work.obs["marker_based_annotation"].nunique()} types, n={len(adata_work)})'
        )
    
    plt.tight_layout()
    comparison_path = os.path.join(marker_anno_dir, f"{prefix}_filtered_consistent_annotation_comparison_umap.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['comparison_umap'] = comparison_path
    print(f"       -> Saved comparison UMAP: {comparison_path}")
    
    # Generate standalone marker-based annotation UMAP (directly comparable to filtered UMAP)
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(
            adata_work,
            color='marker_based_annotation',
            palette=sc.pl.palettes.godsnot_102,
            legend_loc='right margin',
            legend_fontsize=8,
            title=f'Filtered Consistent Cells - Marker-Based Annotation\n({adata_work.obs["marker_based_annotation"].nunique()} types, n={len(adata_work)})',
            show=False,
            size=10
        )
    
    marker_umap_path = os.path.join(marker_anno_dir, f"{prefix}_filtered_consistent_marker_based_umap.png")
    plt.savefig(marker_umap_path, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['marker_based_umap'] = marker_umap_path
    print(f"       -> Saved marker-based UMAP: {marker_umap_path}")
    
    # =========================================================================
    # Step 5: Generate annotation match/mismatch UMAP
    # =========================================================================
    print(f"\n    [Step 5] Generating annotation match/mismatch visualization...")
    
    # Create match column
    original_labels = adata_work.obs[annotation_col].astype(str)
    marker_labels = adata_work.obs['marker_based_annotation'].astype(str)
    
    # Check for matches (case-insensitive, partial match allowed)
    def labels_match(orig, marker):
        orig_lower = orig.lower().strip()
        marker_lower = marker.lower().strip()
        if orig_lower == marker_lower:
            return True
        if orig_lower in marker_lower or marker_lower in orig_lower:
            return True
        return False
    
    match_status = [
        'Match' if labels_match(o, m) else 'Mismatch'
        for o, m in zip(original_labels, marker_labels)
    ]
    adata_work.obs['annotation_match'] = pd.Categorical(match_status)
    
    n_match = (adata_work.obs['annotation_match'] == 'Match').sum()
    n_mismatch = (adata_work.obs['annotation_match'] == 'Mismatch').sum()
    
    print(f"       -> Matches: {n_match} cells ({n_match/len(adata_work)*100:.1f}%)")
    print(f"       -> Mismatches: {n_mismatch} cells ({n_mismatch/len(adata_work)*100:.1f}%)")
    
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(
            adata_work,
            color='annotation_match',
            palette={'Match': '#2ecc71', 'Mismatch': '#e74c3c'},
            legend_loc='right margin',
            legend_fontsize=10,
            title=f'Original vs Marker-Based Annotation Match\n(Match: {n_match}, Mismatch: {n_mismatch})',
            show=False,
            size=10
        )
    
    match_umap_path = os.path.join(marker_anno_dir, f"{prefix}_filtered_consistent_annotation_match_umap.png")
    plt.savefig(match_umap_path, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['match_umap'] = match_umap_path
    print(f"       -> Saved match/mismatch UMAP: {match_umap_path}")
    
    # =========================================================================
    # Step 6: Save CSVs
    # =========================================================================
    print(f"\n    [Step 6] Saving annotation results...")
    
    # Main annotation summary
    annotation_df = pd.DataFrame(annotation_records)
    annotation_df = annotation_df.sort_values('N_Cells', ascending=False)
    annotation_csv_path = os.path.join(marker_anno_dir, f"{prefix}_filtered_consistent_marker_annotation_summary.csv")
    annotation_df.to_csv(annotation_csv_path, index=False)
    output_paths['annotation_summary'] = annotation_csv_path
    print(f"       -> Saved: {annotation_csv_path}")
    
    # Detailed matches report
    if all_matches_records:
        matches_df = pd.DataFrame(all_matches_records)
        matches_csv_path = os.path.join(marker_anno_dir, f"{prefix}_filtered_consistent_marker_all_matches.csv")
        matches_df.to_csv(matches_csv_path, index=False)
        output_paths['all_matches'] = matches_csv_path
        print(f"       -> Saved: {matches_csv_path}")
    
    # Per-cell annotation CSV (for direct comparison)
    cell_annotations_df = pd.DataFrame({
        'cell_barcode': adata_work.obs_names,
        'original_annotation': adata_work.obs[annotation_col].values,
        'marker_based_annotation': adata_work.obs['marker_based_annotation'].values,
        'annotation_match': adata_work.obs['annotation_match'].values
    })
    cell_csv_path = os.path.join(marker_anno_dir, f"{prefix}_filtered_consistent_cell_annotations.csv")
    cell_annotations_df.to_csv(cell_csv_path, index=False)
    output_paths['cell_annotations'] = cell_csv_path
    print(f"       -> Saved: {cell_csv_path}")
    
    # =========================================================================
    # Step 7: Print summary
    # =========================================================================
    print(f"\n    {'='*60}")
    print(f"    MARKER-BASED ANNOTATION SUMMARY (Filtered Consistent Cells)")
    print(f"    {'='*60}")
    print(f"    Total cells: {len(adata_work)}")
    print(f"    Original cell types: {adata_work.obs[annotation_col].nunique()}")
    print(f"    Marker-based cell types: {adata_work.obs['marker_based_annotation'].nunique()}")
    print(f"    Annotation match rate: {n_match/len(adata_work)*100:.1f}%")
    print(f"    {'='*60}")
    
    print(f"\n    ✅ Marker-based annotation complete!")
    print(f"    Output files saved to: {marker_anno_dir}")
    
    return {
        'adata': adata_work,
        'annotation_df': annotation_df,
        'output_paths': output_paths
    }

def annotate_celltypes_by_marker_overlap(
    adata,
    prior_dict: dict,
    output_dir: str,
    prefix: str,
    groupby_key: str = 'leiden',
    n_top_genes: int = 50,
    min_overlap_score: float = 0.05,
    deg_ranking_method: str = 'original',
    deg_weight_fc: float = 0.4,
    deg_weight_expr: float = 0.3,
    deg_weight_pct: float = 0.3,
    species: str = "human",
    show_top_n_matches: int = 5,
    min_cells_per_cluster: int = 3
):
    """
    Automatically annotates cell types by matching top DEG markers against 
    an external marker gene database. Shows scores for all potential matches
    and adopts the top-scoring cell type.
    
    For each cluster/cell type:
    1. Extracts top N marker genes (DEGs)
    2. Compares against canonical markers from the prior database
    3. Calculates overlap scores for ALL matching cell types
    4. Assigns the highest-scoring cell type as the annotation
    5. Exports results to a new directory with UMAP visualization
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with clustering results.
    prior_dict : dict
        Marker prior dictionary from load_marker_prior_database().
    output_dir : str
        Base output directory (a subdirectory will be created).
    prefix : str
        File prefix for outputs.
    groupby_key : str
        Column in adata.obs containing cluster/cell type labels to annotate.
    n_top_genes : int
        Number of top DEG markers to consider per cluster.
    min_overlap_score : float
        Minimum overlap score to consider a match valid.
    deg_ranking_method : str
        'original' (logFC) or 'composite' for DEG ranking.
    deg_weight_fc, deg_weight_expr, deg_weight_pct : float
        Weights for composite DEG ranking.
    species : str
        Species for gene name standardization ('human' or 'mouse').
    show_top_n_matches : int
        Number of top alternative matches to show in output.
    min_cells_per_cluster : int
        Minimum cells required in a cluster for annotation.
    
    Returns
    -------
    dict
        Contains:
        - 'adata': AnnData with new 'marker_based_annotation' column
        - 'annotation_df': DataFrame with detailed annotation results
        - 'output_paths': dict of output file paths
    """
    import scanpy as sc
    
    print(f"\n{'='*70}")
    print(f"--- MARKER-BASED AUTOMATIC ANNOTATION ---")
    print(f"{'='*70}")
    print(f"    Groupby key: {groupby_key}")
    print(f"    Top N genes per cluster: {n_top_genes}")
    print(f"    DEG ranking method: {deg_ranking_method}")
    print(f"    Min overlap score: {min_overlap_score}")
    
    # Create output subdirectory
    marker_annot_dir = os.path.join(output_dir, "marker_based_annotation")
    os.makedirs(marker_annot_dir, exist_ok=True)
    print(f"    Output directory: {marker_annot_dir}")
    
    output_paths = {}
    
    # Validate inputs
    if groupby_key not in adata.obs.columns:
        print(f"[ERROR] Column '{groupby_key}' not found in adata.obs")
        return None
    
    if not prior_dict:
        print(f"[ERROR] Empty prior dictionary. Cannot perform marker-based annotation.")
        return None
    
    # Get cluster information
    cluster_counts = adata.obs[groupby_key].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_cells_per_cluster].index.tolist()
    
    print(f"    Total clusters: {len(cluster_counts)}")
    print(f"    Valid clusters (>= {min_cells_per_cluster} cells): {len(valid_clusters)}")
    
    if len(valid_clusters) < 2:
        print(f"[ERROR] Need at least 2 valid clusters for DEG analysis.")
        return None
    
    # =========================================================================
    # Step 1: Run DEG analysis
    # =========================================================================
    print(f"\n    [Step 1] Running DEG analysis...")
    
    deg_key = f'rank_genes_marker_annot_{groupby_key}'
    
    try:
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby_key,
            groups=valid_clusters,
            reference='rest',
            method='wilcoxon',
            pts=True,
            key_added=deg_key
        )
    except Exception as e:
        print(f"[ERROR] DEG analysis failed: {e}")
        return None
    
    # Extract DEG results
    try:
        marker_df = sc.get.rank_genes_groups_df(adata, group=None, key=deg_key)
    except Exception as e:
        print(f"[ERROR] Failed to extract DEG results: {e}")
        return None
    
    # Apply composite ranking if requested
    if deg_ranking_method == 'composite':
        print(f"       -> Applying composite ranking (FC:{deg_weight_fc}, Expr:{deg_weight_expr}, Pct:{deg_weight_pct})")
        marker_df = apply_composite_deg_ranking(
            marker_df=marker_df,
            deg_weight_fc=deg_weight_fc,
            deg_weight_expr=deg_weight_expr,
            deg_weight_pct=deg_weight_pct
        )
        rank_col = 'composite_score'
    else:
        rank_col = 'logfoldchanges'
    
    # =========================================================================
    # Step 2: Build gene name mapping for standardization
    # =========================================================================
    print(f"\n    [Step 2] Building gene name mappings...")
    
    gene_mapping = build_gene_name_mapping(adata, species)
    original_to_standard = gene_mapping['original_to_standard']
    
    # =========================================================================
    # Step 3: Calculate overlap scores for each cluster against all DB cell types
    # =========================================================================
    print(f"\n    [Step 3] Calculating overlap scores...")
    
    annotation_results = []
    
    for cluster_id in valid_clusters:
        cluster_id_str = str(cluster_id)
        n_cells = int(cluster_counts[cluster_id])
        
        # Get top DEGs for this cluster
        cluster_df = marker_df[marker_df['group'] == cluster_id].copy()
        
        if cluster_df.empty:
            annotation_results.append({
                'Cluster_ID': cluster_id_str,
                'N_Cells': n_cells,
                'Marker_Based_Annotation': 'Unknown',
                'Best_Match_Score': 0.0,
                'N_Overlapping_Genes': 0,
                'Overlapping_Genes': '',
                'Top_DEGs_Used': '',
                'Match_Method': 'no_degs'
            })
            continue
        
        # Sort by ranking column
        if rank_col in cluster_df.columns:
            cluster_df = cluster_df.sort_values(rank_col, ascending=False)
        
        # Get top N genes (original names)
        top_genes_original = cluster_df.head(n_top_genes)['names'].tolist()
        
        # Standardize gene names for matching
        top_genes_standard = {
            standardize_gene_name(g, species) for g in top_genes_original
        }
        
        # Calculate overlap scores against ALL cell types in the database
        all_scores = []
        
        for db_celltype, db_data in prior_dict.items():
            # Extract canonical markers (already standardized in prior_dict)
            if isinstance(db_data, dict):
                canonical_markers = db_data.get('markers_standard', set())
                original_name = db_data.get('original_name', db_celltype)
            elif isinstance(db_data, set):
                canonical_markers = db_data
                original_name = db_celltype
            else:
                continue
            
            if not canonical_markers:
                continue
            
            # Calculate overlap
            overlap_genes = top_genes_standard.intersection(canonical_markers)
            n_overlap = len(overlap_genes)
            
            if n_overlap == 0:
                continue
            
            # Calculate multiple scoring metrics
            # 1. Jaccard similarity
            union_size = len(top_genes_standard.union(canonical_markers))
            jaccard_score = n_overlap / union_size if union_size > 0 else 0
            
            # 2. Recall (proportion of canonical markers found)
            recall = n_overlap / len(canonical_markers)
            
            # 3. Precision (proportion of DEGs that are canonical markers)
            precision = n_overlap / len(top_genes_standard) if top_genes_standard else 0
            
            # 4. F1 score (harmonic mean of precision and recall)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Use F1 as the primary score (balances precision and recall)
            primary_score = f1_score
            
            all_scores.append({
                'db_celltype': db_celltype,
                'original_name': original_name,
                'score': primary_score,
                'jaccard': jaccard_score,
                'recall': recall,
                'precision': precision,
                'f1': f1_score,
                'n_overlap': n_overlap,
                'n_canonical': len(canonical_markers),
                'overlap_genes': overlap_genes
            })
        
        # Sort by score (descending)
        all_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Determine best annotation
        if all_scores and all_scores[0]['score'] >= min_overlap_score:
            best_match = all_scores[0]
            annotation = best_match['original_name']
            best_score = best_match['score']
            n_overlap = best_match['n_overlap']
            overlap_genes_list = sorted(list(best_match['overlap_genes']))
            match_method = 'marker_overlap'
        else:
            annotation = f'Unknown_{cluster_id_str}'
            best_score = 0.0
            n_overlap = 0
            overlap_genes_list = []
            match_method = 'no_significant_match'
        
        # Build result record
        result_record = {
            'Cluster_ID': cluster_id_str,
            'N_Cells': n_cells,
            'Marker_Based_Annotation': annotation,
            'Best_Match_Score': round(best_score, 4),
            'N_Overlapping_Genes': n_overlap,
            'Overlapping_Genes': '; '.join(overlap_genes_list[:20]),  # Limit to 20
            'Top_DEGs_Used': '; '.join(top_genes_original[:10]),  # Show first 10
            'Match_Method': match_method
        }
        
        # Add alternative matches
        for i, alt_match in enumerate(all_scores[1:show_top_n_matches], start=2):
            result_record[f'Alt_Match_{i}'] = alt_match['original_name']
            result_record[f'Alt_Match_{i}_Score'] = round(alt_match['score'], 4)
            result_record[f'Alt_Match_{i}_N_Overlap'] = alt_match['n_overlap']
        
        # Add detailed scoring breakdown for best match
        if all_scores:
            result_record['Best_Jaccard'] = round(all_scores[0]['jaccard'], 4)
            result_record['Best_Recall'] = round(all_scores[0]['recall'], 4)
            result_record['Best_Precision'] = round(all_scores[0]['precision'], 4)
            result_record['Best_F1'] = round(all_scores[0]['f1'], 4)
            result_record['N_Canonical_Markers'] = all_scores[0]['n_canonical']
        
        annotation_results.append(result_record)
    
    # =========================================================================
    # Step 4: Create annotation DataFrame and add to adata
    # =========================================================================
    print(f"\n    [Step 4] Creating annotation mapping...")
    
    annotation_df = pd.DataFrame(annotation_results)
    annotation_df = annotation_df.sort_values('N_Cells', ascending=False)
    
    # Create cluster -> annotation mapping
    cluster_to_annotation = dict(zip(
        annotation_df['Cluster_ID'].astype(str),
        annotation_df['Marker_Based_Annotation']
    ))
    
    # Add new annotation column to adata
    adata.obs['marker_based_annotation'] = adata.obs[groupby_key].astype(str).map(cluster_to_annotation)
    
    # Handle any unmapped clusters
    unmapped_mask = adata.obs['marker_based_annotation'].isna()
    if unmapped_mask.any():
        adata.obs.loc[unmapped_mask, 'marker_based_annotation'] = 'Unmapped'
    
    adata.obs['marker_based_annotation'] = adata.obs['marker_based_annotation'].astype('category')
    
    # =========================================================================
    # Step 5: Export results
    # =========================================================================
    print(f"\n    [Step 5] Exporting results...")
    
    # 5a. Summary CSV
    summary_path = os.path.join(marker_annot_dir, f"{prefix}_marker_based_annotation_summary.csv")
    annotation_df.to_csv(summary_path, index=False)
    output_paths['summary_csv'] = summary_path
    print(f"       -> Saved: {summary_path}")
    
    # 5b. Detailed per-cluster scores CSV (all potential matches)
    detailed_records = []
    for cluster_id in valid_clusters:
        cluster_id_str = str(cluster_id)
        n_cells = int(cluster_counts[cluster_id])
        
        cluster_df = marker_df[marker_df['group'] == cluster_id].copy()
        if cluster_df.empty:
            continue
        
        if rank_col in cluster_df.columns:
            cluster_df = cluster_df.sort_values(rank_col, ascending=False)
        
        top_genes_original = cluster_df.head(n_top_genes)['names'].tolist()
        top_genes_standard = {standardize_gene_name(g, species) for g in top_genes_original}
        
        for db_celltype, db_data in prior_dict.items():
            if isinstance(db_data, dict):
                canonical_markers = db_data.get('markers_standard', set())
                original_name = db_data.get('original_name', db_celltype)
            elif isinstance(db_data, set):
                canonical_markers = db_data
                original_name = db_celltype
            else:
                continue
            
            if not canonical_markers:
                continue
            
            overlap_genes = top_genes_standard.intersection(canonical_markers)
            n_overlap = len(overlap_genes)
            
            if n_overlap > 0:
                union_size = len(top_genes_standard.union(canonical_markers))
                jaccard = n_overlap / union_size if union_size > 0 else 0
                recall = n_overlap / len(canonical_markers)
                precision = n_overlap / len(top_genes_standard) if top_genes_standard else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                detailed_records.append({
                    'Cluster_ID': cluster_id_str,
                    'N_Cells': n_cells,
                    'DB_Cell_Type': original_name,
                    'N_Overlap': n_overlap,
                    'N_Canonical_Markers': len(canonical_markers),
                    'Jaccard_Score': round(jaccard, 4),
                    'Recall': round(recall, 4),
                    'Precision': round(precision, 4),
                    'F1_Score': round(f1, 4),
                    'Overlapping_Genes': '; '.join(sorted(overlap_genes))
                })
    
    if detailed_records:
        detailed_df = pd.DataFrame(detailed_records)
        detailed_df = detailed_df.sort_values(['Cluster_ID', 'F1_Score'], ascending=[True, False])
        detailed_path = os.path.join(marker_annot_dir, f"{prefix}_marker_based_annotation_all_scores.csv")
        detailed_df.to_csv(detailed_path, index=False)
        output_paths['detailed_scores_csv'] = detailed_path
        print(f"       -> Saved: {detailed_path}")
    
    # =========================================================================
    # Step 6: Generate UMAP visualizations
    # =========================================================================
    print(f"\n    [Step 6] Generating UMAP visualizations...")
    
    # 6a. UMAP with marker-based annotation
    try:
        with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
            sc.pl.umap(
                adata,
                color='marker_based_annotation',
                palette=sc.pl.palettes.godsnot_102,
                legend_loc='right margin',
                legend_fontsize=8,
                title=f'Marker-Based Annotation\n({adata.obs["marker_based_annotation"].nunique()} types)',
                show=False,
                size=10
            )
        
        umap_path = os.path.join(marker_annot_dir, f"{prefix}_marker_based_annotation_umap.png")
        _bold_right_margin_legend(umap_path)
        plt.close()
        output_paths['umap'] = umap_path
        print(f"       -> Saved: {umap_path}")
    except Exception as e:
        print(f"       [WARNING] Could not generate marker-based annotation UMAP: {e}")
    
    # 6b. Comparison UMAP (original vs marker-based) if original annotation exists
    original_annot_col = 'ctpt_consensus_prediction'
    if original_annot_col in adata.obs.columns:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(24, 10))
            
            with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
                sc.pl.umap(
                    adata,
                    color=original_annot_col,
                    palette=sc.pl.palettes.godsnot_102,
                    ax=axes[0],
                    show=False,
                    size=8,
                    legend_loc=None,
                    title=f'Original Annotation ({original_annot_col})\n({adata.obs[original_annot_col].nunique()} types)'
                )
                
                sc.pl.umap(
                    adata,
                    color='marker_based_annotation',
                    palette=sc.pl.palettes.godsnot_102,
                    ax=axes[1],
                    show=False,
                    size=8,
                    legend_loc=None,
                    title=f'Marker-Based Annotation\n({adata.obs["marker_based_annotation"].nunique()} types)'
                )
            
            plt.tight_layout()
            comparison_path = os.path.join(marker_annot_dir, f"{prefix}_marker_based_annotation_comparison.png")
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            output_paths['comparison_umap'] = comparison_path
            print(f"       -> Saved: {comparison_path}")
        except Exception as e:
            print(f"       [WARNING] Could not generate comparison UMAP: {e}")
    
    # =========================================================================
    # Step 7: Print summary statistics
    # =========================================================================
    print(f"\n    [Summary]")
    print(f"    " + "-"*50)
    
    n_annotated = (annotation_df['Match_Method'] == 'marker_overlap').sum()
    n_unknown = (annotation_df['Match_Method'] != 'marker_overlap').sum()
    
    print(f"    Clusters with marker-based annotation: {n_annotated}/{len(annotation_df)}")
    print(f"    Clusters marked as Unknown: {n_unknown}/{len(annotation_df)}")
    
    if n_annotated > 0:
        avg_score = annotation_df[annotation_df['Best_Match_Score'] > 0]['Best_Match_Score'].mean()
        print(f"    Average best match score: {avg_score:.4f}")
    
    print(f"\n    Annotation distribution:")
    for annot, count in adata.obs['marker_based_annotation'].value_counts().head(10).items():
        pct = count / len(adata) * 100
        print(f"       {annot}: {count} cells ({pct:.1f}%)")
    
    if adata.obs['marker_based_annotation'].nunique() > 10:
        print(f"       ... and {adata.obs['marker_based_annotation'].nunique() - 10} more types")
    
    print(f"\n    ✅ Marker-based annotation complete!")
    print(f"    Output directory: {marker_annot_dir}")
    
    # =========================================================================
    # NEW: Generate Comparison UMAP - Original vs Marker-Based Annotation
    # =========================================================================
    print(f"\n    [Step X] Generating comparison UMAP: Original vs Marker-Based Annotation...")
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        
        with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
            # Left panel: Original annotation (groupby_key, e.g., leiden or consensus)
            sc.pl.umap(
                adata, 
                color=groupby_key,
                palette=sc.pl.palettes.godsnot_102,
                ax=axes[0], 
                show=False, 
                size=10,
                legend_loc='right margin',
                legend_fontsize=8,
                title=f'Original ({groupby_key})\n({adata.obs[groupby_key].nunique()} clusters)',
                frameon=False
            )
            
            # Right panel: Marker-based annotation
            sc.pl.umap(
                adata, 
                color='marker_based_annotation',
                palette=sc.pl.palettes.godsnot_102,
                ax=axes[1], 
                show=False, 
                size=10,
                legend_loc='right margin',
                legend_fontsize=8,
                title=f'Marker-Based Annotation\n({adata.obs["marker_based_annotation"].nunique()} types)',
                frameon=False
            )
        
        plt.tight_layout()
        
        comparison_umap_path = os.path.join(marker_annot_dir, f"{prefix}_marker_based_annotation_comparison.png")
        plt.savefig(comparison_umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        output_paths['comparison_umap'] = comparison_umap_path
        print(f"       -> Saved comparison UMAP: {comparison_umap_path}")
        
    except Exception as e:
        print(f"[WARNING] Could not generate comparison UMAP. Reason: {e}")
        import traceback
        traceback.print_exc()

    return {
        'adata': adata,
        'annotation_df': annotation_df,
        'output_paths': output_paths
    }


def run_marker_based_reannotation_for_low_confidence(
    adata,
    prior_dict: dict,
    output_dir: str,
    prefix: str,
    cas_csv_path: str,
    cas_threshold: float,
    cas_aggregation_method: str = 'leiden',
    n_top_genes: int = 50,
    deg_ranking_method: str = 'original',
    deg_weight_fc: float = 0.4,
    deg_weight_expr: float = 0.3,
    deg_weight_pct: float = 0.3,
    species: str = "human"
):
    """
    Re-annotates low-confidence clusters using marker-based overlap scoring.
    
    This function:
    1. Identifies clusters below the CAS threshold
    2. Runs marker-based annotation on those clusters
    3. Exports results to a new directory
    4. Generates comparison UMAPs
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with clustering and CAS scores.
    prior_dict : dict
        Marker prior dictionary.
    output_dir : str
        Base output directory.
    prefix : str
        File prefix.
    cas_csv_path : str
        Path to CAS scores CSV.
    cas_threshold : float
        CAS threshold for identifying low-confidence clusters.
    cas_aggregation_method : str
        'leiden' or 'consensus' - how CAS was aggregated.
    n_top_genes : int
        Number of top DEGs to consider.
    deg_ranking_method : str
        'original' or 'composite'.
    deg_weight_fc, deg_weight_expr, deg_weight_pct : float
        Weights for composite ranking.
    species : str
        'human' or 'mouse'.
    
    Returns
    -------
    dict
        Results including updated adata and output paths.
    """
    print(f"\n{'='*70}")
    print(f"--- RE-ANNOTATION OF LOW-CONFIDENCE CLUSTERS ---")
    print(f"{'='*70}")
    
    # Create output directory
    reannot_dir = os.path.join(output_dir, "low_confidence_reannotation")
    os.makedirs(reannot_dir, exist_ok=True)
    
    # Load CAS scores
    if not os.path.exists(cas_csv_path):
        print(f"[ERROR] CAS file not found: {cas_csv_path}")
        return None
    
    cas_df = pd.read_csv(cas_csv_path)
    
    # Identify low-confidence clusters
    if cas_aggregation_method == 'leiden':
        id_col = 'Cluster_ID (Leiden)'
        failing_ids = cas_df[cas_df['Cluster_Annotation_Score_CAS (%)'] < cas_threshold][id_col].astype(str).tolist()
        groupby_key = 'leiden'
    else:
        id_col = 'Consensus_Cell_Type'
        failing_ids = cas_df[cas_df['Cluster_Annotation_Score_CAS (%)'] < cas_threshold][id_col].tolist()
        groupby_key = 'ctpt_consensus_prediction'
    
    print(f"    CAS threshold: {cas_threshold}%")
    print(f"    Low-confidence clusters: {len(failing_ids)}")
    
    if not failing_ids:
        print("    No low-confidence clusters found. Skipping re-annotation.")
        return None
    
    print(f"    Clusters to re-annotate: {failing_ids}")
    
    # Subset to low-confidence cells
    if groupby_key == 'leiden':
        mask = adata.obs['leiden'].astype(str).isin(failing_ids)
    else:
        mask = adata.obs['ctpt_consensus_prediction'].isin(failing_ids)
    
    n_low_conf_cells = mask.sum()
    print(f"    Cells in low-confidence clusters: {n_low_conf_cells}")
    
    if n_low_conf_cells < 10:
        print("    Too few cells for re-annotation. Skipping.")
        return None
    
    # Create a copy for annotation
    adata_subset = adata[mask].copy()
    
    # Run marker-based annotation on the subset
    result = annotate_celltypes_by_marker_overlap(
        adata=adata_subset,
        prior_dict=prior_dict,
        output_dir=reannot_dir,
        prefix=f"{prefix}_low_confidence",
        groupby_key=groupby_key,
        n_top_genes=n_top_genes,
        deg_ranking_method=deg_ranking_method,
        deg_weight_fc=deg_weight_fc,
        deg_weight_expr=deg_weight_expr,
        deg_weight_pct=deg_weight_pct,
        species=species
    )
    
    if result is None:
        return None
    
    # Update the main adata with new annotations for low-confidence cells
    adata.obs['marker_reannotation'] = adata.obs['ctpt_consensus_prediction'].astype(str)
    
    # Map new annotations back
    new_annotations = result['adata'].obs['marker_based_annotation']
    adata.obs.loc[new_annotations.index, 'marker_reannotation'] = new_annotations.values
    
    adata.obs['marker_reannotation'] = adata.obs['marker_reannotation'].astype('category')
    
    # Generate full UMAP with re-annotated cells
    print(f"\n    Generating full UMAP with re-annotations...")
    
    try:
        # Create visualization showing which cells were re-annotated
        viz_col = '_reannotation_status'
        adata.obs[viz_col] = 'Original'
        adata.obs.loc[mask, viz_col] = 'Re-annotated'
        adata.obs[viz_col] = adata.obs[viz_col].astype('category')
        
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))
        
        with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
            # Original annotation
            sc.pl.umap(adata, color='ctpt_consensus_prediction', ax=axes[0], show=False,
                      palette=sc.pl.palettes.godsnot_102, legend_loc=None, size=8,
                      title='Original Consensus Annotation')
            
            # Re-annotation status
            sc.pl.umap(adata, color=viz_col, ax=axes[1], show=False,
                      palette={'Original': '#cccccc', 'Re-annotated': '#e74c3c'},
                      legend_loc='right margin', size=8,
                      title=f'Re-annotation Status\n({n_low_conf_cells} cells re-annotated)')
            
            # Final annotation after re-annotation
            sc.pl.umap(adata, color='marker_reannotation', ax=axes[2], show=False,
                      palette=sc.pl.palettes.godsnot_102, legend_loc=None, size=8,
                      title='After Marker-Based Re-annotation')
        
        plt.tight_layout()
        reannot_umap_path = os.path.join(reannot_dir, f"{prefix}_reannotation_overview.png")
        plt.savefig(reannot_umap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"       -> Saved: {reannot_umap_path}")
        
        # Cleanup
        del adata.obs[viz_col]
        
    except Exception as e:
        print(f"    [WARNING] Could not generate re-annotation UMAP: {e}")
    
    # Export final annotations CSV
    final_csv_path = os.path.join(reannot_dir, f"{prefix}_final_annotations_with_reannotation.csv")
    export_cols = ['leiden', 'ctpt_individual_prediction', 'ctpt_consensus_prediction', 'marker_reannotation']
    export_cols = [c for c in export_cols if c in adata.obs.columns]
    adata.obs[export_cols].to_csv(final_csv_path)
    print(f"       -> Saved: {final_csv_path}")
    
    print(f"\n    ✅ Low-confidence re-annotation complete!")
    
    return {
        'adata': adata,
        'n_reannotated_cells': n_low_conf_cells,
        'output_dir': reannot_dir
    }

def calculate_marker_prior_score(
    adata,
    groupby_key: str,
    prior_dict: dict,
    n_top_genes: int = 100,
    n_background_genes: int = 200,
    deg_ranking_method: str = "original",
    deg_weight_fc: float = 0.4,
    deg_weight_expr: float = 0.3,
    deg_weight_pct: float = 0.3,
    similarity_threshold: float = 0.6,
    verbose_matching: bool = False,
    min_cells_per_group: int = 3
) -> tuple:
    """
    Calculates marker prior score with STRICT cell type matching.
    
    NOW USES: find_matching_cell_type_in_prior() with strict word-boundary rules.
    """
    import scanpy as sc
    
    print("    -> [MPS] Calculating Marker Prior Score with STRICT matching...")
    
    # Validate inputs
    if groupby_key not in adata.obs.columns:
        print(f"    -> [MPS] Warning: '{groupby_key}' not in adata.obs, returning 0")
        return (0.0, {})
    
    if not prior_dict:
        print("    -> [MPS] Warning: Empty prior_dict, returning 0")
        return (0.0, {})
    
    # Get cluster counts and FILTER to only include clusters with cells > 0
    cluster_counts_raw = adata.obs[groupby_key].value_counts()
    cluster_counts = cluster_counts_raw[cluster_counts_raw > 0]
    all_cluster_ids = cluster_counts.index.tolist()
    
    n_empty_categories = len(cluster_counts_raw) - len(cluster_counts)
    if n_empty_categories > 0:
        print(f"    -> [MPS] Note: Excluded {n_empty_categories} empty cell type categories")
    
    # Filter clusters by minimum cell count
    valid_clusters = cluster_counts[cluster_counts >= min_cells_per_group].index.tolist()
    small_clusters = cluster_counts[cluster_counts < min_cells_per_group].index.tolist()
    
    print(f"    -> [MPS] Found {len(all_cluster_ids)} consensus cell types with cells in data")
    print(f"    -> [MPS] {len(valid_clusters)} cell types have >= {min_cells_per_group} cells")
    
    if len(valid_clusters) < 2:
        print("    -> [MPS] Warning: Need at least 2 valid cell types for DEG analysis. Returning 0.")
        return (0.0, {})
    
    print(f"    -> [MPS] Marker database contains {len(prior_dict)} cell types")
    
    # Run DEG only on valid clusters
    try:
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby_key,
            groups=valid_clusters,
            reference='rest',
            method='wilcoxon',
            pts=True,
            key_added='rank_genes_mps'
        )
    except Exception as e:
        print(f"    -> [MPS] Warning: DEG analysis failed: {e}")
        return (0.0, {})
    
    # Extract DEG results
    try:
        marker_df = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_mps')
    except Exception as e:
        print(f"    -> [MPS] Warning: Failed to extract DEG results: {e}")
        return (0.0, {})
    
    # Apply composite ranking if requested
    if deg_ranking_method == 'composite':
        print("       -> [MPS] Applying composite DEG ranking...")
        marker_df = apply_composite_deg_ranking(
            marker_df=marker_df,
            deg_weight_fc=deg_weight_fc,
            deg_weight_expr=deg_weight_expr,
            deg_weight_pct=deg_weight_pct
        )
        rank_column = 'composite_score'
    else:
        rank_column = 'logfoldchanges'
    
    # Calculate MPS for each cluster
    mps_details = {}
    matching_summary = {
        'exact': 0,
        'canonical_synonym': 0,
        'token_jaccard': 0,
        'whole_phrase_match': 0,
        'fuzzy_match': 0,
        'no_match': 0,
        'too_few_cells': 0
    }
    
    valid_cluster_matching = []
    
    # =========================================================================
    # Process each cluster
    # =========================================================================
    for cluster_id in all_cluster_ids:
        cluster_id_str = str(cluster_id)
        
        # =====================================================================
        # EXPAND ABBREVIATED CELL TYPE NAME BEFORE MATCHING
        # =====================================================================
        try:
            cluster_expanded, was_expanded, cluster_original = expand_celltype_with_context(
                cluster_id_str, verbose=verbose_matching
            )
        except Exception as e:
            # Fallback if expansion function fails
            cluster_expanded = cluster_id_str
            was_expanded = False
            cluster_original = cluster_id_str
            if verbose_matching:
                print(f"       [MPS] Warning: Abbreviation expansion failed for '{cluster_id_str}': {e}")
        
        if was_expanded and verbose_matching:
            print(f"       [MPS] Expanded '{cluster_original}' -> '{cluster_expanded}'")
        
        # Use expanded form for marker database lookup
        cluster_for_matching = cluster_expanded
        # =====================================================================
        
        # Handle small clusters separately
        if cluster_id not in valid_clusters:
            matching_summary['too_few_cells'] += 1
            mps_details[cluster_id] = {
                'mps': 0.0,
                'matched_db_type': None,
                'match_method': 'skipped_too_few_cells',
                'match_confidence': 0.0,
                'n_canonical_markers': 0,
                'n_markers_found': 0,
                'n_top_genes_used': 0,
                'recall': 0.0,
                'precision': 0.0,
                'f1_score': 0.0,
                'n_cells': int(cluster_counts[cluster_id]),
                'top_genes': [],
                'matched_markers': [],
                'skip_reason': f'Only {cluster_counts[cluster_id]} cells (need >= {min_cells_per_group})',
                'original_name': cluster_original,
                'expanded_name': cluster_expanded,
                'was_expanded': was_expanded
            }
            continue
        
        # =====================================================================
        # Use EXPANDED name for matching
        # =====================================================================
        matched_db_type, match_method, confidence, canonical_markers = find_matching_cell_type_in_prior(
            query_cell_type=cluster_for_matching,
            prior_dict=prior_dict,
            min_similarity=similarity_threshold,
            min_token_length=4,
            verbose=verbose_matching
        )
        
        # Track matching statistics
        if match_method and match_method != 'no_match':
            matching_summary[match_method] = matching_summary.get(match_method, 0) + 1
        else:
            matching_summary['no_match'] += 1
        
        # Store matching info
        valid_cluster_matching.append({
            'cluster': cluster_id_str,
            'cluster_original': cluster_original,
            'cluster_expanded': cluster_expanded,
            'was_expanded': was_expanded,
            'n_cells': int(cluster_counts[cluster_id]),
            'matched_to': matched_db_type,
            'method': match_method,
            'confidence': confidence,
            'n_markers': len(canonical_markers) if canonical_markers else 0
        })
        
        # Skip if no match found
        if not canonical_markers:
            mps_details[cluster_id] = {
                'mps': 0.0,
                'matched_db_type': None,
                'match_method': match_method,
                'match_confidence': 0.0,
                'n_canonical_markers': 0,
                'n_markers_found': 0,
                'n_top_genes_used': 0,
                'recall': 0.0,
                'precision': 0.0,
                'f1_score': 0.0,
                'n_cells': int(cluster_counts[cluster_id]),
                'top_genes': [],
                'matched_markers': [],
                'original_name': cluster_original,
                'expanded_name': cluster_expanded,
                'was_expanded': was_expanded
            }
            continue
        
        # Get top genes for this cluster
        cluster_df = marker_df[marker_df['group'] == cluster_id].copy()
        
        if cluster_df.empty:
            mps_details[cluster_id] = {
                'mps': 0.0,
                'matched_db_type': matched_db_type,
                'match_method': match_method,
                'match_confidence': confidence,
                'n_canonical_markers': len(canonical_markers),
                'n_markers_found': 0,
                'n_top_genes_used': 0,
                'recall': 0.0,
                'precision': 0.0,
                'f1_score': 0.0,
                'n_cells': int(cluster_counts[cluster_id]),
                'top_genes': [],
                'matched_markers': [],
                'original_name': cluster_original,
                'expanded_name': cluster_expanded,
                'was_expanded': was_expanded
            }
            continue
        
        # Sort by ranking column
        if rank_column in cluster_df.columns:
            cluster_df = cluster_df.sort_values(rank_column, ascending=False)
        else:
            cluster_df = cluster_df.sort_values('logfoldchanges', ascending=False)
        
        # Get top N genes
        top_genes = cluster_df.head(n_top_genes)['names'].tolist()
        top_genes_set = set(top_genes)
        
        # Ensure canonical_markers is a set
        if not isinstance(canonical_markers, set):
            canonical_markers = set(canonical_markers) if canonical_markers else set()
        
        # =====================================================================
        # F1 SCORE CALCULATION (with case-insensitive matching)
        # =====================================================================
        
        # Normalize gene names for comparison (uppercase)
        top_genes_upper = {g.upper() for g in top_genes_set if isinstance(g, str)}
        canonical_upper = {g.upper() for g in canonical_markers if isinstance(g, str)}
        
        # Find overlapping markers (case-insensitive)
        matched_markers_upper = canonical_upper.intersection(top_genes_upper)
        n_overlap = len(matched_markers_upper)
        
        # For reporting, find the original gene names that matched
        matched_markers = {g for g in top_genes_set if isinstance(g, str) and g.upper() in matched_markers_upper}
        
        n_canonical = len(canonical_markers)
        n_top_genes_actual = len(top_genes_set)
        
        # Debug output for first few clusters
        if verbose_matching or len(mps_details) < 3:
            print(f"       [DEBUG] Cluster '{cluster_id}': {n_overlap} overlapping markers")
            print(f"                Top 5 DEGs: {list(top_genes_set)[:5]}")
            print(f"                Top 5 DB markers: {list(canonical_markers)[:5]}")
            if matched_markers:
                print(f"                Matched: {list(matched_markers)[:10]}")
        
        # Recall: Fraction of canonical markers found in top DEGs
        recall = n_overlap / n_canonical if n_canonical > 0 else 0.0
        
        # Precision: Fraction of top DEGs that are canonical markers
        precision = n_overlap / n_top_genes_actual if n_top_genes_actual > 0 else 0.0
        
        # F1 Score: Harmonic mean (balances precision and recall)
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # Final weighted MPS using F1 instead of recall
        weighted_mps = f1_score * confidence
        
        # =====================================================================
        # Store results
        # =====================================================================
        mps_details[cluster_id] = {
            # Primary score
            'mps': weighted_mps,
            
            # All component scores
            'raw_f1': f1_score,
            'raw_recall': recall,
            'raw_precision': precision,
            
            # Matching info
            'matched_db_type': matched_db_type,
            'match_method': match_method,
            'match_confidence': confidence,
            
            # Expansion tracking
            'original_name': cluster_original,
            'expanded_name': cluster_expanded,
            'was_expanded': was_expanded,
            'query_used_for_matching': cluster_for_matching,

            # Statistics
            'n_canonical_markers': n_canonical,
            'n_markers_found': n_overlap,
            'n_top_genes_used': n_top_genes_actual,
            
            # Legacy (backward compatibility)
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            
            # Cluster info
            'n_cells': int(cluster_counts[cluster_id]),
            
            # Details
            'top_genes': top_genes[:20],
            'matched_markers': list(matched_markers)
        }
    
    # =========================================================================
    # Print detailed matching results
    # =========================================================================
    print("\n    -> [MPS] Cell Type Matching Results (Consensus → Marker DB):")
    print("       " + "-" * 100)
    print(f"       {'Consensus Cell Type':<45} {'Cells':>6} {'Matched DB Entry':<35} {'Method':<20} {'Markers':>8}")
    print("       " + "-" * 100)
    
    for info in valid_cluster_matching:
        matched_str = info['matched_to'] if info['matched_to'] else "NO MATCH"
        method_str = info['method'] if info['method'] else "N/A"
        markers_str = str(info['n_markers']) if info['n_markers'] > 0 else "0"
        
        # Show expansion if it occurred
        if info.get('was_expanded', False):
            original = info.get('cluster_original', info['cluster'])
            expanded = info.get('cluster_expanded', info['cluster'])
            cluster_display = f"{original} → {expanded}"
            if len(cluster_display) > 45:
                cluster_display = cluster_display[:43] + '..'
        else:
            cluster_display = info['cluster'][:43] + '..' if len(info['cluster']) > 45 else info['cluster']
        
        matched_display = matched_str[:33] + '..' if len(matched_str) > 35 else matched_str
        
        status_icon = "✓" if info['matched_to'] else "✗"
        print(f"       {status_icon} {cluster_display:<43} {info['n_cells']:>6} {matched_display:<35} {method_str:<20} {markers_str:>8}")
    
    print("       " + "-" * 100)
    
    # Print matching summary
    print("\n    -> [MPS] Matching Summary:")
    for method, count in matching_summary.items():
        if count > 0:
            print(f"       - {method}: {count} cell types")
    
    # =========================================================================
    # Calculate mean MPS
    # =========================================================================
    print("\n    -> [MPS DEBUG] Individual cluster MPS values:")
    for cluster_id, details in mps_details.items():
        method = details.get('match_method', 'None')
        mps_val = details.get('mps', 0.0)
        matched = details.get('matched_db_type', 'None')
        n_markers = details.get('n_canonical_markers', 0)
        n_found = details.get('n_markers_found', 0)
        f1 = details.get('f1_score', 0.0)
        
        # Check if this would be included
        passes_filter = (method is not None and 
                        method not in ['skipped_too_few_cells', 'no_match'])
        
        status = "✓ INCLUDED" if passes_filter else "✗ EXCLUDED"
        print(f"       {status} | {str(cluster_id)[:30]:<30} | method={method:<20} | "
              f"mps={mps_val:.4f} | F1={f1:.4f} | markers={n_found}/{n_markers}")
    
    # Calculate valid MPS scores
    valid_mps = []
    for cluster_id, v in mps_details.items():
        method = v.get('match_method')
        if method is not None and method not in ['skipped_too_few_cells', 'no_match']:
            mps_val = v.get('mps', 0.0)
            valid_mps.append(mps_val)
            print(f"    -> [MPS DEBUG] Adding cluster '{cluster_id}' with MPS={mps_val:.4f}")
    
    # Calculate mean
    if valid_mps:
        mean_mps = np.mean(valid_mps) * 100
        print(f"    -> [MPS DEBUG] valid_mps list: {valid_mps}")
        print(f"    -> [MPS DEBUG] np.mean(valid_mps) = {np.mean(valid_mps):.4f}")
        print(f"    -> [MPS DEBUG] mean_mps (×100) = {mean_mps:.2f}%")
    else:
        mean_mps = 0.0
        print(f"    -> [MPS DEBUG] No valid MPS scores found! valid_mps is empty.")
    
    n_evaluated = len(valid_mps)
    n_total = len(all_cluster_ids)
    n_matched = sum(1 for v in mps_details.values() if v.get('matched_db_type') is not None)
    
    print(f"\n    -> [MPS] Mean MPS: {mean_mps:.2f}% ({n_evaluated}/{n_total} cell types evaluated, {n_matched} matched to DB)")
    
    return (mean_mps, mps_details)

# --- Conditional Import for Harmony ---
try:
    import harmonypy as hm
except ImportError:
    print("Warning: harmonypy is not installed. Multi-sample integration mode will not be available.")
    print("Please run 'pip install harmonypy'")

# --- Helper Function for Flexible Data Loading ---
def load_expression_data(path, var_names='gene_symbols'):
    """
    Loads expression data from either 10x Genomics directory (containing matrix.mtx, 
    genes.tsv/features.tsv, barcodes.tsv) or from an .h5/.h5ad file.
    
    Args:
        path (str): Path to either a directory (10x format) or an .h5/.h5ad file.
        var_names (str): For 10x format, which column to use for var_names.
        
    Returns:
        anndata.AnnData: Loaded AnnData object.
    """
    path = str(path).strip()
    
    if os.path.isfile(path):
        # It's a file - check extension
        ext = os.path.splitext(path)[1].lower()
        if ext == '.h5ad':
            print(f"       -> Loading AnnData from .h5ad file: {path}")
            adata = sc.read_h5ad(path)
        elif ext in ['.h5', '.hdf5']:
            print(f"       -> Loading from 10x .h5 file: {path}")
            adata = sc.read_10x_h5(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Expected .h5, .h5ad, or a 10x directory.")
    elif os.path.isdir(path):
        # It's a directory - assume 10x format
        print(f"       -> Loading from 10x directory: {path}")
        adata = sc.read_10x_mtx(path, var_names=var_names, cache=True)
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    adata.var_names_make_unique()
    return adata

def detect_batch_from_barcodes(adata):
    """
    Detect batch/sample information from barcode suffixes.
    
    Handles patterns like:
    - AAACCCAAGACTCTTG-1_Br6522mid (10x format with sample suffix)
    - AAACCCAAGACTCTTG_Br6522mid (simple underscore separator)
    - AAACCCAAGACTCTTG-Br6522mid (hyphen separator)
    
    Returns:
        adata: AnnData object with 'sample' column added if pattern detected
        bool: True if batch pattern was detected, False otherwise
    """
    import re
    
    barcodes = adata.obs_names.tolist()
    
    if not barcodes:
        print("       -> [Batch Detection] No barcodes found")
        return adata, False
    
    print(f"       -> [Batch Detection] Analyzing {len(barcodes)} barcodes...")
    print(f"       -> [Batch Detection] Example barcodes: {barcodes[:3]}")
    
    batch_labels = []
    pattern_found = None
    
    # Pattern 1: BARCODE-N_SAMPLENAME (e.g., AAACCCAAGACTCTTG-1_Br6522mid)
    # This is the most common 10x multi-sample format
    pattern1_regex = re.compile(r'^([ACGTN]+-\d+)_(.+)$')
    
    # Pattern 2: BARCODE_SAMPLENAME (e.g., AAACCCAAGACTCTTG_Br6522mid)
    pattern2_regex = re.compile(r'^([ACGTN]+)_(.+)$')
    
    # Pattern 3: BARCODE-SAMPLENAME (e.g., AAACCCAAGACTCTTG-Br6522mid)
    pattern3_regex = re.compile(r'^([ACGTN]+)-([A-Za-z].*)$')
    
    # Try Pattern 1 first (your specific format: BARCODE-1_SampleName)
    matches = []
    for bc in barcodes:
        match = pattern1_regex.match(bc)
        if match:
            matches.append(match.group(2))  # Extract sample name after underscore
        else:
            matches.append(None)
    
    success_rate = sum(1 for m in matches if m is not None) / len(barcodes)
    
    if success_rate >= 0.95:  # Allow 5% mismatch for edge cases
        batch_labels = [m if m is not None else 'Unknown' for m in matches]
        pattern_found = "BARCODE-N_SAMPLE"
        print(f"       -> [Batch Detection] Pattern detected: {pattern_found} (success rate: {success_rate:.1%})")
    else:
        # Try Pattern 2: BARCODE_SAMPLENAME
        matches = []
        for bc in barcodes:
            match = pattern2_regex.match(bc)
            if match:
                matches.append(match.group(2))
            else:
                matches.append(None)
        
        success_rate = sum(1 for m in matches if m is not None) / len(barcodes)
        
        if success_rate >= 0.95:
            batch_labels = [m if m is not None else 'Unknown' for m in matches]
            pattern_found = "BARCODE_SAMPLE"
            print(f"       -> [Batch Detection] Pattern detected: {pattern_found} (success rate: {success_rate:.1%})")
        else:
            # Try Pattern 3: BARCODE-SAMPLENAME
            matches = []
            for bc in barcodes:
                match = pattern3_regex.match(bc)
                if match:
                    matches.append(match.group(2))
                else:
                    matches.append(None)
            
            success_rate = sum(1 for m in matches if m is not None) / len(barcodes)
            
            if success_rate >= 0.95:
                batch_labels = [m if m is not None else 'Unknown' for m in matches]
                pattern_found = "BARCODE-SAMPLE"
                print(f"       -> [Batch Detection] Pattern detected: {pattern_found} (success rate: {success_rate:.1%})")
            else:
                # Final attempt: generic split by last underscore
                try:
                    potential_batches = []
                    for bc in barcodes:
                        if '_' in bc:
                            potential_batches.append(bc.rsplit('_', 1)[-1])
                        else:
                            potential_batches.append(None)
                    
                    unique_batches = set(b for b in potential_batches if b is not None)
                    success_rate = sum(1 for b in potential_batches if b is not None) / len(barcodes)
                    
                    # Accept if: reasonable number of batches (2-50) and high success rate
                    if len(unique_batches) >= 2 and len(unique_batches) <= 50 and success_rate >= 0.95:
                        batch_labels = [b if b is not None else 'Unknown' for b in potential_batches]
                        pattern_found = "GENERIC_UNDERSCORE"
                        print(f"       -> [Batch Detection] Pattern detected: {pattern_found} (success rate: {success_rate:.1%})")
                except Exception as e:
                    print(f"       -> [Batch Detection] Generic pattern detection failed: {e}")
    
    # Apply detected batches
    if batch_labels and pattern_found:
        unique_batches = sorted(set(batch_labels))
        print(f"       -> [Batch Detection] Found {len(unique_batches)} batches/samples:")
        
        # Count cells per batch
        from collections import Counter
        batch_counts = Counter(batch_labels)
        for batch_name in unique_batches:
            print(f"          - {batch_name}: {batch_counts[batch_name]} cells")
        
        # Add to adata.obs
        adata.obs['sample'] = batch_labels
        adata.obs['sample'] = adata.obs['sample'].astype('category')
        
        # Also create 'batch' column as alias
        adata.obs['batch'] = adata.obs['sample'].copy()
        
        return adata, True
    
    print(f"       -> [Batch Detection] No batch pattern detected in barcodes")
    print(f"       -> [Batch Detection] Will treat all cells as single sample")
    
    # Create dummy sample column
    adata.obs['sample'] = 'sample_1'
    adata.obs['sample'] = adata.obs['sample'].astype('category')
    
    return adata, False

def _add_protected_markers(current_hvgs: list, prior_dict: dict, adata, 
                           max_additions: int, species: str = "human") -> list:
    """
    Adds canonical markers from prior database to HVG list if not already present.
    
    Args:
        current_hvgs: Current list of HVG gene names
        prior_dict: Marker prior dictionary
        adata: AnnData object
        max_additions: Maximum number of markers to add
        species: Species for gene standardization
    
    Returns:
        Updated HVG list with protected markers added
    """
    if not prior_dict:
        return current_hvgs
    
    # Build gene mapping
    gene_mapping = build_gene_name_mapping(adata, species)
    standard_to_original = gene_mapping['standard_to_original']
    available_genes_standard = gene_mapping['standard_set']
    
    # Current HVGs standardized
    current_hvgs_standard = {
        standardize_gene_name(g, species) for g in current_hvgs
    }
    
    # All canonical markers
    all_canonical_standard = set()
    for data in prior_dict.values():
        all_canonical_standard.update(data['markers_standard'])
    
    # Canonical in data but not in HVGs
    canonical_in_data = all_canonical_standard & available_genes_standard
    missing_canonical_standard = canonical_in_data - current_hvgs_standard
    
    if not missing_canonical_standard:
        return current_hvgs
    
    # Convert back to original names
    markers_to_add = [
        standard_to_original[g] for g in list(missing_canonical_standard)[:max_additions]
        if g in standard_to_original
    ]
    
    if markers_to_add:
        print(f"       -> [MPS-HVG] Adding {len(markers_to_add)} protected canonical markers")
    
    return current_hvgs + markers_to_add

def detect_species_from_model_or_data(model_path: str = None, adata=None) -> str:
    """
    Attempts to detect species from CellTypist model name or gene names in data.
    
    Args:
        model_path: Path to CellTypist model file
        adata: AnnData object
    
    Returns:
        "human" or "mouse"
    """
    species = "human"  # Default
    
    # Check model path for hints
    if model_path:
        model_name = os.path.basename(model_path).lower()
        if 'mouse' in model_name or 'mm' in model_name:
            species = "mouse"
        elif 'human' in model_name or 'hs' in model_name:
            species = "human"
    
    # Check gene names in data
    if adata is not None:
        gene_names = adata.var_names.tolist()[:100]  # Sample first 100
        
        # Mouse genes typically have first letter capitalized (Cd4, Ptprc)
        # Human genes are typically all uppercase (CD4, PTPRC)
        n_uppercase = sum(1 for g in gene_names if g.isupper())
        n_titlecase = sum(1 for g in gene_names if g[0].isupper() and not g.isupper() and len(g) > 1)
        
        if n_titlecase > n_uppercase:
            species = "mouse"
        else:
            species = "human"
        
        print(f"       -> [Species Detection] Detected species: {species} "
              f"(uppercase: {n_uppercase}, titlecase: {n_titlecase})")
    
    return species

def check_existing_batch_column(adata, batch_key=None):
    """
    Checks if batch information already exists in adata.obs.
    
    Args:
        adata (anndata.AnnData): AnnData object to check.
        batch_key (str): Optional specific column name to use as batch key.
    
    Returns:
        tuple: (batch_column_name or None, bool indicating if batch found)
    """
    print("\n--- Checking for existing batch information in metadata ---")
    
    # If user specified a batch key, check for it
    if batch_key is not None:
        if batch_key in adata.obs.columns:
            n_batches = adata.obs[batch_key].nunique()
            print(f"       -> ✓ Found specified batch column '{batch_key}' with {n_batches} unique values")
            return batch_key, n_batches >= 2
        else:
            print(f"       -> ✗ Specified batch column '{batch_key}' not found in adata.obs")
            return None, False
    
    # Common batch column names to check
    common_batch_cols = ['batch', 'sample', 'orig.ident', 'orig_ident', 'Sample', 'Batch', 
                         'library', 'Library', 'donor', 'Donor', 'patient', 'Patient']
    
    for col in common_batch_cols:
        if col in adata.obs.columns:
            n_unique = adata.obs[col].nunique()
            if n_unique >= 2:
                print(f"       -> ✓ Found batch column '{col}' with {n_unique} unique values")
                
                # Show distribution
                value_counts = adata.obs[col].value_counts()
                for val, count in value_counts.head(10).items():
                    print(f"          - {val}: {count} cells")
                if len(value_counts) > 10:
                    print(f"          - ... and {len(value_counts) - 10} more batches")
                
                return col, True
    
    print("       -> No existing batch column found in metadata")
    return None, False

# --- Bayesian Optimization Imports ---
try:
    from skopt import gp_minimize, dump, load
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args
    from skopt.plots import plot_evaluations, plot_objective
except ImportError:
    print("Error: scikit-optimize is not installed. Please run 'pip install scikit-optimize'")
    exit()

# --- Visualization Imports ---
try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import umap
except ImportError:
    print("Warning: Matplotlib, Seaborn, Scikit-learn, or UMAP not installed. Visualization feature will not work.")
    print("Please run 'pip install matplotlib seaborn scikit-learn umap-learn'")


# ==============================================================================
# --- *** CONFIGURATION SECTION *** ---
# ==============================================================================
MITO_REGEX_PATTERN = r'^(MT|Mt|mt)[-._:]'

# Default search space for Stage 1, 'n_hvg' may be dynamically changed later
search_space = [
    Integer(800, 20000, name='n_hvg'),
    Integer(10, 100, name='n_pcs'),
    Integer(10, 50, name='n_neighbors'),
    Real(0.2, 1.0, name='resolution')
]

# --- Global variables for Stage 1 ---
adata_base = None
model = None
RANDOM_SEED = None
ARGS = None  # Will hold parsed command-line arguments
CURRENT_OPTIMIZATION_TARGET = None
CURRENT_STRATEGY_NAME = ""
TRIAL_METADATA = [] # Holds per-trial metadata (e.g., scores, label counts)


# ==============================================================================
# ==============================================================================
# --- *** STAGE 1: BAYESIAN OPTIMIZATION FUNCTIONS *** ---
# ==============================================================================
# ==============================================================================
@use_named_args(dimensions=search_space)
def objective_function(n_hvg, n_pcs, n_neighbors, resolution):
    """
    (Stage 1) Runs the appropriate pipeline (single-sample or integrated), calculates all
    metrics (CAS, MCS, Silhouette), and returns a score based on the global
    CURRENT_OPTIMIZATION_TARGET.
    """
    global adata_base, model, RANDOM_SEED, ARGS, CURRENT_OPTIMIZATION_TARGET, CURRENT_STRATEGY_NAME, TRIAL_METADATA

    print(f"\n---> [{CURRENT_STRATEGY_NAME}] Trial: HVGs={n_hvg}, PCs={n_pcs}, Neighbors={n_neighbors}, Resolution={resolution:.3f}")
    start_time = time.time()

    adata_proc = adata_base.copy()
    is_multi_sample = 'sample' in adata_base.obs.columns

    # Use the raw layer for annotation if it exists, otherwise use .X
    adata_for_annot = adata_proc.raw.to_adata() if adata_proc.raw is not None else adata_proc
    print("     [INFO] Annotating individual cells on full log-normalized data...")
    predictions = celltypist.annotate(adata_for_annot, model=model, majority_voting=False)
    adata_proc.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels']

    is_two_step_hvg = all(p is not None for p in [ARGS.hvg_min_mean, ARGS.hvg_max_mean, ARGS.hvg_min_disp])
    if is_two_step_hvg:
        print("     [INFO] Trial using two-step sequential HVG selection.")
        sc.pp.highly_variable_genes(
            adata_proc,
            min_mean=ARGS.hvg_min_mean,
            max_mean=ARGS.hvg_max_mean,
            min_disp=ARGS.hvg_min_disp,
            batch_key='sample' if is_multi_sample else None
        )
        hvg_df = adata_proc.var[adata_proc.var.highly_variable]
        hvg_df = hvg_df.sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:n_hvg].tolist()
        
        # NEW: Protect canonical markers
        if ARGS.protect_canonical_markers and MARKER_PRIOR_DICT:
            top_genes = _add_protected_markers(
                current_hvgs=top_genes,
                prior_dict=MARKER_PRIOR_DICT,
                adata=adata_proc,
                max_additions=int(n_hvg * 0.1)
            )
        
        adata_proc.var['highly_variable'] = False
        adata_proc.var.loc[top_genes, 'highly_variable'] = True
    else:
        print("     [INFO] Trial using rank-based HVG selection.")
        if is_multi_sample:
            sc.pp.highly_variable_genes(adata_proc, n_top_genes=n_hvg, batch_key='sample', flavor='seurat_v3')
        else:
            sc.pp.highly_variable_genes(adata_proc, n_top_genes=n_hvg, flavor='seurat_v3')

    adata_proc = adata_proc[:, adata_proc.var.highly_variable].copy()
    sc.pp.scale(adata_proc, max_value=10)
    
    # Cap n_pcs_compute by the number of available features
    n_pcs_to_compute = min(ARGS.n_pcs_compute, adata_proc.n_obs - 1, adata_proc.n_vars - 1)
    if n_pcs > n_pcs_to_compute:
        print(f"     [WARNING] Requested n_pcs ({n_pcs}) > computed PCs ({n_pcs_to_compute}). Capping at {n_pcs_to_compute}.")
        n_pcs = n_pcs_to_compute

    sc.tl.pca(adata_proc, svd_solver='arpack', n_comps=n_pcs_to_compute, random_state=RANDOM_SEED)
    embedding_to_use = 'X_pca'
    
    if is_multi_sample:
        n_batches = adata_proc.obs['sample'].nunique()
        
        if n_batches >= 2:
            # Multiple batches found - run Harmony integration
            print(f"       -> [Harmony] Running integration across {n_batches} batches...")
            try:
                sc.external.pp.harmony_integrate(
                    adata_proc,
                    key='sample',
                    basis='X_pca',
                    adjusted_basis='X_pca_harmony',
                    random_state=RANDOM_SEED
                )
                embedding_to_use = 'X_pca_harmony'
            except Exception as e:
                print(f"       -> [Harmony] Integration failed: {e}. Using standard PCA.")
                embedding_to_use = 'X_pca'
        else:
            # Only 1 batch - skip Harmony (it's meaningless with single batch)
            print(f"       -> [Harmony] Skipping: Only {n_batches} batch found. Using standard PCA.")
            embedding_to_use = 'X_pca'

    sc.pp.neighbors(adata_proc, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=embedding_to_use, random_state=RANDOM_SEED)
    sc.tl.leiden(adata_proc, resolution=resolution, random_state=RANDOM_SEED)

    silhouette_avg = 0.0
    rescaled_silhouette = 0.0
    try:
        n_clusters = adata_proc.obs['leiden'].nunique()
        if n_clusters > 1:
            silhouette_avg = silhouette_score(adata_proc.obsm[embedding_to_use][:, :n_pcs], adata_proc.obs['leiden'])
            rescaled_silhouette = (silhouette_avg + 1) / 2
        else:
            silhouette_avg = -1.0; rescaled_silhouette = 0.0
    except Exception as e:
        print(f"     [WARNING] Could not calculate silhouette score. Error: {e}. Scores set to worst values.")
        silhouette_avg = -1.0; rescaled_silhouette = 0.0

    cluster2label = adata_proc.obs.groupby('leiden')['ctpt_individual_prediction'].agg(lambda x: x.value_counts().idxmax())
    adata_proc.obs['ctpt_consensus_prediction'] = adata_proc.obs['leiden'].map(cluster2label)
    total_cells = len(adata_proc.obs)
    total_matching = (adata_proc.obs['ctpt_individual_prediction'] == adata_proc.obs['ctpt_consensus_prediction']).sum()
    weighted_mean_cas = (total_matching / total_cells) * 100 if total_cells > 0 else 0.0

    simple_mean_cas = 0.0
    if ARGS.cas_aggregation_method == 'leiden':
        cas_per_cluster = [
            g['ctpt_individual_prediction'].eq(g['ctpt_consensus_prediction'].iloc[0]).mean() * 100
            for _, g in adata_proc.obs.groupby('leiden') if not g.empty
        ]
        simple_mean_cas = np.mean(cas_per_cluster) if cas_per_cluster else 0.0
    elif ARGS.cas_aggregation_method == 'consensus':
        cas_per_consensus_group = [
            g['ctpt_individual_prediction'].eq(g['ctpt_consensus_prediction'].iloc[0]).mean() * 100
            for _, g in adata_proc.obs.groupby('ctpt_consensus_prediction') if not g.empty
        ]
        simple_mean_cas = np.mean(cas_per_consensus_group) if cas_per_consensus_group else 0.0

    mean_mcs = 0.0
    try:
        label_counts = adata_proc.obs['ctpt_consensus_prediction'].value_counts()
        valid_labels = label_counts[label_counts > 1].index.tolist()
        if len(valid_labels) >= 2:
            sc.tl.rank_genes_groups(
                adata_proc, 'ctpt_consensus_prediction', groups=valid_labels,
                method='wilcoxon', use_raw=True, key_added='rank_genes_consensus'
            )
            marker_df = sc.get.rank_genes_groups_df(adata_proc, key='rank_genes_consensus', group=None)
            is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
            if ARGS.marker_gene_model == 'non-mitochondrial':
                filtered_rows = [sub[~sub['names'].map(is_mito)].head(ARGS.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            else:
                filtered_rows = [sub.head(ARGS.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            top_genes_per_group = pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame()

            if not top_genes_per_group.empty:
                unique_top_genes = top_genes_per_group['names'].unique().tolist()
                data_df = sc.get.obs_df(adata_proc, keys=['ctpt_consensus_prediction'] + unique_top_genes, use_raw=True)
                fraction_df = data_df.groupby('ctpt_consensus_prediction').apply(lambda x: (x[unique_top_genes] > 0).mean())
                mcs_scores = {cell_type: fraction_df.loc[cell_type, top_genes_per_group[top_genes_per_group['group'] == cell_type]['names']].mean() for cell_type in top_genes_per_group['group'].unique()}
                if mcs_scores: mean_mcs = np.mean(list(mcs_scores.values())) * 100
    except Exception as e:
        print(f"     [WARNING] Could not calculate MCS for this trial. Error: {e}. MCS set to 0.")
        mean_mcs = 0.0

    mean_mps = 0.0
    mps_details = {}
    if MARKER_PRIOR_DICT:
        try:
            mean_mps, mps_details = calculate_marker_prior_score(
                adata=adata_proc,
                groupby_key='ctpt_consensus_prediction',
                prior_dict=MARKER_PRIOR_DICT,
                n_top_genes=getattr(ARGS, 'mps_n_top_genes', 100),
                n_background_genes=getattr(ARGS, 'mps_n_background', 200),
                deg_ranking_method=getattr(ARGS, 'deg_ranking_method', 'original'),
                deg_weight_fc=getattr(ARGS, 'deg_weight_fc', 0.4),
                deg_weight_expr=getattr(ARGS, 'deg_weight_expr', 0.3),
                deg_weight_pct=getattr(ARGS, 'deg_weight_pct', 0.3),
                similarity_threshold=getattr(ARGS, 'mps_similarity_threshold', 0.6),  # NEW
                verbose_matching=getattr(ARGS, 'mps_verbose_matching', False)          # NEW
            )
        except Exception as e:
            print(f"     [WARNING] MPS calculation failed: {e}. Setting to 0.")
            mean_mps = 0.0

    TRIAL_METADATA.append({
        'n_individual_labels': adata_proc.obs['ctpt_individual_prediction'].nunique(),
        'n_consensus_labels': adata_proc.obs['ctpt_consensus_prediction'].nunique(),
        'weighted_mean_cas': weighted_mean_cas, 'simple_mean_cas': simple_mean_cas,
        'mean_mcs': mean_mcs, 'mean_mps': mean_mps,  # NEW: Added MPS
        'silhouette_score_original': silhouette_avg, 'silhouette_score_rescaled': rescaled_silhouette
    })
    end_time = time.time()

    if CURRENT_OPTIMIZATION_TARGET == 'weighted_cas':
        score = weighted_mean_cas
    elif CURRENT_OPTIMIZATION_TARGET == 'simple_cas':
        score = simple_mean_cas
    elif CURRENT_OPTIMIZATION_TARGET == 'mcs':
        score = mean_mcs
    elif CURRENT_OPTIMIZATION_TARGET == 'balanced':
        epsilon = 1e-6
        
        # =======================================================================
        # ADDITIVE BONUS SYSTEM FOR MPS
        # =======================================================================
        # MPS provides a BONUS on top of the base score, not a penalty
        # mps_bonus_weight controls the maximum bonus (0.2 = up to 20% bonus)
        # 
        # Formula: Final Score = Base Score + (mps_bonus_weight * MPS)
        # 
        # Example with mps_bonus_weight = 0.2:
        #   - MPS = 0%   -> Bonus = 0%   -> Final = Base + 0%
        #   - MPS = 50%  -> Bonus = 10%  -> Final = Base + 10%
        #   - MPS = 100% -> Bonus = 20%  -> Final = Base + 20%
        # =======================================================================
        
        mps_bonus_weight = getattr(ARGS, 'mps_bonus_weight', 0.2)  # Default: 20% max bonus
        
        # Calculate BASE SCORE (without MPS) using geometric mean
        if ARGS.model_type == 'structural':
            # 4-component: CAS_w, CAS_s, MCS, Silhouette
            base_score = (((weighted_mean_cas / 100 + epsilon) * 
                          (simple_mean_cas / 100 + epsilon) * 
                          (mean_mcs / 100 + epsilon) * 
                          (rescaled_silhouette + epsilon)) ** (1/4.0)) * 100
        elif ARGS.model_type == 'silhouette':
            base_score = silhouette_avg * 100  # Scale to percentage for consistency
        else:  # 'biological' model (default)
            # 3-component: CAS_w, CAS_s, MCS
            base_score = (((weighted_mean_cas / 100 + epsilon) * 
                          (simple_mean_cas / 100 + epsilon) * 
                          (mean_mcs / 100 + epsilon)) ** (1/3.0)) * 100
        
        # Calculate MPS BONUS (only if MPS > 0 and bonus weight > 0)
        if mps_bonus_weight > 0 and mean_mps > 0:
            # MPS is already in percentage (0-100), convert to 0-1 for bonus calculation
            mps_bonus = mps_bonus_weight * mean_mps  # e.g., 0.2 * 50 = 10% bonus
            score = base_score + mps_bonus
            
            # Log the bonus for transparency
            print(f"     [MPS Bonus] Base: {base_score:.2f}% + Bonus: {mps_bonus:.2f}% (MPS: {mean_mps:.1f}%) = Final: {score:.2f}%")
        else:
            # No MPS bonus - use base score only
            score = base_score
            if mps_bonus_weight > 0:
                print(f"     [MPS Bonus] Base: {base_score:.2f}% + Bonus: 0% (MPS: {mean_mps:.1f}%) = Final: {score:.2f}%")
    else:
        raise ValueError(f"Invalid optimization target: '{CURRENT_OPTIMIZATION_TARGET}'")

    print(f"<--- Results (Time: {end_time - start_time:.1f}s) -> Score: {score:.3f} (MPS: {mean_mps:.1f}%)")
    return -score

def evaluate_final_metrics(params_dict):
    """(Stage 1) Runs the appropriate pipeline once to get final metrics and the AnnData object for saving."""
    print("\n--- Re-running analysis with overall best parameters for final report ---")
    adata_final = adata_base.copy()
    is_multi_sample = 'sample' in adata_base.obs.columns

    adata_for_annot = adata_final.raw.to_adata() if adata_final.raw is not None else adata_final
    print("     [INFO] Final run: Annotating individual cells on full log-normalized data...")
    predictions = celltypist.annotate(adata_for_annot, model=model, majority_voting=False)
    adata_final.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels']

    is_two_step_hvg = all(p is not None for p in [ARGS.hvg_min_mean, ARGS.hvg_max_mean, ARGS.hvg_min_disp])
    if is_two_step_hvg:
        print("     [INFO] Final run using two-step sequential HVG selection.")
        sc.pp.highly_variable_genes(adata_final, min_mean=ARGS.hvg_min_mean, max_mean=ARGS.hvg_max_mean, min_disp=ARGS.hvg_min_disp, batch_key='sample' if is_multi_sample else None)
        hvg_df = adata_final.var[adata_final.var.highly_variable].sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:params_dict['n_hvg']]
        adata_final.var['highly_variable'] = False
        adata_final.var.loc[top_genes, 'highly_variable'] = True
    else:
        print("     [INFO] Final run using rank-based HVG selection.")
        if is_multi_sample:
            sc.pp.highly_variable_genes(adata_final, n_top_genes=params_dict['n_hvg'], batch_key='sample', flavor='seurat_v3')
        else:
            sc.pp.highly_variable_genes(adata_final, n_top_genes=params_dict['n_hvg'], flavor='seurat_v3')

    adata_final = adata_final[:, adata_final.var.highly_variable].copy()
    sc.pp.scale(adata_final, max_value=10)
    
    n_pcs_to_compute = min(ARGS.n_pcs_compute, adata_final.n_obs - 1, adata_final.n_vars - 1)
    n_pcs = min(params_dict['n_pcs'], n_pcs_to_compute)
    sc.tl.pca(adata_final, svd_solver='arpack', n_comps=n_pcs_to_compute, random_state=RANDOM_SEED)

    embedding_to_use = 'X_pca'
    if is_multi_sample:
        sc.external.pp.harmony_integrate(adata_final, key='sample', basis='X_pca', adjusted_basis='X_pca_harmony', random_state=RANDOM_SEED)
        embedding_to_use = 'X_pca_harmony'

    sc.pp.neighbors(adata_final, n_neighbors=params_dict['n_neighbors'], n_pcs=n_pcs, use_rep=embedding_to_use, random_state=RANDOM_SEED)
    sc.tl.leiden(adata_final, resolution=params_dict['resolution'], random_state=RANDOM_SEED)
    sc.tl.umap(adata_final, random_state=RANDOM_SEED)

    silhouette_avg, rescaled_silhouette = 0.0, 0.0
    try:
        if adata_final.obs['leiden'].nunique() > 1:
            silhouette_avg = silhouette_score(adata_final.obsm[embedding_to_use][:, :n_pcs], adata_final.obs['leiden'])
            rescaled_silhouette = (silhouette_avg + 1) / 2
        else:
            silhouette_avg = -1.0; rescaled_silhouette = 0.0
    except Exception as e:
        print(f"[WARNING] Final silhouette calculation failed. Error: {e}. Scores set to worst values.")
        silhouette_avg = -1.0; rescaled_silhouette = 0.0

    cluster2label = adata_final.obs.groupby('leiden')['ctpt_individual_prediction'].agg(lambda x: x.value_counts().idxmax())
    adata_final.obs['ctpt_consensus_prediction'] = adata_final.obs['leiden'].map(cluster2label)
    total_cells, total_matching = len(adata_final.obs), (adata_final.obs['ctpt_individual_prediction'] == adata_final.obs['ctpt_consensus_prediction']).sum()
    weighted_cas = (total_matching / total_cells) * 100 if total_cells > 0 else 0.0

    simple_cas = 0.0
    if ARGS.cas_aggregation_method == 'leiden':
        cas_per_cluster = [g['ctpt_individual_prediction'].eq(g['ctpt_consensus_prediction'].iloc[0]).mean() * 100 for _, g in adata_final.obs.groupby('leiden') if not g.empty]
        simple_cas = np.mean(cas_per_cluster) if cas_per_cluster else 0.0
    elif ARGS.cas_aggregation_method == 'consensus':
        cas_per_consensus_group = [g['ctpt_individual_prediction'].eq(g['ctpt_consensus_prediction'].iloc[0]).mean() * 100 for _, g in adata_final.obs.groupby('ctpt_consensus_prediction') if not g.empty]
        simple_cas = np.mean(cas_per_consensus_group) if cas_per_consensus_group else 0.0

    mean_mcs = 0.0
    try:
        label_counts = adata_final.obs['ctpt_consensus_prediction'].value_counts()
        valid_labels = label_counts[label_counts > 1].index.tolist()
        if len(valid_labels) >= 2:
            sc.tl.rank_genes_groups(adata_final, 'ctpt_consensus_prediction', groups=valid_labels, method='wilcoxon', use_raw=True, key_added='rank_genes_consensus')
            marker_df = sc.get.rank_genes_groups_df(adata_final, key='rank_genes_consensus', group=None)
            is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
            if ARGS.marker_gene_model == 'non-mitochondrial':
                filtered_rows = [sub[~sub['names'].map(is_mito)].head(ARGS.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            else:
                filtered_rows = [sub.head(ARGS.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            top_genes_per_group = pd.concat(filtered_rows, ignore_index=True) if filtered_rows else pd.DataFrame()
            if not top_genes_per_group.empty:
                unique_top_genes = top_genes_per_group['names'].unique().tolist()
                data_df = sc.get.obs_df(adata_final, keys=['ctpt_consensus_prediction'] + unique_top_genes, use_raw=True)
                fraction_df = data_df.groupby('ctpt_consensus_prediction').apply(lambda x: (x[unique_top_genes] > 0).mean())
                mcs_scores = {cell_type: fraction_df.loc[cell_type, top_genes_per_group[top_genes_per_group['group'] == cell_type]['names']].mean() for cell_type in top_genes_per_group['group'].unique()}
                if mcs_scores: mean_mcs = np.mean(list(mcs_scores.values())) * 100
    except Exception as e:
        print(f"[WARNING] Final MCS calculation failed. Error: {e}. MCS set to 0.")
        mean_mcs = 0.0

    mean_mps = 0.0
    mps_details = {}
    if MARKER_PRIOR_DICT:
        try:
            mean_mps, mps_details = calculate_marker_prior_score(
                adata=adata_final,
                groupby_key='ctpt_consensus_prediction',
                prior_dict=MARKER_PRIOR_DICT,
                n_top_genes=getattr(ARGS, 'n_degs_for_mps', 100),  # FIX: correct parameter name
                n_background_genes=200,
                deg_ranking_method=getattr(ARGS, 'deg_ranking_method', 'original'),
                deg_weight_fc=getattr(ARGS, 'deg_weight_fc', 0.4),
                deg_weight_expr=getattr(ARGS, 'deg_weight_expr', 0.3),
                deg_weight_pct=getattr(ARGS, 'deg_weight_pct', 0.3),
                similarity_threshold=getattr(ARGS, 'mps_similarity_threshold', 0.6),
                verbose_matching=getattr(ARGS, 'mps_verbose_matching', False),
                min_cells_per_group=getattr(ARGS, 'mps_min_cells_per_group', 3)
            )
        except Exception as e:
            print(f"[WARNING] Final MPS calculation failed: {e}. Setting to 0.")
            mean_mps = 0.0

    epsilon = 1e-6
    mps_bonus_weight = getattr(ARGS, 'mps_bonus_weight', 0.2)  # Default: 20% max bonus
    
    # =======================================================================
    # ADDITIVE BONUS SYSTEM FOR MPS (Final Metrics)
    # =======================================================================
    
    # Calculate BASE SCORE (without MPS)
    if ARGS.model_type == 'structural':
        # 4-component: CAS_w, CAS_s, MCS, Silhouette
        base_score = (((weighted_cas / 100 + epsilon) * 
                      (simple_cas / 100 + epsilon) * 
                      (mean_mcs / 100 + epsilon) * 
                      (rescaled_silhouette + epsilon)) ** (1/4.0)) * 100
    elif ARGS.model_type == 'silhouette':
        base_score = silhouette_avg * 100
    else:  # 'biological' model (default)
        # 3-component: CAS_w, CAS_s, MCS
        base_score = (((weighted_cas / 100 + epsilon) * 
                      (simple_cas / 100 + epsilon) * 
                      (mean_mcs / 100 + epsilon)) ** (1/3.0)) * 100
    
    # Calculate MPS BONUS
    if mps_bonus_weight > 0 and mean_mps > 0:
        mps_bonus = mps_bonus_weight * mean_mps
        balanced_score = base_score + mps_bonus
        print(f"    [Final MPS Bonus] Base: {base_score:.2f}% + Bonus: {mps_bonus:.2f}% = Final: {balanced_score:.2f}%")
    else:
        balanced_score = base_score

    return {
        "weighted_mean_cas": weighted_cas, "simple_mean_cas": simple_cas, "mean_mcs": mean_mcs,
        "mean_mps": mean_mps,  # NEW
        "mps_details": mps_details,  # NEW: Per-cell-type breakdown
        "silhouette_score_original": silhouette_avg, "rescaled_silhouette_score": rescaled_silhouette,
        "balanced_score": balanced_score, "n_individual_labels": adata_final.obs['ctpt_individual_prediction'].nunique(),
        "n_consensus_labels": adata_final.obs['ctpt_consensus_prediction'].nunique()
    }, adata_final

def print_final_report(target, best_params, final_metrics, winning_strategy_name, 
                       stored_trial_metadata=None):
    """
    Prints the final optimization report.
    
    NOW USES stored trial metadata for score consistency with CSV output.
    If stored_trial_metadata is provided, uses those values instead of re-evaluated metrics.
    
    Args:
        target: Optimization target ('balanced', 'weighted_cas', etc.)
        best_params: Dictionary of best parameters
        final_metrics: Re-evaluated metrics (used as fallback)
        winning_strategy_name: Name of the winning optimization strategy
        stored_trial_metadata: Optional dict with stored scores from the actual winning trial
    """
    global ARGS
    
    print("\n" + "="*70)
    print("FINAL OPTIMIZATION REPORT")
    print("="*70)
    
    print(f"\nOptimization Target: {target.upper()}")
    print(f"Winning Strategy: {winning_strategy_name}")
    print(f"Model Type: {ARGS.model_type}")
    
    print("\n--- Best Parameters ---")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")
    
    print("\n--- Final Metrics ---")
    
    # =========================================================================
    # USE STORED TRIAL METADATA IF AVAILABLE (for consistency with CSV)
    # =========================================================================
    if stored_trial_metadata is not None:
        print("  [Using stored trial scores for consistency with optimization CSV]")
        
        # Extract scores from stored metadata
        weighted_cas = stored_trial_metadata.get('weighted_mean_cas', final_metrics.get('weighted_mean_cas', 0))
        simple_cas = stored_trial_metadata.get('simple_mean_cas', final_metrics.get('simple_mean_cas', 0))
        mean_mcs = stored_trial_metadata.get('mean_mcs', final_metrics.get('mean_mcs', 0))
        mean_mps = stored_trial_metadata.get('mean_mps', final_metrics.get('mean_mps', 0))
        silhouette_original = stored_trial_metadata.get('silhouette_score_original', final_metrics.get('silhouette_score', 0))
        silhouette_rescaled = stored_trial_metadata.get('silhouette_score_rescaled', final_metrics.get('silhouette_rescaled', 0))
        n_individual = stored_trial_metadata.get('n_individual_labels', final_metrics.get('n_individual_labels', 0))
        n_consensus = stored_trial_metadata.get('n_consensus_labels', final_metrics.get('n_consensus_labels', 0))
        
    else:
        print("  [Using re-evaluated metrics - stored trial data not available]")
        
        # Fallback to re-evaluated metrics
        weighted_cas = final_metrics.get('weighted_mean_cas', 0)
        simple_cas = final_metrics.get('simple_mean_cas', 0)
        mean_mcs = final_metrics.get('mean_mcs', 0)
        mean_mps = final_metrics.get('mean_mps', 0)
        silhouette_original = final_metrics.get('silhouette_score', 0)
        silhouette_rescaled = final_metrics.get('silhouette_rescaled', 0)
        n_individual = final_metrics.get('n_individual_labels', 0)
        n_consensus = final_metrics.get('n_consensus_labels', 0)
    
    # Print all metrics
    print(f"  Weighted Mean CAS: {weighted_cas:.2f}%")
    print(f"  Simple Mean CAS: {simple_cas:.2f}%")
    print(f"  Mean MCS: {mean_mcs:.2f}%")
    
    if mean_mps > 0 or (stored_trial_metadata and 'mean_mps' in stored_trial_metadata):
        print(f"  Mean MPS (Marker Prior Score): {mean_mps:.2f}%")
    
    print(f"  Silhouette Score (Original): {silhouette_original:.4f}")
    print(f"  Silhouette Score (Rescaled 0-1): {silhouette_rescaled:.4f}")
    print(f"  Number of Individual Labels: {n_individual}")
    print(f"  Number of Consensus Labels: {n_consensus}")
    
    # =========================================================================
    # Calculate and display the EXACT balanced score that was used for ranking
    # =========================================================================
    print("\n--- Balanced Score Calculation ---")
    
    if ARGS.model_type == 'silhouette':
        final_score = silhouette_rescaled * 100
        print(f"  Mode: Silhouette-only")
        print(f"  Final Score: {final_score:.2f}")
        
    elif ARGS.model_type == 'structural':
        # Structural: includes silhouette in the geometric mean
        epsilon = 1e-6
        base_gmean = (max(weighted_cas, epsilon) * max(mean_mcs, epsilon) * max(silhouette_rescaled * 100, epsilon)) ** (1/3)
        
        # Add MPS bonus if applicable
        mps_bonus_weight = getattr(ARGS, 'mps_bonus_weight', 0.2)
        mps_bonus = mps_bonus_weight * mean_mps
        final_score = base_gmean + mps_bonus
        
        print(f"  Mode: Structural (CAS × MCS × Silhouette)")
        print(f"  Base G-Mean: {base_gmean:.2f}")
        if mean_mps > 0:
            print(f"  MPS Bonus ({mps_bonus_weight*100:.0f}% × {mean_mps:.2f}%): +{mps_bonus:.2f}")
        print(f"  Final Score: {final_score:.2f}")
        
    else:  # biological (default)
        epsilon = 1e-6
        base_gmean = (max(weighted_cas, epsilon) * max(mean_mcs, epsilon)) ** 0.5
        
        # Add MPS bonus if applicable
        mps_bonus_weight = getattr(ARGS, 'mps_bonus_weight', 0.2)
        mps_bonus = mps_bonus_weight * mean_mps
        final_score = base_gmean + mps_bonus
        
        print(f"  Mode: Biological (CAS × MCS)")
        print(f"  Base G-Mean (√(CAS × MCS)): {base_gmean:.2f}")
        if mean_mps > 0:
            print(f"  MPS Bonus ({mps_bonus_weight*100:.0f}% × {mean_mps:.2f}%): +{mps_bonus:.2f}")
        print(f"  Final Score (used for ranking): {final_score:.2f}")
    
    print("\n" + "="*70)


def save_results_to_file(filepath, target, best_params, final_metrics, winning_strategy_name,
                         stored_trial_metadata=None):
    """
    Saves the optimization results to a text file.
    
    NOW USES stored trial metadata for consistency with CSV output.
    
    Args:
        filepath: Path to save the results file
        target: Optimization target
        best_params: Dictionary of best parameters
        final_metrics: Re-evaluated metrics (used as fallback)
        winning_strategy_name: Name of winning strategy
        stored_trial_metadata: Optional dict with stored scores from the actual winning trial
    """
    global ARGS
    
    # =========================================================================
    # USE STORED TRIAL METADATA IF AVAILABLE
    # =========================================================================
    if stored_trial_metadata is not None:
        weighted_cas = stored_trial_metadata.get('weighted_mean_cas', final_metrics.get('weighted_mean_cas', 0))
        simple_cas = stored_trial_metadata.get('simple_mean_cas', final_metrics.get('simple_mean_cas', 0))
        mean_mcs = stored_trial_metadata.get('mean_mcs', final_metrics.get('mean_mcs', 0))
        mean_mps = stored_trial_metadata.get('mean_mps', final_metrics.get('mean_mps', 0))
        silhouette_original = stored_trial_metadata.get('silhouette_score_original', final_metrics.get('silhouette_score', 0))
        silhouette_rescaled = stored_trial_metadata.get('silhouette_score_rescaled', final_metrics.get('silhouette_rescaled', 0))
        n_individual = stored_trial_metadata.get('n_individual_labels', final_metrics.get('n_individual_labels', 0))
        n_consensus = stored_trial_metadata.get('n_consensus_labels', final_metrics.get('n_consensus_labels', 0))
        score_source = "stored_trial"
    else:
        weighted_cas = final_metrics.get('weighted_mean_cas', 0)
        simple_cas = final_metrics.get('simple_mean_cas', 0)
        mean_mcs = final_metrics.get('mean_mcs', 0)
        mean_mps = final_metrics.get('mean_mps', 0)
        silhouette_original = final_metrics.get('silhouette_score', 0)
        silhouette_rescaled = final_metrics.get('silhouette_rescaled', 0)
        n_individual = final_metrics.get('n_individual_labels', 0)
        n_consensus = final_metrics.get('n_consensus_labels', 0)
        score_source = "re_evaluated"
    
    # Calculate balanced score
    epsilon = 1e-6
    mps_bonus_weight = getattr(ARGS, 'mps_bonus_weight', 0.2)
    
    if ARGS.model_type == 'silhouette':
        balanced_score = silhouette_rescaled * 100
    elif ARGS.model_type == 'structural':
        base_gmean = (max(weighted_cas, epsilon) * max(mean_mcs, epsilon) * max(silhouette_rescaled * 100, epsilon)) ** (1/3)
        balanced_score = base_gmean + (mps_bonus_weight * mean_mps)
    else:  # biological
        base_gmean = (max(weighted_cas, epsilon) * max(mean_mcs, epsilon)) ** 0.5
        balanced_score = base_gmean + (mps_bonus_weight * mean_mps)
    
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BAYESIAN OPTIMIZATION - FINAL RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Score_Source: {score_source}\n")
        f.write(f"Optimization_Target: {target}\n")
        f.write(f"Winning_Strategy: {winning_strategy_name}\n")
        f.write(f"Model_Type: {ARGS.model_type}\n")
        f.write(f"Random_Seed: {ARGS.seed}\n\n")
        
        f.write("--- Best Parameters ---\n")
        for param, value in best_params.items():
            if isinstance(value, float):
                f.write(f"{param}: {value:.4f}\n")
            else:
                f.write(f"{param}: {value}\n")
        
        f.write("\n--- Final Metrics ---\n")
        f.write(f"Highest_balanced_score: {balanced_score:.4f}\n")
        f.write(f"Corresponding_weighted_mean_cas: {weighted_cas:.2f}\n")
        f.write(f"Corresponding_simple_mean_cas: {simple_cas:.2f}\n")
        f.write(f"Corresponding_mean_mcs: {mean_mcs:.2f}\n")
        f.write(f"Corresponding_mean_mps: {mean_mps:.2f}\n")
        f.write(f"Corresponding_marker_prior_score_pct: {mean_mps:.2f}\n")
        f.write(f"Corresponding_silhouette_score: {silhouette_original:.4f}\n")
        f.write(f"Corresponding_silhouette_rescaled: {silhouette_rescaled:.4f}\n")
        f.write(f"Final_n_individual_labels: {n_individual}\n")
        f.write(f"Final_n_consensus_labels: {n_consensus}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"Results saved to: {filepath}")

def generate_yield_csv(results_dict, target_metric, output_dir, output_prefix):
    """(Stage 1) Generates a consolidated CSV file with detailed results from all trials."""
    print("\n--- Generating consolidated yield CSV report ---")
    param_names = ['n_hvg', 'n_pcs', 'n_neighbors', 'resolution']
    all_dfs = []
    for name, result in results_dict.items():
        params_df = pd.DataFrame(result.x_iters, columns=param_names)
        if hasattr(result, 'trial_metadata') and len(result.trial_metadata) == len(params_df):
            metadata_df = pd.DataFrame(result.trial_metadata)
            base_df = pd.concat([params_df, metadata_df], axis=1)
        else:
            print(f"  [WARNING] Per-trial metadata not found for strategy '{name}'. Metric/label columns will be empty.")
            base_df = params_df.copy()
            for col in ['n_individual_labels', 'n_consensus_labels', 'weighted_mean_cas', 'simple_mean_cas', 'mean_mcs', 'mean_mps', 'silhouette_score_original', 'silhouette_score_rescaled']:
                base_df[col] = np.nan
        base_df['yield_score_target'], base_df['call_number'], base_df['strategy'] = -np.array(result.func_vals), range(1, len(result.func_vals) + 1), name
        all_dfs.append(base_df)
    if not all_dfs: print("  [ERROR] No results found to generate CSV. Skipping."); return
    final_df = pd.concat(all_dfs, ignore_index=True)
    epsilon = 1e-6

    required_cols = ['weighted_mean_cas', 'simple_mean_cas', 'mean_mcs', 'silhouette_score_rescaled', 'silhouette_score_original']
    if all(col in final_df.columns for col in required_cols):
        mps_weight = ARGS.mps_weight if hasattr(ARGS, 'mps_weight') else 0.2  # Default to 0.2 for additive bonus
        has_mps = 'mean_mps' in final_df.columns and final_df['mean_mps'].notna().any() and mps_weight > 0
        
        if ARGS.model_type == 'structural':
            # Base score: 4-component geometric mean (CAS_w, CAS_s, MCS, Silhouette)
            base_score = (((final_df['weighted_mean_cas'].fillna(0) / 100 + epsilon) *
                           (final_df['simple_mean_cas'].fillna(0) / 100 + epsilon) *
                           (final_df['mean_mcs'].fillna(0) / 100 + epsilon) *
                           (final_df['silhouette_score_rescaled'].fillna(0) + epsilon)) ** (1/4.0)) * 100
            
            if has_mps:
                # ADDITIVE BONUS: MPS adds up to (mps_weight * 100)% bonus
                # MPS=100% gives full bonus, MPS=0% gives no bonus (but no penalty)
                mps_bonus = mps_weight * final_df['mean_mps'].fillna(0)
                final_df['balanced_score_gmean'] = base_score + mps_bonus
                print(f"  📊 Structural model: Base (4-component) + MPS bonus (weight={mps_weight})")
            else:
                final_df['balanced_score_gmean'] = base_score
                print(f"  📊 Structural model: Base (4-component), no MPS available")
                
        elif ARGS.model_type == 'silhouette':
            final_df['balanced_score_gmean'] = final_df['silhouette_score_original']
            print(f"  📊 Silhouette model: Using silhouette_score_original directly")
            
        else:  # 'biological' model
            # Base score: 3-component geometric mean (CAS_w, CAS_s, MCS)
            base_score = (((final_df['weighted_mean_cas'].fillna(0) / 100 + epsilon) *
                           (final_df['simple_mean_cas'].fillna(0) / 100 + epsilon) *
                           (final_df['mean_mcs'].fillna(0) / 100 + epsilon)) ** (1/3.0)) * 100
            
            if has_mps:
                # ADDITIVE BONUS: MPS adds up to (mps_weight * 100)% bonus
                mps_bonus = mps_weight * final_df['mean_mps'].fillna(0)
                final_df['balanced_score_gmean'] = base_score + mps_bonus
                print(f"  📊 Biological model: Base (3-component) + MPS bonus (weight={mps_weight})")
            else:
                final_df['balanced_score_gmean'] = base_score
                print(f"  📊 Biological model: Base (3-component), no MPS available")
        
        # Log example calculation for first row (debugging/transparency)
        if len(final_df) > 0 and has_mps:
            first_row = final_df.iloc[0]
            print(f"  📈 Example (row 0): Base={base_score.iloc[0]:.2f}, "
                  f"MPS={first_row.get('mean_mps', 0):.2f}, "
                  f"Bonus={mps_weight * first_row.get('mean_mps', 0):.2f}, "
                  f"Final={first_row['balanced_score_gmean']:.2f}")
    else:
        final_df['balanced_score_gmean'] = np.nan
        print(f"  ⚠️ Missing required columns for balanced_score_gmean calculation")
        
    final_df.rename(columns={'silhouette_score_original': 'silhouette_score'}, inplace=True)
    final_column_order = [
        'call_number', 'strategy', 
        'n_hvg', 'n_pcs', 'n_neighbors', 'resolution', 
        'yield_score_target', 'balanced_score_gmean', 
        'weighted_mean_cas', 'simple_mean_cas', 
        'mean_mcs', 'mean_mps',  # <-- MPS column included
        'silhouette_score', 
        'n_individual_labels', 'n_consensus_labels'
    ]
    final_df = final_df.reindex(columns=final_column_order)
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_yield_scores_report.csv")
    final_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"✅ Success! Saved consolidated CSV report to: {output_path}")

def plot_optimizer_paths_tsne(results, target_metric, output_dir, output_prefix, n_points_to_show=25):
    """(Stage 1) Generates a t-SNE plot of the search space with publication-quality styling."""
    print("\n--- Generating t-SNE visualization with publication-quality style ---")
    all_points = np.array(list(set(tuple(p) for res in results.values() for p in res.x_iters)))
    if len(all_points) <= 1: print("Skipping t-SNE plot: not enough unique points to embed."); return
    print(f"Found {len(all_points)} unique points. Performing t-SNE embedding...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_points) - 1), random_state=RANDOM_SEED, max_iter=1000, init='pca', learning_rate='auto')
    tsne_coords_map = {tuple(p): tsne_coord for p, tsne_coord in zip(all_points, tsne.fit_transform(StandardScaler().fit_transform(all_points)))}
    all_tsne_coords = np.array(list(tsne_coords_map.values()))
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(12, 10)); ax.grid(False)
    cluster_labels = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init='auto').fit_predict(all_tsne_coords)
    ax.scatter(all_tsne_coords[:, 0], all_tsne_coords[:, 1], c=cluster_labels, cmap='tab10', alpha=0.2, s=80, zorder=1)
    colors = {'Exploit': '#d62728', 'BO-EI': "#fcbe06", 'Explore': "#9015d2"}
    for name, result in results.items():
        if name in colors:
            path_coords = np.array([tsne_coords_map[tuple(p)] for p in result.x_iters[:n_points_to_show]])
            for i in range(len(path_coords) - 1): ax.annotate('', xy=path_coords[i+1], xytext=path_coords[i], arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.5", color=colors[name], lw=2.0, shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0.2"), zorder=3)
    legend = ax.legend(handles=[Line2D([0], [0], label=name, color=color, linestyle='-', linewidth=4) for name, color in colors.items() if name in results], title='Strategy', fontsize=28, loc='best', title_fontsize=28)
    legend.get_title().set_fontweight('bold'); [text.set_fontweight('bold') for text in legend.get_texts()]
    ax.set_title(f"Optimizer Paths (First {n_points_to_show} Steps)\nTarget: {target_metric.replace('_', ' ').title()}", fontsize=28, fontweight='bold'); ax.set_xlabel("t-SNE 1", fontsize=28, fontweight='bold'); ax.set_ylabel("t-SNE 2", fontsize=28, fontweight='bold'); ax.tick_params(axis='both', which='major', labelsize=28, width=1.2)
    [label.set_fontweight('bold') for label in ax.get_xticklabels() + ax.get_yticklabels()]
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_optimizer_paths_tsne.png")
    plt.savefig(output_path, dpi=500, bbox_inches='tight'); plt.close()
    print(f"✅ Success! Saved t-SNE plot to: {output_path}")
def plot_optimizer_paths_umap(results, target_metric, output_dir, output_prefix, n_points_to_show=25):
    """(Stage 1) Generates a UMAP plot of the search space with publication-quality styling."""
    print("\n--- Generating UMAP visualization with publication-quality style ---")
    all_points = np.array(list(set(tuple(p) for res in results.values() for p in res.x_iters)))
    if len(all_points) <= 1: print("Skipping UMAP plot: not enough unique points to embed."); return
    print(f"Found {len(all_points)} unique points. Performing UMAP embedding...")

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED)
    umap_coords_map = {tuple(p): umap_coord for p, umap_coord in zip(all_points, reducer.fit_transform(StandardScaler().fit_transform(all_points)))}
    all_umap_coords = np.array(list(umap_coords_map.values()))

    plt.style.use('seaborn-v0_8-white'); fig, ax = plt.subplots(figsize=(12, 10)); ax.grid(False)
    cluster_labels = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init='auto').fit_predict(all_umap_coords)
    ax.scatter(all_umap_coords[:, 0], all_umap_coords[:, 1], c=cluster_labels, cmap='tab10', alpha=0.2, s=80, zorder=1)
    colors = {'Exploit': '#d62728', 'BO-EI': "#fcbe06", 'Explore': "#9015d2"}

    for name, result in results.items():
        if name in colors:
            path_coords = np.array([umap_coords_map[tuple(p)] for p in result.x_iters[:n_points_to_show]])
            for i in range(len(path_coords) - 1): ax.annotate('', xy=path_coords[i+1], xytext=path_coords[i], arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.5", color=colors[name], lw=2.0, shrinkA=2, shrinkB=2, connectionstyle="arc3,rad=0.2"), zorder=3)
    
    legend = ax.legend(handles=[Line2D([0], [0], label=name, color=color, linestyle='-', linewidth=4) for name, color in colors.items() if name in results], title='Strategy', fontsize=28, loc='best', title_fontsize=28)
    legend.get_title().set_fontweight('bold'); [text.set_fontweight('bold') for text in legend.get_texts()]
    ax.set_title(f"Optimizer Paths (First {n_points_to_show} Steps)\nTarget: {target_metric.replace('_', ' ').title()}", fontsize=28, fontweight='bold')
    ax.set_xlabel("UMAP 1", fontsize=28, fontweight='bold'); ax.set_ylabel("UMAP 2", fontsize=28, fontweight='bold'); ax.tick_params(axis='both', which='major', labelsize=28, width=1.2)
    [label.set_fontweight('bold') for label in ax.get_xticklabels() + ax.get_yticklabels()]
    
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_optimizer_paths_umap.png")
    plt.savefig(output_path, dpi=500, bbox_inches='tight'); plt.close()
    print(f"✅ Success! Saved UMAP plot to: {output_path}")
def plot_optimizer_convergence(results, target_metric, output_dir, output_prefix):
    """(Stage 1) Generates a convergence plot with publication-quality styling."""
    print("\n--- Generating convergence plot with publication-quality style ---")
    plt.style.use('seaborn-v0_8-white'); fig, ax = plt.subplots(figsize=(22, 10)); ax.grid(False)
    colors, font_size, max_x = {'Exploit': '#d62728', 'BO-EI': "#fcbe06", 'Explore': "#9015d2"}, 28, 0
    for name, result in results.items():
        if name in colors:
            best_so_far = np.maximum.accumulate(-np.array(result.func_vals))
            x = np.arange(1, len(best_so_far) + 1); max_x = max(max_x, x.max())
            ax.plot(x, best_so_far, marker='o', linestyle='-', lw=3, color=colors[name], label=name, alpha=0.9)
    title_map = {'weighted_cas': 'Weighted Mean CAS', 'simple_cas': 'Simple Mean CAS', 'mcs': 'Mean MCS', 'balanced': 'Balanced Score (CAS & MCS)'}
    if ARGS.model_type == 'structural': title_map['balanced'] = 'Balanced Score (CAS, MCS & Silhouette)'
    elif ARGS.model_type == 'silhouette': title_map['balanced'] = 'Silhouette Score'
    ax.set_title(f"Bayesian Optimization Convergence\nTarget: {title_map.get(target_metric, target_metric)}", fontsize=font_size, fontweight='bold'); ax.set_xlabel('Call Number (Experiment Iteration)', fontsize=font_size, fontweight='bold'); ax.set_ylabel(f"Best Score Found", fontsize=font_size, fontweight='bold')
    legend = ax.legend(title='Strategy', fontsize=font_size, loc='best', title_fontsize=font_size); legend.get_title().set_fontweight('bold'); [text.set_fontweight('bold') for text in legend.get_texts()]
    ax.tick_params(axis='both', which='major', labelsize=font_size, width=1.2, direction='out', length=6); [label.set_fontweight('bold') for label in ax.get_xticklabels() + ax.get_yticklabels()]; ax.set_xlim(left=0, right=max_x + 1 if max_x > 0 else 1)
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_optimizer_convergence.png"); plt.savefig(output_path, dpi=500, bbox_inches='tight'); plt.close()
    print(f"✅ Success! Saved convergence plot to: {output_path}")
def plot_exact_scores_per_trial(results, target_metric, output_dir, output_prefix):
    """(Stage 1) Generates a plot of the exact score for each trial with publication-quality styling."""
    print("\n--- Generating per-trial exact score plot with publication-quality style ---")
    plt.style.use('seaborn-v0_8-white'); fig, ax = plt.subplots(figsize=(22, 10)); ax.grid(False)
    colors, font_size, max_x = {'Exploit': '#d62728', 'BO-EI': "#fcbe06", 'Explore': "#9015d2"}, 28, 0
    for name, result in results.items():
        if name in colors:
            exact_scores = -np.array(result.func_vals)
            x = np.arange(1, len(exact_scores) + 1); max_x = max(max_x, x.max())
            ax.plot(x, exact_scores, marker='.', linestyle='-', lw=2.5, color=colors[name], label=name, alpha=0.85)

    title_map = {'weighted_cas': 'Weighted Mean CAS', 'simple_cas': 'Simple Mean CAS', 'mcs': 'Mean MCS', 'balanced': 'Balanced Score (CAS & MCS)'}
    if ARGS and ARGS.model_type == 'structural': title_map['balanced'] = 'Balanced Score (CAS, MCS & Silhouette)'
    elif ARGS and ARGS.model_type == 'silhouette': title_map['balanced'] = 'Silhouette Score'
    ax.set_title(f"Per-Trial Score Progression\nTarget: {title_map.get(target_metric, target_metric)}", fontsize=font_size, fontweight='bold')
    ax.set_xlabel('Call Number (Experiment Iteration)', fontsize=font_size, fontweight='bold'); ax.set_ylabel(f"Score of Individual Trial", fontsize=font_size, fontweight='bold')
    legend = ax.legend(title='Strategy', fontsize=font_size, loc='best', title_fontsize=font_size); legend.get_title().set_fontweight('bold'); [text.set_fontweight('bold') for text in legend.get_texts()]
    ax.tick_params(axis='both', which='major', labelsize=font_size, width=1.2, direction='out', length=6); [label.set_fontweight('bold') for label in ax.get_xticklabels() + ax.get_yticklabels()]; ax.set_xlim(left=0, right=max_x + 1 if max_x > 0 else 1)
    
    output_path = os.path.join(output_dir, f"{output_prefix}_{target_metric}_per_trial_exact_scores.png")
    plt.savefig(output_path, dpi=500, bbox_inches='tight'); plt.close()
    print(f"✅ Success! Saved per-trial exact score plot to: {output_path}")
def _get_metric_and_strategy_from_filename(filename):
    """(Stage 1) Helper to parse metric and strategy from a .skopt filename."""
    base = os.path.basename(filename).lower()
    if 'bo_ei' in base: strategy = 'BO-EI'
    elif 'exploit' in base: strategy = 'Exploit'
    elif 'explore' in base: strategy = 'Explore'
    else: strategy = (m.group(1).capitalize() if (m := re.search(r'_(\w+)_opt_result', base)) else 'Unknown')
    if 'weighted_cas' in base: metric_label = 'Weighted CAS (%)'
    elif 'simple_cas' in base: metric_label = 'Simple CAS (%)'
    elif 'mcs' in base: metric_label = 'Mean MCS (%)'
    elif 'balanced' in base and ARGS.model_type == 'silhouette': metric_label = 'Silhouette Score'
    elif 'balanced' in base: metric_label = 'Balanced Score (%)'
    else: metric_label = 'Objective Score (%)'
    return metric_label, strategy
def _style_skopt_axes(fig):
    """(Stage 1) Applies bold styling to all axes in a matplotlib figure."""
    for ax in fig.get_axes():
        ax.xaxis.label.set_fontweight('bold'); ax.yaxis.label.set_fontweight('bold')
        for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontweight('bold')
def generate_skopt_visualizations(skopt_files, output_prefix_base, target_metric):
    """(Stage 1) Loads saved .skopt results and generates detailed visualizations for each strategy."""
    print("\n--- Generating detailed skopt visualizations (Evaluations & Objective Landscape) ---")
    for f in skopt_files:
        try:
            res = load(f)
            metric_label, strategy = _get_metric_and_strategy_from_filename(f)
            clean_title = f"{strategy} ({metric_label.replace(' (%)', '')})"
            print(f"  -> Processing plots for strategy: {strategy}")
            plot_evaluations(res); fig_eval = plt.gcf(); fig_eval.set_size_inches(14, 14); fig_eval.suptitle(f'Pairwise Parameter Evaluations: {clean_title}', fontsize=16, y=0.98, fontweight='bold'); _style_skopt_axes(fig_eval); fig_eval.text(0.5, -0.02, "Diagonal: Distribution of tested values. Off-diagonal: Correlation between parameters.", ha='center', va='top', fontsize=10, fontweight='bold'); fig_eval.savefig(f"{output_prefix_base}_{strategy}_evaluations.png", dpi=300, bbox_inches='tight'); plt.close(fig_eval)
            plot_objective(res, n_points=50); fig_obj = plt.gcf(); fig_obj.set_size_inches(14, 14); fig_obj.suptitle(f'Objective Function Landscape: {clean_title}', fontsize=16, y=0.98, fontweight='bold')
            for ax in fig_obj.get_axes():
                if "Partial dependence" in ax.get_ylabel():
                    ax.set_ylabel(metric_label, fontweight='bold')
                    yticks = ax.get_yticks()
                    if np.any(yticks < 0):
                        if '%' in metric_label:
                            ax.set_yticklabels([f"{-y:.1f}" for y in yticks])
                        else: # For non-percentage scores like silhouette
                            ax.set_yticklabels([f"{-y:.2f}" for y in yticks])
            _style_skopt_axes(fig_obj); fig_obj.text(0.5, -0.02, f"Diagonal: Effect of a single parameter on the score.\nOff-diagonal: Interaction effects between two parameters.", ha='center', va='top', fontsize=10, fontweight='bold'); fig_obj.savefig(f"{output_prefix_base}_{strategy}_objective.png", dpi=300, bbox_inches='tight'); plt.close(fig_obj)
            print(f"     ✅ Saved Evaluations and Objective plots for {strategy}")
        except Exception as e: print(f"  [ERROR] Could not generate skopt plots for {f}. Reason: {e}")

# ==============================================================================
# ==============================================================================
# --- *** STAGE 2 & HELPER FUNCTIONS *** ---
# ==============================================================================
# ==============================================================================

# ==============================================================================
# --- START: INTEGRATED REFINEMENT JOURNEY SUMMARY FUNCTION ---
# ==============================================================================
def summarize_annotation_journey(input_file, output_file):
    """
    (Refinement Helper) Reads a detailed annotation scores log and creates a 
    summarized, wide-format table tracking each cell type across analysis stages.

    Args:
        input_file (str): Path to the input CSV file 
                          (e.g., 'sc_analysis_repro_combined_cluster_annotation_scores.csv').
        output_file (str): Path to save the summarized output CSV file.
    """
    try:
        print(f"Reading input file for journey summary: {input_file}")
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        sys.exit(1)

    # --- 1. Data Cleaning and Pre-aggregation ---
    # MODIFICATION START: Standardize column names for aggregation
    df.rename(columns={
        'Total_Cells_in_Group': 'Total_Cells',
        'Matching_Individual_Predictions': 'Matching_Cells',
        'Cluster_Annotation_Score_CAS (%)': 'CAS_Score'
    }, inplace=True)
    # MODIFICATION END
    
    df['source_level'] = df['source_level'].replace('initial_high_confidence', 'initial')

    print("Aggregating data for each analysis stage...")
    agg_df = df.groupby(['Consensus_Cell_Type', 'source_level']).agg(
        Total_Cells=('Total_Cells', 'sum'),
        Matching_Cells=('Matching_Cells', 'sum')
    ).reset_index()

    agg_df['Aggregated_CAS_%'] = (agg_df['Matching_Cells'] / agg_df['Total_Cells']) * 100

    # --- 2. Pivoting Data to Wide Format ---
    print("Pivoting data to create summary table...")
    pivot_df = agg_df.pivot(
        index='Consensus_Cell_Type',
        columns='source_level',
        values=['Total_Cells', 'Matching_Cells', 'Aggregated_CAS_%']
    )

    # --- 3. Formatting the Output DataFrame ---
    def sort_key(level):
        if level == 'initial':
            return 0
        match = re.search(r'refinement_depth_(\d+)', level)
        return int(match.group(1)) if match else 999

    all_stages = sorted(df['source_level'].unique(), key=sort_key)
    
    final_columns = ['Consensus_Cell_Type']
    column_mapping = {
        'Total_Cells': 'Total_Cells',
        'Matching_Cells': 'Matching_Predictions',
        'Aggregated_CAS_%': 'CAS_Score_(%)'
    }
    
    pivot_df.columns = [f'{col[1]}_{column_mapping[col[0]]}' for col in pivot_df.columns]

    for stage in all_stages:
        for suffix in column_mapping.values():
            final_columns.append(f'{stage}_{suffix}')
            
    summary_df = pivot_df.reset_index().reindex(columns=final_columns)

    for col in summary_df.columns:
        if 'CAS_Score' in col:
            summary_df[col] = summary_df[col].map('{:.2f}'.format).replace('nan', '-')
        elif 'Cells' in col or 'Predictions' in col:
            summary_df[col] = summary_df[col].apply(lambda x: int(x) if pd.notna(x) else '-')

    summary_df.fillna('-', inplace=True)
    
    # --- 4. Save to CSV ---
    print(f"✅ Success! Saving cell type journey summary report to: {output_file}")
    summary_df.to_csv(output_file, index=False)

# ==============================================================================
# --- END: INTEGRATED REFINEMENT JOURNEY SUMMARY FUNCTION ---
# ==============================================================================

def _generate_greyed_out_umap_plot(adata, cas_df, threshold, cas_aggregation_method, output_path, title, legend_fontsize=8):
    """
    (Stage 2 Helper) Generates a UMAP plot highlighting low-confidence cells in grey.
    This function is used for the *initial* Stage 2 run to identify the first batch of failing cells.
    """
    print(f"--- Identifying cells from clusters with CAS < {threshold}% for initial plot ---")
    failing_clusters_df = cas_df[cas_df['Cluster_Annotation_Score_CAS (%)'] < threshold]

    if failing_clusters_df.empty:
        print("✅ No clusters found below the threshold. All cells will be colored by type.")
        failing_cell_indices = []
    else:
        if cas_aggregation_method == 'leiden':
            if 'Cluster_ID (Leiden)' not in failing_clusters_df.columns:
                print(f"[ERROR] Column 'Cluster_ID (Leiden)' not in CAS file for greyed-out plot. Skipping.")
                return
            failing_ids = failing_clusters_df['Cluster_ID (Leiden)'].astype(str).tolist()
            failing_cell_indices = adata.obs[adata.obs['leiden'].isin(failing_ids)].index
        elif cas_aggregation_method == 'consensus':
            if 'Consensus_Cell_Type' not in failing_clusters_df.columns:
                 print(f"[ERROR] Column 'Consensus_Cell_Type' not in CAS file for greyed-out plot. Skipping.")
                 return
            failing_ids = failing_clusters_df['Consensus_Cell_Type'].tolist()
            failing_cell_indices = adata.obs[adata.obs['ctpt_consensus_prediction'].isin(failing_ids)].index
        else:
            print(f"[ERROR] Invalid CAS aggregation method provided to greyed-out plot function: {cas_aggregation_method}")
            return
        print(f"       -> Found {len(failing_ids)} failing groups: {failing_ids}")
        print(f"       -> Total cells identified as low-confidence for plotting: {len(failing_cell_indices)}")

    # Create a temporary annotation column for plotting
    plot_annotation_col = 'plot_annotation_greyed'
    adata.obs[plot_annotation_col] = adata.obs['ctpt_consensus_prediction'].astype(str)
    
    if len(failing_cell_indices) > 0:
        low_conf_label = 'Low-Confidence (<{:.0f}%)'.format(threshold)
        adata.obs.loc[failing_cell_indices, plot_annotation_col] = low_conf_label
    
    adata.obs[plot_annotation_col] = adata.obs[plot_annotation_col].astype('category')

    # Create a custom color palette to ensure consistency and add grey
    original_cats = adata.obs['ctpt_consensus_prediction'].cat.categories.tolist()
    # Use a consistent, large palette
    palette_to_use = sc.pl.palettes.godsnot_102 if len(original_cats) > 28 else sc.pl.palettes.default_102
    color_map = {cat: color for cat, color in zip(original_cats, palette_to_use)}
    
    if len(failing_cell_indices) > 0:
        color_map[low_conf_label] = '#bbbbbb'  # Medium grey

    # Generate the plot in memory
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(
            adata,
            color=plot_annotation_col,
            palette=color_map,
            title=title,
            legend_loc='right margin',
            legend_fontsize=legend_fontsize,
            frameon=False,
            size=10,
            show=False,
            save=False
        )
    
    _bold_right_margin_legend(output_path)
    plt.close()

    # Clean up the temporary column from adata.obs
    del adata.obs[plot_annotation_col]

def _bold_right_margin_legend(fig_path):
    """(Stage 2) Finds legend in current figure, makes text bold, and saves."""
    fig = plt.gcf()
    for ax in fig.axes:
        if (leg := ax.get_legend()) is not None:
            for txt in leg.get_texts(): txt.set_fontweight('bold')
    fig.savefig(fig_path, dpi=plt.rcParams['savefig.dpi'], bbox_inches='tight')
def reformat_dotplot_data(fraction_df: pd.DataFrame, top_genes_df: pd.DataFrame, output_dir: str, output_prefix: str, groupby_key: str):
    """
    (Stage 2) Reformats dot plot fraction data to a gene-centric sparse table.
    
    IMPORTANT: This function preserves ALL gene-cell type pairs, even if a gene
    appears as a top marker for multiple cell types. No uniqueness filtering is applied.
    """
    print(f"[INFO] Reformatting dot plot data for '{groupby_key}' (preserving all gene-cell type pairs)...")
    
    cell_types = top_genes_df['group'].unique().tolist()
    output_rows = []
    
    # Iterate through top_genes_df in order, preserving duplicates
    for _, row in top_genes_df.iterrows():
        gene = row['names']
        group = row['group']
        
        # Get the fraction value for this gene in this group
        try:
            fraction = fraction_df.loc[group, gene]
        except KeyError:
            print(f"       [WARNING] Gene '{gene}' not found in fraction_df for group '{group}'. Setting to 0.")
            fraction = 0.0
        
        # Create a row with the gene, its source cell type, and the fraction
        # All other cell type columns are empty (sparse format)
        new_row_data = {'Gene': gene, 'Source_Cell_Type': group}
        for ct in cell_types:
            if ct == group:
                new_row_data[ct] = fraction
            else:
                new_row_data[ct] = ''
        
        output_rows.append(new_row_data)
    
    # Create DataFrame with Gene, Source_Cell_Type, then all cell type columns
    column_order = ['Gene', 'Source_Cell_Type'] + cell_types
    reformatted_df = pd.DataFrame(output_rows)[column_order]
    
    reformatted_csv_path = os.path.join(output_dir, f"{output_prefix}_dotplot_fractions_{groupby_key}_reformatted.csv")
    reformatted_df.to_csv(reformatted_csv_path, index=False)
    print(f"       -> Saved reformatted fraction data ({len(reformatted_df)} rows) to: {reformatted_csv_path}")
    
    # Also save a version that shows fractions across ALL cell types for each gene occurrence
    # This is useful for seeing how a marker gene expresses in non-target cell types
    print(f"[INFO] Generating full cross-cell-type fraction table...")
    full_fraction_rows = []
    
    for _, row in top_genes_df.iterrows():
        gene = row['names']
        source_group = row['group']
        
        full_row_data = {'Gene': gene, 'Source_Cell_Type': source_group}
        for ct in cell_types:
            try:
                full_row_data[ct] = fraction_df.loc[ct, gene]
            except KeyError:
                full_row_data[ct] = 0.0
        
        full_fraction_rows.append(full_row_data)
    
    full_fraction_df = pd.DataFrame(full_fraction_rows)[column_order]
    full_csv_path = os.path.join(output_dir, f"{output_prefix}_dotplot_fractions_{groupby_key}_full_matrix.csv")
    full_fraction_df.to_csv(full_csv_path, index=False)
    print(f"       -> Saved full fraction matrix ({len(full_fraction_df)} rows) to: {full_csv_path}")

def extract_fraction_data_and_calculate_mcs(adata: anndata.AnnData, output_dir: str, output_prefix: str, groupby_key: str, top_genes_df: pd.DataFrame, cli_args):
    """
    (Stage 2, Single-Sample) Calculates and saves expression fractions and the MCS.
    MODIFIED: Now extracts fractions for ALL genes in top_genes_df (with duplicates),
    not just unique genes.
    """
    print(f"[INFO] Calculating MCS and expression fractions for '{groupby_key}'...")
    
    if groupby_key not in adata.obs.columns:
        print(f"[ERROR] Grouping key '{groupby_key}' not found. Skipping MCS.")
        return None
    
    # Get ALL genes from top_genes_df (may contain duplicates)
    all_top_genes = top_genes_df['names'].tolist()
    
    # For fraction calculation, we need unique genes to avoid redundant computation
    unique_top_genes = list(dict.fromkeys(all_top_genes))  # Preserves order, removes duplicates
    
    print(f"       -> Total gene entries in top markers: {len(all_top_genes)}")
    print(f"       -> Unique genes for fraction calculation: {len(unique_top_genes)}")
    
    # Calculate fractions for unique genes
    data_df = sc.get.obs_df(adata, keys=[groupby_key] + unique_top_genes, use_raw=(adata.raw is not None))
    fraction_df = data_df.groupby(groupby_key).apply(lambda x: (x[unique_top_genes] > 0).mean())
    
    # Calculate MCS scores per cell type
    mcs_scores = {}
    for cell_type in top_genes_df['group'].unique():
        cell_type_genes = top_genes_df[top_genes_df['group'] == cell_type]['names'].tolist()
        # Get fractions for this cell type's marker genes
        fractions = [fraction_df.loc[cell_type, gene] for gene in cell_type_genes if gene in fraction_df.columns]
        if fractions:
            mcs_scores[cell_type] = np.mean(fractions)
        else:
            mcs_scores[cell_type] = 0.0
    
    # Save MCS scores
    mcs_df = pd.DataFrame.from_dict(mcs_scores, orient='index', columns=['MCS'])
    mcs_df.index.name = 'Cell_Type'
    mcs_df.to_csv(os.path.join(output_dir, f"{output_prefix}_marker_concordance_scores.csv"))
    print(f"       -> Saved MCS scores.")
    
    # Save full fraction data (unique genes only, for reference)
    fraction_df.to_csv(os.path.join(output_dir, f"{output_prefix}_dotplot_fractions_{groupby_key}.csv"))
    print(f"       -> Saved full fraction data (unique genes).")
    
    # Call the reformatting function which preserves all gene-cell type pairs
    reformat_dotplot_data(fraction_df, top_genes_df, output_dir, output_prefix, groupby_key)
    
    return mcs_df

def extract_fraction_data_for_dotplot(adata: anndata.AnnData, output_dir: str, output_prefix: str, groupby_key: str, top_genes_df: pd.DataFrame):
    """
    (Stage 2, Multi-Sample) Calculates and saves expression fractions for dotplot.
    
    MODIFIED: Now handles all gene-cell type pairs, preserving duplicates in the
    reformatted output.
    """
    print(f"[INFO] Calculating expression fractions for dotplot for '{groupby_key}'...")
    
    if groupby_key not in adata.obs.columns:
        print(f"[ERROR] Grouping key '{groupby_key}' not found in adata.obs. Skipping.")
        return
    
    # Get ALL genes from top_genes_df (may contain duplicates)
    all_top_genes = top_genes_df['names'].tolist()
    
    # For fraction calculation, we need unique genes
    unique_top_genes = list(dict.fromkeys(all_top_genes))  # Preserves order, removes duplicates
    
    print(f"       -> Total gene entries in top markers: {len(all_top_genes)}")
    print(f"       -> Unique genes for fraction calculation: {len(unique_top_genes)}")
    
    # Calculate fractions
    data_df = sc.get.obs_df(adata, keys=[groupby_key] + unique_top_genes, use_raw=(adata.raw is not None))
    fraction_df = data_df.groupby(groupby_key).apply(lambda x: (x[unique_top_genes] > 0).mean())
    
    # Save full fraction data (unique genes)
    output_csv_path = os.path.join(output_dir, f"{output_prefix}_dotplot_fractions_{groupby_key}.csv")
    fraction_df.to_csv(output_csv_path)
    print(f"       -> Saved full fraction data to: {output_csv_path}")
    
    # Call the reformatting function which preserves all gene-cell type pairs
    reformat_dotplot_data(fraction_df, top_genes_df, output_dir, output_prefix, groupby_key)

def run_stage_two_final_analysis(cli_args, optimal_params, output_dir, data_dir=None, adata_input=None):
    """
    (Stage 2) Executes the detailed single-sample analysis pipeline using
    parameters discovered in Stage 1. All outputs are saved to a subdirectory.
    Can either load data from `data_dir` or use a pre-loaded `adata_input`.
    """
    print("--- Initializing Stage 2: CAS-MCS Scoring Pipeline with Optimal Parameters ---")

    random.seed(cli_args.seed); np.random.seed(cli_args.seed); sc.settings.njobs = 1
    print(f"[INFO] Global random seed set to: {cli_args.seed}")

    sc.settings.verbosity = 3; sc.logging.print_header()
    sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False, dpi_save=cli_args.fig_dpi)
    sc.settings.figdir = output_dir
    print(f"[INFO] Scanpy version: {sc.__version__}")
    print(f"[INFO] Outputting to subdirectory: {os.path.abspath(output_dir)}")

    print("\n--- Step 1: Loading Data ---")
    if adata_input is not None:
        print("       -> Using provided AnnData object for analysis.")
        adata = adata_input.copy()
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
    elif data_dir is not None:
        adata = load_expression_data(data_dir)
        adata.layers["counts"] = adata.X.copy()
    else:
        raise ValueError("Must provide 'data_dir' or 'adata_input' to run_stage_two_final_analysis.")
    print(f"       -> Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

    # === NEW: Check for batch information and decide on integration ===
    is_integrated_mode = False
    integration_key = None
    
    if not (cli_args.no_integration or cli_args.integration_method == 'none'):
        # Check if 'sample' column already exists (from Stage 1 or multi_sample mode)
        if 'sample' in adata.obs.columns and adata.obs['sample'].nunique() >= 2:
            is_integrated_mode = True
            integration_key = 'sample'
            print(f"       -> Detected multi-batch data via 'sample' column ({adata.obs['sample'].nunique()} batches)")
        elif cli_args.batch_key is not None:
            existing_batch, has_batches = check_existing_batch_column(adata, batch_key=cli_args.batch_key)
            if has_batches:
                is_integrated_mode = True
                integration_key = existing_batch
                adata.obs['sample'] = adata.obs[integration_key].astype(str)
        else:
            # Try auto-detection
            existing_batch, has_batches = check_existing_batch_column(adata)
            if has_batches:
                is_integrated_mode = True
                integration_key = existing_batch
                adata.obs['sample'] = adata.obs[integration_key].astype(str)
            else:
                adata, detected_batches = detect_batch_from_barcodes(adata)
                if detected_batches:
                    is_integrated_mode = True
                    integration_key = 'barcode_batch'
                    adata.obs['sample'] = adata.obs[integration_key].astype(str)
    
    if is_integrated_mode:
        print(f"       -> Running Stage 2 in INTEGRATED mode (batch key: '{integration_key}')")
    else:
        print(f"       -> Running Stage 2 in SINGLE-SAMPLE mode")

    if cli_args.st_data_dir is not None:
        print(f"\n[INFO] Stage 2: Intersecting with Spatial genes...")
        adata_st_temp = load_expression_data(cli_args.st_data_dir)
        common_genes = list(set(adata.var_names) & set(adata_st_temp.var_names))
        common_genes.sort()
        adata = adata[:, common_genes].copy()
        # Update counts layer if it exists to match dimensions
        if "counts" in adata.layers:
             adata.layers["counts"] = adata.X.copy()
        print(f"       -> scRNA-seq subsetted to {adata.n_vars} common genes.")
        del adata_st_temp

    print("\n--- Step 2: Quality Control and Filtering ---")
    adata.var['mt'] = adata.var_names.str.contains(MITO_REGEX_PATTERN, regex=True)
    print(f"       -> Identified {adata.var['mt'].sum()} mitochondrial genes using robust regex.")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=False)
    plt.savefig(os.path.join(output_dir, f"{cli_args.final_run_prefix}_qc_plots_before_filtering.png")); plt.close()

    sc.pp.filter_cells(adata, min_genes=cli_args.min_genes)
    sc.pp.filter_cells(adata, max_genes=cli_args.max_genes)
    adata = adata[adata.obs.pct_counts_mt < cli_args.max_pct_mt, :]
    sc.pp.filter_genes(adata, min_cells=cli_args.min_cells)
    print(f"       -> Filtered dims: {adata.n_obs} cells, {adata.n_vars} genes")

    print("\n--- Step 3: Normalization, HVG, Scaling (using optimal params) ---")
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata); adata.raw = adata.copy()

    if all(p is not None for p in [cli_args.hvg_min_mean, cli_args.hvg_max_mean, cli_args.hvg_min_disp]):
        print("[INFO] Using two-step sequential HVG selection.")
        sc.pp.highly_variable_genes(
            adata, 
            min_mean=cli_args.hvg_min_mean, 
            max_mean=cli_args.hvg_max_mean, 
            min_disp=cli_args.hvg_min_disp,
            batch_key='sample' if is_integrated_mode else None  # === NEW: Add batch key ===
        )
        hvg_df = adata.var[adata.var.highly_variable].sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:optimal_params['n_hvg']]
        adata.var['highly_variable'] = False; adata.var.loc[top_genes, 'highly_variable'] = True
    else:
        print(f"[INFO] Using rank-based HVG selection with n_top_genes={optimal_params['n_hvg']}")
        if is_integrated_mode:
            sc.pp.highly_variable_genes(adata, n_top_genes=optimal_params['n_hvg'], flavor='seurat_v3', batch_key='sample')
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=optimal_params['n_hvg'], flavor='seurat_v3')

    sc.pl.highly_variable_genes(adata, save=f"_{cli_args.final_run_prefix}_hvg_plot.png", show=False); plt.close()
    adata = adata[:, adata.var.highly_variable]
    print(f"       -> Final selection: {adata.n_vars} highly variable genes for downstream analysis.")
    sc.pp.scale(adata, max_value=10)

    print("\n--- Step 4: Dimensionality Reduction and Clustering (using optimal params) ---")
    # --- BUG FIX START ---
    # Robustly cap the number of PCs by both cells and genes, crucial for refinement runs.
    n_pcs_to_compute = min(cli_args.n_pcs_compute, adata.n_obs - 1, adata.n_vars - 1)
    # --- BUG FIX END ---
    n_pcs_to_use = min(optimal_params['n_pcs'], n_pcs_to_compute)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs_to_compute, random_state=cli_args.seed)
    sc.pl.pca_variance_ratio(adata, log=True, n_pcs=n_pcs_to_compute, save=f"_{cli_args.final_run_prefix}_pca_variance.png", show=False); plt.close()

    # === NEW: Apply Harmony integration if in integrated mode ===
    embedding_to_use = 'X_pca'
    if is_integrated_mode:
        print("       -> Applying Harmony batch correction...")
        try:
            sc.external.pp.harmony_integrate(
                adata, 
                key='sample', 
                basis='X_pca', 
                adjusted_basis='X_pca_harmony', 
                random_state=cli_args.seed
            )
            embedding_to_use = 'X_pca_harmony'
            print("       -> ✓ Harmony integration complete. Using 'X_pca_harmony' for downstream.")
            
            # Save UMAP colored by batch/sample for QC
            sc.pp.neighbors(adata, n_neighbors=optimal_params['n_neighbors'], n_pcs=n_pcs_to_use, use_rep=embedding_to_use, random_state=cli_args.seed)
            sc.tl.umap(adata, random_state=cli_args.seed)
            sc.pl.umap(adata, color='sample', title='UMAP by Batch (after Harmony)', 
                      save=f"_{cli_args.final_run_prefix}_umap_batch.png", show=False, size=10)
            plt.close()
            
            # Reset neighbors to continue with normal flow (will be recalculated below)
            
        except ImportError:
            print("       -> [WARNING] harmonypy not installed. Skipping integration.")
            print("       -> Install with: pip install harmonypy")
        except Exception as e:
            print(f"       -> [WARNING] Harmony integration failed: {e}")
            print(f"       -> Proceeding without batch correction.")
    
    sc.pp.neighbors(adata, n_neighbors=optimal_params['n_neighbors'], n_pcs=n_pcs_to_use, use_rep=embedding_to_use, random_state=cli_args.seed)
    sc.tl.leiden(adata, resolution=optimal_params['resolution'], random_state=cli_args.seed)
    sc.tl.umap(adata, random_state=cli_args.seed)

    # --- MODIFICATION START: Safe Silhouette Calculation (Prevents Crash) ---
    silhouette_avg = 0.0
    n_clusters = adata.obs['leiden'].nunique()
    n_cells = adata.n_obs
    
    # Only calculate if we have valid clusters and not 1 cluster per cell (which causes ValueError)
    if 1 < n_clusters < n_cells:
        try:
            # Use the correct embedding (Harmony-corrected if available)
            silhouette_avg = silhouette_score(adata.obsm[embedding_to_use][:, :n_pcs_to_use], adata.obs['leiden'])
            print(f"       -> Average Silhouette Score for Leiden clustering: {silhouette_avg:.3f}")
        except Exception as e:
            print(f"       -> [WARNING] Silhouette calculation error: {e}")
            silhouette_avg = 0.0
    else:
        print(f"       -> [WARNING] Silhouette skipped. Too many clusters ({n_clusters}) for n_cells ({n_cells}) or only 1 cluster.")
        silhouette_avg = 0.0
    # --- MODIFICATION END ---

    sc.pl.umap(adata, color='leiden', legend_fontweight='bold', legend_loc='on data', title=f'Leiden Clusters ({adata.obs["leiden"].nunique()} clusters)\nSilhouette: {silhouette_avg:.3f}', palette=sc.pl.palettes.godsnot_102, save=f"_{cli_args.final_run_prefix}_umap_leiden.png", show=False, size=10); plt.close()

    print("\n--- Step 5: CellTypist Annotation and CAS Calculation ---")
    model_ct = models.Model.load(cli_args.model_path)
    print("[INFO] Annotating cells using the full log-normalized transcriptome (from adata.raw)...")
    predictions = celltypist.annotate(adata.raw.to_adata(), model=model_ct, majority_voting=False)
    adata.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels'].astype('category')
    if 'conf_score' in predictions.predicted_labels.columns: adata.obs['ctpt_confidence'] = predictions.predicted_labels['conf_score']

    sc.pl.umap(adata, color='ctpt_individual_prediction', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Per-Cell CellTypist Annotation ({adata.obs["ctpt_individual_prediction"].nunique()} types)', show=False, size=10)
    _bold_right_margin_legend(os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_per_cell_celltypist.png")); plt.close()

    cluster2label = adata.obs.groupby('leiden')['ctpt_individual_prediction'].agg(lambda x: x.value_counts().idxmax()).to_dict()
    adata.obs['ctpt_consensus_prediction'] = adata.obs['leiden'].map(cluster2label).astype('category')
    sc.pl.umap(adata, color='ctpt_consensus_prediction', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Cluster-Consensus CellTypist Annotation ({adata.obs["ctpt_consensus_prediction"].nunique()} types)', show=False, size=10)
    _bold_right_margin_legend(os.path.join(output_dir, f"{cli_args.final_run_prefix}_cluster_celltypist_umap.png")); plt.close()
    
    leiden_purity_results = []
    leiden_groups = adata.obs.groupby('leiden')
    for leiden_id, group in leiden_groups:
        consensus_name = group['ctpt_consensus_prediction'].iloc[0]
        # MODIFICATION START: Standardize column name
        leiden_purity_results.append({
            "Cluster_ID (Leiden)": leiden_id, "Consensus_Cell_Type": consensus_name, "Total_Cells_in_Group": len(group),
            "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == consensus_name).sum(),
            "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == consensus_name).sum() / len(group) if len(group) > 0 else 0
        })
        # MODIFICATION END
    cas_leiden_df = pd.DataFrame(leiden_purity_results).sort_values(by="Cluster_Annotation_Score_CAS (%)", ascending=False)
    cas_leiden_output_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_leiden_cluster_annotation_scores.csv")
    cas_leiden_df.to_csv(cas_leiden_output_path, index=False)
    print(f"       -> Saved Leiden-based CAS (technical purity) scores to: {cas_leiden_output_path}")

    consensus_purity_results = []
    for name, group in adata.obs.groupby('ctpt_consensus_prediction'):
        # MODIFICATION START: Standardize column name
        consensus_purity_results.append({
            "Consensus_Cell_Type": name, "Total_Cells_in_Group": len(group),
            "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == name).sum(),
            "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == name).sum() / len(group) if len(group) > 0 else 0
        })
        # MODIFICATION END
    cas_consensus_df = pd.DataFrame(consensus_purity_results).sort_values(by="Cluster_Annotation_Score_CAS (%)", ascending=False)
    cas_consensus_output_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_consensus_group_annotation_scores.csv")
    cas_consensus_df.to_csv(cas_consensus_output_path, index=False)
    print(f"       -> Saved Consensus-based CAS (final label purity) scores to: {cas_consensus_output_path}")

    if cli_args.cas_aggregation_method == 'leiden':
        cas_df_for_refinement, cas_path_for_refinement = cas_leiden_df, cas_leiden_output_path
        print("[INFO] Using Leiden-based CAS report for refinement thresholding.")
    else: # 'consensus'
        cas_df_for_refinement, cas_path_for_refinement = cas_consensus_df, cas_consensus_output_path
        print("[INFO] Using Consensus-based CAS report for refinement thresholding.")
    
    if cli_args.cas_refine_threshold is not None:
        print("\n--- Generating verification UMAP with low-confidence cells highlighted ---")
        greyed_umap_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_low_confidence_greyed.png")
        _generate_greyed_out_umap_plot(adata=adata, cas_df=cas_df_for_refinement, threshold=cli_args.cas_refine_threshold, cas_aggregation_method=cli_args.cas_aggregation_method, output_path=greyed_umap_path, title=f'Consensus Annotation (Failing Cells <{cli_args.cas_refine_threshold}% CAS in Grey)', legend_fontsize=8)
        print(f"       -> Saved greyed-out UMAP to: {greyed_umap_path}")

    print("\n--- Step 6: Marker Gene Analysis and MCS Calculation ---")
    marker_groupby_key = 'ctpt_consensus_prediction'; top_genes_df, mcs_df = None, None
    if marker_groupby_key in adata.obs.columns:
        adata.obs[marker_groupby_key] = adata.obs[marker_groupby_key].cat.remove_unused_categories()
        print(f"       -> Cleaned '{marker_groupby_key}': {adata.obs[marker_groupby_key].nunique()} active categories")
    valid_labels = adata.obs[marker_groupby_key].value_counts()[lambda x: x > 1].index.tolist()
    if len(valid_labels) < 2: print(f"[WARNING] Skipping marker gene analysis: Fewer than 2 consensus groups with >1 cell.")
    else:
        sc.tl.rank_genes_groups(adata, marker_groupby_key, groups=valid_labels, method='wilcoxon', use_raw=True, key_added=f"wilcoxon_{marker_groupby_key}")
        marker_df = sc.get.rank_genes_groups_df(adata, key=f"wilcoxon_{marker_groupby_key}", group=None)

        is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
        filtered_rows = [sub[~sub['names'].map(is_mito)].head(cli_args.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
        top_genes_df = pd.concat(filtered_rows, ignore_index=True)

        # [MODIFICATION START: Wrap DotPlot in Try/Except]
        try:
            print(f"       -> Attempting to generate marker gene dotplot...")
            with plt.rc_context({'font.size': 18, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
                # Safety check for categories existence
                valid_cats = set(adata.obs[marker_groupby_key].unique())
                cats_to_plot = [c for c in top_genes_df['group'].unique().tolist() if c in valid_cats]
                
                if cats_to_plot:
                    sc.pl.dotplot(adata, 
                                  var_names=top_genes_df.groupby('group')['names'].apply(list).to_dict(), 
                                  groupby=marker_groupby_key, 
                                  categories_order=cats_to_plot, 
                                  use_raw=True, 
                                  save=f"_{cli_args.final_run_prefix}_markers_celltypist_dotplot.png", 
                                  show=False)
                    plt.close()
                else:
                    print("       -> [SKIP] No valid categories overlap for dotplot.")
        except Exception as e:
            print(f"       -> [WARNING] Dotplot generation failed. Error: {e}")
            print("       -> Pipeline continuing without this plot...")
        # [MODIFICATION END]

        mcs_df = extract_fraction_data_and_calculate_mcs(adata, output_dir, cli_args.final_run_prefix, marker_groupby_key, top_genes_df, cli_args)
        if mcs_df is not None and top_genes_df is not None:
            top_genes_agg = top_genes_df.groupby('group')['names'].apply(', '.join).reset_index().rename(columns={'names': f'Top_{cli_args.n_top_genes}_Markers', 'group': 'Cell_Type'})
            pd.merge(mcs_df, top_genes_agg, on='Cell_Type')[['Cell_Type', 'MCS', f'Top_{cli_args.n_top_genes}_Markers']].to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_mcs_and_top_markers.csv"), index=False); print(f"       -> Saved combined MCS and Top Markers.")

    # === NEW SECTION: Step 6b - Marker Prior Score Calculation ===
    print("\n--- Step 6b: Marker Prior Score (MPS) Calculation ---")
    
    mean_mps_s2 = 0.0
    mps_details_s2 = {}
    
    if MARKER_PRIOR_DICT:
        try:
            mean_mps_s2, mps_details_s2 = calculate_marker_prior_score(
                adata=adata,
                groupby_key='ctpt_consensus_prediction',
                prior_dict=MARKER_PRIOR_DICT,
                n_top_degs=cli_args.n_degs_for_mps if hasattr(cli_args, 'n_degs_for_mps') else 50,
                use_raw=True
            )
            
            # Save MPS details to CSV
            mps_output_rows = []
            for cell_type, details in mps_details_s2.items():
                mps_output_rows.append({
                    'Cell_Type': cell_type,
                    'Matched_Prior': details.get('matched_prior', 'None'),
                    'N_Canonical_Markers': details.get('n_canonical_markers', 0),
                    'N_Markers_In_Data': details.get('n_markers_in_data', 0),
                    'Recall_Score_Pct': details.get('recall_score', 0),
                    'Expression_Score_Pct': details.get('expression_score', 0),
                    'MPS': details.get('mps', 0),
                    'Matched_Markers': '; '.join(details.get('matched_markers', [])),
                    'Missing_Markers': '; '.join(details.get('missing_markers', [])[:10])  # Limit to 10
                })
            
            mps_df = pd.DataFrame(mps_output_rows)
            mps_csv_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_marker_prior_scores.csv")
            mps_df.to_csv(mps_csv_path, index=False)
            print(f"       -> Saved MPS details to: {mps_csv_path}")
            
        except Exception as e:
            print(f"[WARNING] MPS calculation failed in Stage 2: {e}")
    else:
        print("       -> MPS disabled (no marker prior database loaded).")

    print("\n--- Step 7: Optional Manual-Style Annotation & Scoring ---")
    if cli_args.cellmarker_db and os.path.exists(cli_args.cellmarker_db):
        try:
            print(f"       -> Annotating using marker DB: {cli_args.cellmarker_db}")
            
            sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', use_raw=True, key_added='wilcoxon_leiden')
            leiden_markers_df = sc.get.rank_genes_groups_df(adata, key='wilcoxon_leiden', group=None)

            header = pd.read_csv(cli_args.cellmarker_db, nrows=0).columns.tolist()
            type_col, gene_col = ('cell_name', 'Symbol') if 'cell_name' in header and 'Symbol' in header else (('Cell Type', 'Cell Marker') if 'Cell Type' in header and 'Cell Marker' in header else (None, None))
            if not type_col: raise ValueError("Marker DB must contain ('cell_name', 'Symbol') or ('Cell Type', 'Cell Marker') columns.")
            print(f"       -> Auto-detected format: TYPE='{type_col}', GENE='{gene_col}'")

            db_df = pd.read_csv(cli_args.cellmarker_db)
            db_markers_dict = defaultdict(set)
            for _, row in db_df.iterrows():
                if pd.notna(row.get(gene_col)) and pd.notna(row.get(type_col)):
                    db_markers_dict[row[type_col]].update({m.strip().upper() for m in str(row[gene_col]).split(',')})
            print(f"       -> Aggregated markers for {len(db_markers_dict)} unique cell types.")

            cluster_annotations = {}
            for cluster in adata.obs['leiden'].cat.categories:
                cluster_genes = set(leiden_markers_df[leiden_markers_df['group'] == cluster].head(cli_args.n_top_genes)['names'].str.upper())
                scores = {cell_type: len(cluster_genes.intersection(db_genes)) / (len(cluster_genes.union(db_genes)) or 1) for cell_type, db_genes in db_markers_dict.items()}
                best_cell_type = max(scores, key=scores.get) if scores else None
                cluster_annotations[cluster] = best_cell_type if best_cell_type and scores[best_cell_type] > 0 else f"Unknown_{cluster}"

            adata.obs['manual_annotation'] = adata.obs['leiden'].map(cluster_annotations).astype('category')
            pd.DataFrame.from_dict(cluster_annotations, orient='index', columns=['AssignedType']).to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_leiden_to_manual_annotation.csv"))
            sc.pl.umap(adata, color='manual_annotation', title='Manual Cluster Annotation', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, size=10, show=False)
            _bold_right_margin_legend(os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_manual_annotation.png")); plt.close()

            print("       -> Calculating Marker Capture Score for manual annotation...")
            score_results = []
            leiden_degs_structured = adata.uns['wilcoxon_leiden']['names']
            for cluster_id, assigned_label in cluster_annotations.items():
                if pd.isna(assigned_label) or assigned_label.startswith("Unknown"): continue
                reference_genes = db_markers_dict.get(assigned_label, set())
                if not reference_genes: continue
                
                cluster_degs_for_capture = {g.upper() for g in leiden_degs_structured[cluster_id][:cli_args.n_degs_for_capture]}
                captured_genes = cluster_degs_for_capture.intersection(reference_genes)
                score = (len(captured_genes) / len(reference_genes)) * 100
                
                score_results.append({
                    "Cluster_ID": cluster_id, "Assigned_Cell_Type": assigned_label, "Marker_Capture_Score (%)": score,
                    "Captured_Genes_Count": len(captured_genes), "Total_Reference_Genes": len(reference_genes),
                    "Captured_Genes_List": ", ".join(sorted(list(captured_genes)))
                })
            
            if score_results:
                capture_df = pd.DataFrame(score_results).sort_values(by="Marker_Capture_Score (%)", ascending=False)
                capture_df.to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_manual_annotation_marker_capture_scores.csv"), index=False)
                print(f"       -> Saved Marker Capture Scores.")

        except Exception as e:
            print(f"[ERROR] Manual annotation/scoring failed. Reason: {e}")
    else:
        print("[INFO] Cell marker DB not provided or not found. Skipping manual-style annotation.")

    print("\n--- Step 8: Exporting All Results ---")
    cols_to_save = [col for col in ['leiden', 'ctpt_individual_prediction', 'ctpt_confidence', 'ctpt_consensus_prediction', 'manual_annotation'] if col in adata.obs.columns]
    adata.obs[cols_to_save].to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_all_annotations.csv"))
    print(f"       -> All cell annotations saved."); adata.write(os.path.join(output_dir, f"{cli_args.final_run_prefix}_final_processed.h5ad")); print(f"       -> Final AnnData object saved.")

    # === NEW: Save batch/integration information if applicable ===
    if is_integrated_mode:
        print("\n--- Batch Integration Summary ---")
        batch_counts = adata.obs['sample'].value_counts()
        print(f"       -> Integration method: Harmony")
        print(f"       -> Number of batches: {len(batch_counts)}")
        for batch, count in batch_counts.items():
            print(f"          - {batch}: {count} cells ({count/len(adata)*100:.1f}%)")
        
        # Save batch composition to CSV
        batch_df = pd.DataFrame({
            'Batch': batch_counts.index,
            'Cell_Count': batch_counts.values,
            'Percentage': (batch_counts.values / len(adata) * 100).round(2)
        })
        batch_csv_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_batch_composition.csv")
        batch_df.to_csv(batch_csv_path, index=False)
        print(f"       -> Saved batch composition to: {batch_csv_path}")

    print("\n--- Step 9: Verifying Metrics Against Optimization Run ---")
    total_matching = (adata.obs['ctpt_individual_prediction'].astype(str) == adata.obs['ctpt_consensus_prediction'].astype(str)).sum()
    weighted_cas = (total_matching / len(adata.obs)) * 100 if len(adata.obs) > 0 else 0.0
    
    simple_cas = 0.0
    if cli_args.cas_aggregation_method == 'leiden':
        simple_cas_groups = [(g['ctpt_individual_prediction'].astype(str) == g['ctpt_consensus_prediction'].astype(str).iloc[0]).mean() * 100 for _, g in adata.obs.groupby('leiden') if not g.empty]
        simple_cas = np.mean(simple_cas_groups) if simple_cas_groups else 0.0
    elif cli_args.cas_aggregation_method == 'consensus':
        cas_per_consensus_group = [(g['ctpt_individual_prediction'].astype(str) == g['ctpt_consensus_prediction'].astype(str).iloc[0]).mean() * 100 for _, g in adata.obs.groupby('ctpt_consensus_prediction') if not g.empty]
        simple_cas = np.mean(cas_per_consensus_group) if cas_per_consensus_group else 0.0

    mean_mcs = mcs_df['MCS'].mean() if mcs_df is not None and not mcs_df.empty else 0.0
    target_map = {'simple_cas': "Simple Mean CAS", 'weighted_cas': "Weighted Mean CAS", 'mcs': "Mean MCS", 'balanced': "Balanced Score"}
    print("\n" + "="*50 + f"\n--- Final Verification Summary (Single-Sample) ---\nOptimization Target from Stage 1: {target_map.get(cli_args.target, 'N/A')}\nRandom Seed Used: {cli_args.seed}\n\n--- Optimal Parameters Used ---\nBest n_hvg: {optimal_params['n_hvg']}\nBest n_pcs: {n_pcs_to_use}\nBest n_neighbors: {optimal_params['n_neighbors']}\nBest resolution: {optimal_params['resolution']:.3f}\n")
    if cli_args.target == 'simple_cas': print(f"Highest_simple_mean_cas_pct: {simple_cas:.2f}\nCorresponding_weighted_mean_cas_pct: {weighted_cas:.2f}\nCorresponding_mean_mcs_pct: {mean_mcs * 100:.2f}\nCorresponding_silhouette_score: {silhouette_avg:.3f}\n")
    elif cli_args.target == 'weighted_cas': print(f"Highest_weighted_mean_cas_pct: {weighted_cas:.2f}\nCorresponding_simple_mean_cas_pct: {simple_cas:.2f}\nCorresponding_mean_mcs_pct: {mean_mcs * 100:.2f}\nCorresponding_silhouette_score: {silhouette_avg:.3f}\n")
    elif cli_args.target in ['mcs', 'balanced']: print(f"Target Score ({cli_args.target}): {mean_mcs * 100 if cli_args.target=='mcs' else 'N/A'}\nCorresponding_weighted_mean_cas_pct: {weighted_cas:.2f}\nCorresponding_simple_mean_cas_pct: {simple_cas:.2f}\nCorresponding_mean_mcs_pct: {mean_mcs * 100:.2f}\nCorresponding_silhouette_score: {silhouette_avg:.3f}\n")
    print(f"Final_n_individual_labels: {adata.obs['ctpt_individual_prediction'].nunique()}\nFinal_n_consensus_labels: {adata.obs['ctpt_consensus_prediction'].nunique()}\n" + "="*50)
    
    return adata, cas_path_for_refinement

def run_stage_two_final_analysis_multi_sample(cli_args, optimal_params, output_dir, wt_path=None, treated_path=None, adata_input=None):
    """
    (Stage 2) Executes the detailed two-sample integration analysis pipeline using
    parameters discovered in Stage 1.
    """
    print("--- Initializing Stage 2: Two-Sample Integration Pipeline with Optimal Parameters ---")

    random.seed(cli_args.seed); np.random.seed(cli_args.seed); sc.settings.njobs = 1
    print(f"[INFO] Global random seed set to: {cli_args.seed} for reproducibility.")

    CONDITION_OF_INTEREST, REFERENCE_CONDITION = 'Treated', 'WT'
    FINAL_ANNOTATION_COLUMN = 'ctpt_consensus_prediction'

    sc.settings.verbosity = 3; sc.logging.print_header()
    sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False, dpi_save=cli_args.fig_dpi)
    sc.settings.figdir = output_dir
    print(f"[INFO] Scanpy version: {sc.__version__}\n[INFO] Outputting to subdirectory: {os.path.abspath(output_dir)}")

    if adata_input is not None:
        print("\n--- Step 1 & 2: Using Provided AnnData for Analysis ---")
        adata = adata_input.copy()
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        if 'sample' not in adata.obs.columns:
            print("\n   [INFO] 'sample' column not found in metadata, attempting to detect from barcodes...")
            adata, batch_detected = detect_batch_from_barcodes(adata)
            if batch_detected:
                print(f"   [INFO] Successfully detected sample information from barcodes")
            else:
                print(f"   [WARNING] Could not detect sample information, treating as single sample")
        else:
            print(f"   [INFO] Using existing 'sample' column with {adata.obs['sample'].nunique()} unique values")
    elif wt_path is not None and treated_path is not None:
        print("\n--- Step 1 & 2: Loading and Concatenating Datasets ---")
        adatas = {
            'WT': load_expression_data(wt_path), 
            'Treated': load_expression_data(treated_path)
        }
        for sid, adata_sample in adatas.items():
            adata_sample.obs['sample'] = sid
        adata = anndata.AnnData.concatenate(*adatas.values(), batch_key='sample', batch_categories=list(adatas.keys()))
    else:
        raise ValueError("Must provide ('wt_path', 'treated_path') or 'adata_input' to run_stage_two_final_analysis_multi_sample.")
    
    if cli_args.st_data_dir is not None:
        print(f"\n[INFO] Stage 2 (Multi-Sample): Intersecting with Spatial genes...")
        adata_st_temp = load_expression_data(cli_args.st_data_dir)
        common_genes = list(set(adata.var_names) & set(adata_st_temp.var_names))
        common_genes.sort()
        adata = adata[:, common_genes].copy()
        if "counts" in adata.layers:
             adata.layers["counts"] = adata.X.copy()
        print(f"       -> scRNA-seq subsetted to {adata.n_vars} common genes.")
        del adata_st_temp

    print("\n--- Step 3: Quality Control ---")
    adata.var['mt'] = adata.var_names.str.contains(MITO_REGEX_PATTERN, regex=True)
    print(f"       -> Identified {adata.var['mt'].sum()} mitochondrial genes using robust regex.")
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True, percent_top=None, log1p=False)

    sc.pp.filter_cells(adata, min_genes=cli_args.min_genes)
    sc.pp.filter_cells(adata, max_genes=cli_args.max_genes)
    adata = adata[adata.obs.pct_counts_mt < cli_args.max_pct_mt, :]
    sc.pp.filter_genes(adata, min_cells=cli_args.min_cells)
    print(f"       -> Filtered dims: {adata.n_obs} cells, {adata.n_vars} genes")

    print("\n--- Step 4: Normalization, HVG selection, Scaling ---")
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata); adata.raw = adata.copy()

    if all(p is not None for p in [cli_args.hvg_min_mean, cli_args.hvg_max_mean, cli_args.hvg_min_disp]):
        print("[INFO] Using two-step sequential HVG selection.")
        sc.pp.highly_variable_genes(adata, min_mean=cli_args.hvg_min_mean, max_mean=cli_args.hvg_max_mean, min_disp=cli_args.hvg_min_disp, batch_key='sample')
        hvg_df = adata.var[adata.var.highly_variable].sort_values('dispersions_norm', ascending=False)
        top_genes = hvg_df.index[:optimal_params['n_hvg']]
        adata.var['highly_variable'] = False; adata.var.loc[top_genes, 'highly_variable'] = True
    else:
        print(f"[INFO] Using rank-based HVG selection with n_top_genes={optimal_params['n_hvg']}")
        sc.pp.highly_variable_genes(adata, n_top_genes=optimal_params['n_hvg'], flavor='seurat_v3', batch_key='sample')

    sc.pl.highly_variable_genes(adata, save=f"_{cli_args.final_run_prefix}_hvg_plot.png", show=False); plt.close()
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"       -> Final selection: {adata.n_vars} highly variable genes for downstream analysis.")
    sc.pp.scale(adata, max_value=10)

    print("\n--- Step 5: PCA and Batch Correction with Harmony ---")
    # --- BUG FIX START ---
    # Robustly cap the number of PCs by both cells and genes, crucial for refinement runs.
    n_pcs_to_compute = min(cli_args.n_pcs_compute, adata.n_obs - 1, adata.n_vars - 1)
    # --- BUG FIX END ---
    n_pcs_to_use = min(optimal_params['n_pcs'], n_pcs_to_compute)
    print(f"[INFO] Computing {n_pcs_to_compute} PCs, using top {n_pcs_to_use} for downstream.")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs_to_compute, random_state=cli_args.seed)

    try:
        import harmonypy as hm
        print("harmonypy is installed. Performing batch correction.")
        sc.external.pp.harmony_integrate(adata, key='sample', basis='X_pca', adjusted_basis='X_pca_harmony', random_state=cli_args.seed)
        pca_rep_key = 'X_pca_harmony'
    except ImportError:
        print("[WARNING] harmonypy not found. Skipping Harmony integration."); pca_rep_key = 'X_pca'

    print("\n--- Step 6: Neighborhood, Clustering, and UMAP on Integrated Data ---")
    sc.pp.neighbors(adata, n_neighbors=optimal_params['n_neighbors'], n_pcs=n_pcs_to_use, use_rep=pca_rep_key, random_state=cli_args.seed)
    sc.tl.leiden(adata, resolution=optimal_params['resolution'], random_state=cli_args.seed)
    sc.tl.umap(adata, random_state=cli_args.seed)
    sc.pl.umap(adata, color='sample', title='UMAP by Sample', save=f"_{cli_args.final_run_prefix}_umap_sample.png", show=False, size=10); plt.close()

    # --- MODIFICATION START: Safe Silhouette Calculation (Prevents Crash) ---
    silhouette_avg = 0.0
    n_clusters = adata.obs['leiden'].nunique()
    n_cells = adata.n_obs

    # Only calculate if we have valid clusters and not 1 cluster per cell
    if 1 < n_clusters < n_cells:
        try:
            silhouette_avg = silhouette_score(adata.obsm[pca_rep_key][:, :n_pcs_to_use], adata.obs['leiden'])
            print(f"       -> Average Silhouette Score for Leiden clustering (on '{pca_rep_key}'): {silhouette_avg:.3f}")
        except Exception as e:
             print(f"       -> [WARNING] Silhouette calculation error: {e}")
             silhouette_avg = 0.0
    else:
        print(f"       -> [WARNING] Silhouette skipped. Too many clusters ({n_clusters}) for n_cells ({n_cells}) or only 1 cluster.")
        silhouette_avg = 0.0
    # --- MODIFICATION END ---

    sc.pl.umap(adata, color='leiden', legend_loc='on data', legend_fontweight='bold', title=f'Leiden Clusters (res={optimal_params["resolution"]})\nSilhouette: {silhouette_avg:.3f}', palette=sc.pl.palettes.godsnot_102, save=f"_{cli_args.final_run_prefix}_umap_leiden.png", show=False, size=10); plt.close()


    print("\n--- Step 7: Cell Type Annotation with CellTypist ---")
    top_genes_df = None
    cas_path_for_refinement = None # Initialize path
    
    if cli_args.model_path and os.path.exists(cli_args.model_path):
        model_ct = models.Model.load(cli_args.model_path)
        print("[INFO] Annotating cells using the full log-normalized transcriptome (from adata.raw)...")
        predictions = celltypist.annotate(adata.raw.to_adata(), model=model_ct, majority_voting=False)
        adata.obs['ctpt_individual_prediction'] = predictions.predicted_labels['predicted_labels']
        
        # START: ADDED PLOT FOR PER-CELL ANNOTATION
        sc.pl.umap(adata, color='ctpt_individual_prediction', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, title=f'Per-Cell CellTypist Annotation ({adata.obs["ctpt_individual_prediction"].nunique()} types)', show=False, size=10)
        _bold_right_margin_legend(os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_per_cell_celltypist.png")); plt.close()
        # END: ADDED PLOT

        adata.obs[FINAL_ANNOTATION_COLUMN] = adata.obs.groupby('leiden')['ctpt_individual_prediction'].transform(lambda x: x.value_counts().idxmax()).astype('category')
        adata.obs[FINAL_ANNOTATION_COLUMN] = adata.obs[FINAL_ANNOTATION_COLUMN].cat.remove_unused_categories()
        print(f"       -> Cleaned '{FINAL_ANNOTATION_COLUMN}': {adata.obs[FINAL_ANNOTATION_COLUMN].nunique()} active categories")
        leiden_purity_results = []
        leiden_groups = adata.obs.groupby('leiden')
        for leiden_id, group in leiden_groups:
            consensus_name = group[FINAL_ANNOTATION_COLUMN].iloc[0]
            # MODIFICATION START: Standardize column name
            leiden_purity_results.append({
                "Cluster_ID (Leiden)": leiden_id, "Consensus_Cell_Type": consensus_name, "Total_Cells_in_Group": len(group),
                "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == consensus_name).sum(),
                "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == consensus_name).sum() / len(group) if len(group) > 0 else 0
            })
            # MODIFICATION END
        cas_leiden_df = pd.DataFrame(leiden_purity_results).sort_values(by="Cluster_Annotation_Score_CAS (%)", ascending=False)
        cas_leiden_output_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_leiden_cluster_annotation_scores.csv")
        cas_leiden_df.to_csv(cas_leiden_output_path, index=False)
        print(f"       -> Saved Leiden-based CAS (technical purity) scores to: {cas_leiden_output_path}")

        consensus_purity_results = []
        for name, group in adata.obs.groupby(FINAL_ANNOTATION_COLUMN):
            # MODIFICATION START: Standardize column name
            consensus_purity_results.append({
                "Consensus_Cell_Type": name, "Total_Cells_in_Group": len(group),
                "Matching_Individual_Predictions": (group['ctpt_individual_prediction'] == name).sum(),
                "Cluster_Annotation_Score_CAS (%)": 100 * (group['ctpt_individual_prediction'] == name).sum() / len(group) if len(group) > 0 else 0
            })
            # MODIFICATION END
        cas_consensus_df = pd.DataFrame(consensus_purity_results).sort_values(by="Cluster_Annotation_Score_CAS (%)", ascending=False)
        cas_consensus_output_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_consensus_group_annotation_scores.csv")
        cas_consensus_df.to_csv(cas_consensus_output_path, index=False)
        print(f"       -> Saved Consensus-based CAS (final label purity) scores to: {cas_consensus_output_path}")

        if cli_args.cas_aggregation_method == 'leiden':
            cas_df_for_refinement, cas_path_for_refinement = cas_leiden_df, cas_leiden_output_path
            print("[INFO] Using Leiden-based CAS report for refinement thresholding.")
        else: # 'consensus'
            cas_df_for_refinement, cas_path_for_refinement = cas_consensus_df, cas_consensus_output_path
            print("[INFO] Using Consensus-based CAS report for refinement thresholding.")

        sc.pl.umap(adata, color=FINAL_ANNOTATION_COLUMN, title='Cluster-Consensus CellTypist Annotation', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, size=10, show=False)
        fig_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_cluster_celltypist_umap.png"); _bold_right_margin_legend(fig_path); plt.close()

        if cli_args.cas_refine_threshold is not None:
            print("\n--- Generating verification UMAP with low-confidence cells highlighted ---")
            greyed_umap_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_low_confidence_greyed.png")
            _generate_greyed_out_umap_plot(adata=adata, cas_df=cas_df_for_refinement, threshold=cli_args.cas_refine_threshold, cas_aggregation_method=cli_args.cas_aggregation_method, output_path=greyed_umap_path, title=f'Consensus Annotation (Failing Cells <{cli_args.cas_refine_threshold}% CAS in Grey)', legend_fontsize=8)
            print(f"       -> Saved greyed-out UMAP to: {greyed_umap_path}")

        marker_key = f"wilcoxon_{FINAL_ANNOTATION_COLUMN}"

        # [NEW SAFETY CHECK START]
        # Ensure we have at least 2 groups with data before ranking genes
        unique_groups = adata.obs[FINAL_ANNOTATION_COLUMN].dropna().unique()
        if len(unique_groups) < 2:
            print(f"       -> [SKIP] Skipping marker analysis/dotplot. Only {len(unique_groups)} cell type(s) present in this subset.")
        else:
            # [EXISTING CODE MOVED INSIDE ELSE BLOCK]
            sc.tl.rank_genes_groups(adata, FINAL_ANNOTATION_COLUMN, method='wilcoxon', use_raw=True, key_added=marker_key)
            marker_df = sc.get.rank_genes_groups_df(adata, key=marker_key, group=None)
            is_mito = lambda g: bool(re.match(MITO_REGEX_PATTERN, str(g)))
            filtered_rows = [sub[~sub['names'].map(is_mito)].head(cli_args.n_top_genes) for _, sub in marker_df.groupby('group', sort=False)]
            top_genes_df = pd.concat(filtered_rows, ignore_index=True)

            # [MODIFICATION START: Ultra-Safe Try/Except Block]
            try:
                print(f"       -> Attempting to generate marker gene dotplot...")
                with plt.rc_context({'font.size': 18, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
                    genes_to_plot = top_genes_df.groupby('group')['names'].apply(list).to_dict()
                    
                    # Explicitly verify categories exist in .obs before plotting
                    valid_cats_in_obs = set(adata.obs[FINAL_ANNOTATION_COLUMN].unique())
                    safe_categories_order = [cat for cat in list(genes_to_plot.keys()) if cat in valid_cats_in_obs]
                    
                    if safe_categories_order:
                        sc.pl.dotplot(adata, var_names=genes_to_plot, groupby=FINAL_ANNOTATION_COLUMN, 
                                      categories_order=safe_categories_order, 
                                      use_raw=True, save=f"_{cli_args.final_run_prefix}_markers_celltypist_dotplot.png", show=False)
                        plt.close()
                    else:
                         print("       -> [SKIP] No valid categories found for dotplot ordering.")
            except Exception as e:
                print(f"       -> [WARNING] Dotplot generation failed. Error: {e}")
                print("       -> Pipeline continuing without this plot...")
            # [MODIFICATION END]
            
            extract_fraction_data_for_dotplot(adata, output_dir, cli_args.final_run_prefix, FINAL_ANNOTATION_COLUMN, top_genes_df)
        # [NEW SAFETY CHECK END]
    else:
        print("[INFO] CellTypist not run. Using Leiden clusters for downstream analysis.")
        adata.obs[FINAL_ANNOTATION_COLUMN] = adata.obs['leiden'].astype('category')

    print("\n--- Step 8: Find Marker Genes for raw Leiden clusters ---")
    if 'leiden' in adata.obs.columns:
        adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
        print(f"       -> Cleaned 'leiden': {adata.obs['leiden'].nunique()} active categories")
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon', use_raw=True, key_added='wilcoxon_leiden')
    sc.pl.rank_genes_groups(adata, n_genes=20, key='wilcoxon_leiden', sharey=False, save=f"_{cli_args.final_run_prefix}_markers_leiden.png", show=False); plt.close()

    print("\n--- Step 9: Manual Annotation ---")
    if cli_args.cellmarker_db and os.path.exists(cli_args.cellmarker_db):
        try:
            print(f"       -> Annotating using marker DB: {cli_args.cellmarker_db}")
            header = pd.read_csv(cli_args.cellmarker_db, nrows=0).columns.tolist()
            type_col, gene_col = ('cell_name', 'Symbol') if 'cell_name' in header and 'Symbol' in header else (('Cell Type', 'Cell Marker') if 'Cell Type' in header and 'Cell Marker' in header else (None, None))
            if not type_col: raise ValueError("Marker DB must contain ('cell_name', 'Symbol') or ('Cell Type', 'Cell Marker') columns.")
            print(f"       -> Auto-detected format: TYPE='{type_col}', GENE='{gene_col}'")
            db_df = pd.read_csv(cli_args.cellmarker_db)
            db_markers_dict = defaultdict(set)
            for _, row in db_df.iterrows():
                if pd.notna(row.get(gene_col)) and pd.notna(row.get(type_col)):
                    db_markers_dict[row[type_col]].update({m.strip().upper() for m in str(row[gene_col]).split(',')})
            print(f"       -> Aggregated markers for {len(db_markers_dict)} unique cell types.")
            leiden_markers_df = sc.get.rank_genes_groups_df(adata, key='wilcoxon_leiden', group=None)
            cluster_annotations = {}
            for cluster in adata.obs['leiden'].cat.categories:
                cluster_genes = set(leiden_markers_df[leiden_markers_df['group'] == cluster].head(cli_args.n_top_genes)['names'].str.upper())
                scores = {cell_type: len(cluster_genes.intersection(db_genes)) / (len(cluster_genes.union(db_genes)) or 1) for cell_type, db_genes in db_markers_dict.items()}
                best_cell_type = max(scores, key=scores.get) if scores else None
                cluster_annotations[cluster] = best_cell_type if best_cell_type and scores[best_cell_type] > 0 else f"Unknown_{cluster}"
            adata.obs['manual_annotation'] = adata.obs['leiden'].map(cluster_annotations).astype('category')
            pd.DataFrame.from_dict(cluster_annotations, orient='index', columns=['AssignedType']).to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_leiden_to_manual_annotation.csv"))
            sc.pl.umap(adata, color='manual_annotation', title='Manual Cluster Annotation', palette=sc.pl.palettes.godsnot_102, legend_loc='right margin', legend_fontsize=8, size=10, show=False)
            fig_path = os.path.join(output_dir, f"{cli_args.final_run_prefix}_umap_manual_annotation.png"); _bold_right_margin_legend(fig_path); plt.close()
            
            print("       -> Calculating Marker Capture Score for manual annotation...")
            score_results = []
            leiden_degs_structured = adata.uns['wilcoxon_leiden']['names']
            for cluster_id, assigned_label in cluster_annotations.items():
                if pd.isna(assigned_label) or assigned_label.startswith("Unknown"): continue
                reference_genes = db_markers_dict.get(assigned_label, set())
                if not reference_genes: continue
                cluster_degs_for_capture = {g.upper() for g in leiden_degs_structured[cluster_id][:cli_args.n_degs_for_capture]}
                captured_genes = cluster_degs_for_capture.intersection(reference_genes)
                score = (len(captured_genes) / len(reference_genes)) * 100
                score_results.append({"Cluster_ID": cluster_id, "Assigned_Cell_Type": assigned_label, "Marker_Capture_Score (%)": score, "Captured_Genes_Count": len(captured_genes), "Total_Reference_Genes": len(reference_genes), "Captured_Genes_List": ", ".join(sorted(list(captured_genes)))})
            if score_results:
                capture_df = pd.DataFrame(score_results).sort_values(by="Marker_Capture_Score (%)", ascending=False)
                capture_df.to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_manual_annotation_marker_capture_scores.csv"), index=False)
                print(f"       -> Saved Marker Capture Scores.")
        except Exception as e: print(f"[ERROR] Manual annotation failed. Reason: {e}")
    else: print("[INFO] Cell marker DB not provided or not found. Skipping manual annotation.")

    print("\n--- Step 10: Compositional Analysis ---")
    composition_counts = pd.crosstab(adata.obs[FINAL_ANNOTATION_COLUMN], adata.obs['sample'])
    composition_perc = composition_counts.div(composition_counts.sum(axis=0), axis=1) * 100
    composition_perc.to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_composition_percentages.csv"))
    fig, ax = plt.subplots(figsize=(12, 8)); composition_perc.T.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_ylabel('Percentage of Cells'); ax.set_xlabel('Sample'); ax.set_title('Cell Type Composition by Sample')
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{cli_args.final_run_prefix}_composition_barchart.png")); plt.close()

    print("\n--- Step 11: Differential Gene Expression (DGE) Analysis ---")
    dge_results = []
    for cell_type in adata.obs[FINAL_ANNOTATION_COLUMN].cat.categories:
        print(f"  -> Running DGE for: {cell_type}")
        sub_adata = adata[(adata.obs[FINAL_ANNOTATION_COLUMN] == cell_type)].copy()
        
        # MODIFICATION START: More robust check for DGE viability
        counts_per_sample = sub_adata.obs['sample'].value_counts()
        if (CONDITION_OF_INTEREST not in counts_per_sample) or \
           (REFERENCE_CONDITION not in counts_per_sample) or \
           (counts_per_sample[CONDITION_OF_INTEREST] < 2) or \
           (counts_per_sample[REFERENCE_CONDITION] < 2):
            print(f"     [SKIP] Not enough cells for DGE in '{cell_type}'. "
                  f"Counts: {counts_per_sample.to_dict()}. Need at least 2 cells in both '{CONDITION_OF_INTEREST}' and '{REFERENCE_CONDITION}'.")
            continue
        # MODIFICATION END

        try:
            sc.tl.rank_genes_groups(sub_adata, 'sample', groups=[CONDITION_OF_INTEREST], reference=REFERENCE_CONDITION, method='wilcoxon', use_raw=True, key_added='dge_result')
            dge_df = sc.get.rank_genes_groups_df(sub_adata, key='dge_result', group=CONDITION_OF_INTEREST); dge_df['cell_type'] = cell_type
            dge_results.append(dge_df)
        except Exception as e: print(f"     [ERROR] DGE failed for '{cell_type}'. Reason: {e}")
    if dge_results:
        pd.concat(dge_results, ignore_index=True).to_csv(os.path.join(output_dir, f"{cli_args.final_run_prefix}_DGE_Treated_vs_WT_by_celltype.csv"), index=False)
        print("DGE analysis complete. Full results saved.")
    else: print("No DGE results were generated.")

    print("\n--- Step 12: Final Marker Heatmap ---")
    marker_key = f"wilcoxon_{FINAL_ANNOTATION_COLUMN}"
    if marker_key in adata.uns and top_genes_df is not None and not top_genes_df.empty:
        genes_to_plot_list = top_genes_df['names'].unique().tolist()
        print(f"       -> Generating heatmap with top {cli_args.n_top_genes} non-mitochondrial marker genes per cell type.")
        
        # --- FIX: Explicitly re-calculate the dendrogram for the current adata state ---
        sc.tl.dendrogram(adata, groupby=FINAL_ANNOTATION_COLUMN)
        # --- END FIX ---
        
        sc.pl.heatmap(adata, var_names=genes_to_plot_list, groupby=FINAL_ANNOTATION_COLUMN, show=False, dendrogram=True, save=f"_{cli_args.final_run_prefix}_top_markers_heatmap.png"); plt.close()
    else: 
        print(f"[WARNING] Marker key '{marker_key}' not found or markers not computed. Cannot generate heatmap.")

    print("\n--- Step 13: Saving Final AnnData Object ---")
    adata.write(os.path.join(output_dir, f"{cli_args.final_run_prefix}_final_processed.h5ad"))
    print(f"       -> Final annotated AnnData object saved.")

    print("\n" + "="*50 + f"\n--- Final Parameters Summary (Multi-Sample) ---\nRandom Seed Used: {cli_args.seed}\n\n--- Optimal Parameters Used ---\nBest n_hvg: {optimal_params['n_hvg']}\nBest n_pcs: {n_pcs_to_use}\nBest n_neighbors: {optimal_params['n_neighbors']}\nBest resolution: {optimal_params['resolution']:.3f}\n")
    print(f"Final_n_leiden_clusters: {adata.obs['leiden'].nunique()}")
    print(f"Final_silhouette_score: {silhouette_avg:.3f}")
    if FINAL_ANNOTATION_COLUMN in adata.obs.columns: print(f"Final_n_consensus_labels: {adata.obs[FINAL_ANNOTATION_COLUMN].nunique()}")
    print("="*50 + "\n\n--- MULTI-SAMPLE ANALYSIS PIPELINE COMPLETE ---")
    
    return adata, cas_path_for_refinement

# ==============================================================================
# ==============================================================================
# --- *** STAGE 3 & 4: REFINEMENT PIPELINE *** ---
# ==============================================================================
# ==============================================================================

# =========================================================================================
# === NEW HELPER FUNCTION for cumulative UMAP plotting during refinement ===
# =========================================================================================
def _generate_cumulative_refinement_umap(adata_full, failing_cell_indices, threshold, output_path, title, legend_fontsize=8):
    """
    (Refinement Helper) Generates a UMAP plot showing the CUMULATIVE progress of refinement.
    
    This function operates on the main, full AnnData object. It colors cells by their
    most up-to-date 'combined_annotation' and specifically colors a provided list of
    'failing_cell_indices' in grey.

    Args:
        adata_full (anndata.AnnData): The complete, original AnnData object from Stage 2.
                                      Must contain 'X_umap' and a 'combined_annotation' column.
        failing_cell_indices (pd.Index): Cell indices of the low-confidence cells for this level.
        threshold (float): The CAS percentage threshold, used for the plot title.
        output_path (str): Full path to save the output PNG image.
        title (str): The title for the UMAP plot.
        legend_fontsize (int): The font size for the legend text.
    """
    print(f"--- Generating cumulative refinement UMAP showing {len(failing_cell_indices)} failing cells in grey ---")

    # Use a copy of the full AnnData object to avoid modifying it
    adata_plot = adata_full.copy()

    # Create a temporary annotation column for plotting
    plot_annotation_col = 'plot_annotation_cumulative'
    adata_plot.obs[plot_annotation_col] = adata_plot.obs['combined_annotation'].astype(str)
    
    low_conf_label = 'Low-Confidence (<{:.0f}%)'.format(threshold)
    if len(failing_cell_indices) > 0:
        # Mark the currently failing cells with the special grey label
        adata_plot.obs.loc[failing_cell_indices, plot_annotation_col] = low_conf_label
    
    adata_plot.obs[plot_annotation_col] = adata_plot.obs[plot_annotation_col].astype('category')

    # Create a custom color palette that includes all seen annotations plus grey
    all_seen_cats = adata_plot.obs['combined_annotation'].cat.categories.tolist()
    # Use a consistent, large palette
    palette_to_use = sc.pl.palettes.godsnot_102
    color_map = {cat: color for cat, color in zip(all_seen_cats, palette_to_use)}
    
    if len(failing_cell_indices) > 0:
        color_map[low_conf_label] = '#bbbbbb'  # Medium grey

    # Generate the plot in memory
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(
            adata_plot,
            color=plot_annotation_col,
            palette=color_map,
            title=title,
            legend_loc='right margin',
            legend_fontsize=legend_fontsize,
            frameon=False,
            size=10,
            show=False,
            save=False
        )
    
    _bold_right_margin_legend(output_path)
    plt.close()
    print(f"       -> Saved cumulative progress UMAP to: {output_path}")

# =========================================================================================
# === NEW/REPLACEMENT FUNCTION for multi-level refinement ===
# =========================================================================================
def run_iterative_refinement_pipeline(args, adata_s2, cas_csv_path_s2):
    """
    Orchestrates iterative refinement and produces cumulative UMAPs showing the
    results of each refinement step, saving all final results.

    For each refinement level, this function:
    1. Identifies failing cells from the previous analysis level.
    2. Runs Stage 1 (BO) and Stage 2 (Final Analysis) on these failing cells. This
       preserves the detailed analysis files for each subset.
    3. Updates a master annotation column ('combined_annotation') in the main AnnData object.
    4. Generates a CUMULATIVE UMAP showing the *result* of this refinement:
       - Newly passing cells are now colored with their new labels.
       - Cells that *still* fail are shown in grey.
    5. Repeats this process up to `args.refinement_depth`.
    6. Saves a final, combined AnnData object and CSV with the new refined annotations.
    """
    global adata_base, model, RANDOM_SEED, ARGS, CURRENT_OPTIMIZATION_TARGET, CURRENT_STRATEGY_NAME, TRIAL_METADATA

    print("\n\n" + "="*80 + "\n### STARTING STAGE 3/4: ITERATIVE REFINEMENT PIPELINE ###\n" + "="*80)
    
    # --- Step 1: Initial Setup ---
    main_stage1_dir = os.path.join(args.output_dir, "stage_1_bayesian_optimization")
    stage2_output_dir = os.path.join(args.output_dir, "stage_2_final_analysis")
    
    # These variables track the state from one loop to the next
    current_cas_csv_path = cas_csv_path_s2
    adata_to_check = adata_s2 # AnnData from the previous analysis level
    adata_raw_full = adata_s2.raw.to_adata() # Full, original raw data
    
    all_refinement_cas_paths = []
    # This 'combined_annotation' is the master column that gets progressively updated
    adata_s2.obs['combined_annotation'] = adata_s2.obs['ctpt_consensus_prediction'].astype('category')

    original_bo_output_dir = args.output_dir
    original_bo_prefix = args.output_prefix
    original_final_run_prefix = args.final_run_prefix
    
    for depth in range(1, args.refinement_depth + 1):
        print("\n\n" + "#"*70 + f"\n### REFINEMENT DEPTH {depth}/{args.refinement_depth} ###\n" + "#"*70)

        # --- Step 2: Identify failing cells from the PREVIOUS level ---
        if not os.path.exists(current_cas_csv_path):
             print(f"[ERROR] Cannot find CAS file for depth {depth-1} at: {current_cas_csv_path}. Stopping refinement.")
             break

        cas_df_prev_level = pd.read_csv(current_cas_csv_path)

        # Identify the cells that are the INPUT for this refinement round
        if args.cas_aggregation_method == 'leiden':
            failing_cluster_ids = cas_df_prev_level[cas_df_prev_level['Cluster_Annotation_Score_CAS (%)'] < args.cas_refine_threshold]['Cluster_ID (Leiden)'].astype(str).tolist()
            if not failing_cluster_ids:
                print(f"✅ All clusters at depth {depth-1} met the {args.cas_refine_threshold}% CAS threshold. No further refinement needed.")
                break
            print(f"Found {len(failing_cluster_ids)} Leiden clusters below threshold at depth {depth-1}: {failing_cluster_ids}")
            failing_cell_indices_input = adata_to_check.obs[adata_to_check.obs['leiden'].isin(failing_cluster_ids)].index
        
        else: # 'consensus'
            failing_clusters = cas_df_prev_level[cas_df_prev_level['Cluster_Annotation_Score_CAS (%)'] < args.cas_refine_threshold]['Consensus_Cell_Type'].tolist()
            if not failing_clusters:
                print(f"✅ All clusters at depth {depth-1} met the {args.cas_refine_threshold}% CAS threshold. No further refinement needed.")
                break
            print(f"Found {len(failing_clusters)} consensus types below threshold at depth {depth-1}: {', '.join(failing_clusters)}")
            failing_cell_indices_input = adata_to_check.obs[adata_to_check.obs['ctpt_consensus_prediction'].isin(failing_clusters)].index

        if len(failing_cell_indices_input) < args.min_cells_refinement:
            print(f"\n[STOP] Stopping refinement. Only {len(failing_cell_indices_input)} failing cells, below the minimum of {args.min_cells_refinement}.\n")
            break

        # Isolate the subset of raw data for this refinement level
        adata_refine_raw = adata_raw_full[failing_cell_indices_input, :].copy()
        print(f"Isolated {adata_refine_raw.n_obs} cells for refinement analysis at depth {depth}.")

        # --- ADDED CHECK: Verify Gene Content (Pre-Flight Check) ---
        # Create a temp object to check if we have enough HVGs to satisfy the optimizer
        check_adata = adata_refine_raw.copy()
        sc.pp.normalize_total(check_adata, target_sum=1e4)
        sc.pp.log1p(check_adata)

        n_found_hvgs = 0
        # Check if using strict thresholding (Two-Step)
        if all(p is not None for p in [args.hvg_min_mean, args.hvg_max_mean, args.hvg_min_disp]):
             sc.pp.highly_variable_genes(
                check_adata,
                min_mean=args.hvg_min_mean,
                max_mean=args.hvg_max_mean,
                min_disp=args.hvg_min_disp,
                batch_key='sample' if 'sample' in check_adata.obs.columns else None
            )
             n_found_hvgs = check_adata.var.highly_variable.sum()
        else:
            # If rank based, the limit is just the total number of genes available
            n_found_hvgs = check_adata.n_vars

        MIN_HVG_LIMIT = 200 # As defined in Integer(200, 20000)
        if n_found_hvgs < MIN_HVG_LIMIT:
            print(f"\n[STOP] Stopping refinement at depth {depth}.")
            print(f"       Reason: Found only {n_found_hvgs} potential HVGs (or total genes), which is less than the optimizer lower bound of {MIN_HVG_LIMIT}.")
            print(f"       This usually happens when the subset is too homogeneous or too small.")
            break
        # -----------------------------------------------------------
        
        # --- Step 3: Run Stage 1 (BO) on the subset ---
        stage1_refinement_dir = os.path.join(main_stage1_dir, f"refinement_depth_{depth}")
        os.makedirs(stage1_refinement_dir, exist_ok=True)
        
        args.output_dir = stage1_refinement_dir
        args.output_prefix = f"{original_bo_prefix}_refinement_depth_{depth}"
        
        print(f"\n--- [Depth {depth}] Running new Bayesian Optimization on subset. Outputs will be in: {stage1_refinement_dir} ---")
        refinement_bo_results = run_stage_one_optimization(args, adata_input=adata_refine_raw)
        refinement_optimal_params = refinement_bo_results['params']
        
        args.output_dir = original_bo_output_dir # Restore for next loop or finalization
        args.output_prefix = original_bo_prefix

        # --- Step 4: Run Stage 2 (Final Analysis) on the subset ---
        stage2_refinement_dir = os.path.join(stage2_output_dir, f"refinement_depth_{depth}")
        os.makedirs(stage2_refinement_dir, exist_ok=True)

        print(f"\n--- [Depth {depth}] Running Final Analysis on subset. Outputs will be in: {stage2_refinement_dir} ---")
        is_multi_sample_refinement = 'sample' in adata_refine_raw.obs.columns
        
        args.final_run_prefix = f"{original_final_run_prefix}_refinement_depth_{depth}"

        if is_multi_sample_refinement:
            adata_refinement_processed, cas_csv_path_refinement = run_stage_two_final_analysis_multi_sample(
                cli_args=args, optimal_params=refinement_optimal_params, output_dir=stage2_refinement_dir, adata_input=adata_refine_raw
            )
        else:
            adata_refinement_processed, cas_csv_path_refinement = run_stage_two_final_analysis(
                cli_args=args, optimal_params=refinement_optimal_params, output_dir=stage2_refinement_dir, adata_input=adata_refine_raw
            )
        
        args.final_run_prefix = original_final_run_prefix # Restore for next loop

        # Check if the refinement produced too many clusters (Over-clustering check)
        # If clusters > 1/5th of the cells, discard this level and stop.
        
        current_n_clusters = adata_refinement_processed.obs['leiden'].nunique()
        current_n_cells = adata_refinement_processed.n_obs
        ratio_threshold = current_n_cells / 5.0

        if current_n_clusters > ratio_threshold:
            print(f"\n[STOP] Refinement stopped at Depth {depth} due to over-clustering.")
            print(f"       Reason: Cluster count ({current_n_clusters}) is higher than 1/5th of processing cells ({current_n_cells}).")
            print(f"       Threshold was: > {ratio_threshold:.2f}")
            print(f"       Results from this depth will NOT be merged into the final object.")
            break
        # ==============================================================================

        # --- Step 5: Update master annotation in the main adata_s2 object ---
        all_refinement_cas_paths.append(cas_csv_path_refinement)
        
        refinement_annotations = adata_refinement_processed.obs['ctpt_consensus_prediction']
        
        # Ensure new categories from this refinement are added to the master annotation column
        current_categories = adata_s2.obs['combined_annotation'].cat.categories.tolist()
        new_labels = refinement_annotations.unique()
        new_categories_to_add = [label for label in new_labels if label not in current_categories]
        if new_categories_to_add:
            print(f"       -> Adding new categories to master list: {new_categories_to_add}")
            adata_s2.obs['combined_annotation'] = adata_s2.obs['combined_annotation'].cat.add_categories(new_categories_to_add)

        # Now, perform the assignment of new labels
        adata_s2.obs.loc[refinement_annotations.index, 'combined_annotation'] = refinement_annotations.astype(str)
        adata_s2.obs['combined_annotation'] = adata_s2.obs['combined_annotation'].astype('category') # Recategorize
        adata_s2.obs['combined_annotation'] = adata_s2.obs['combined_annotation'].cat.remove_unused_categories()
        print(f"--- [Depth {depth}] Updated {len(refinement_annotations)} cell annotations in the main object. Active categories: {adata_s2.obs['combined_annotation'].nunique()} ---")
        print(f"--- [Depth {depth}] Updated {len(refinement_annotations)} cell annotations in the main object. ---")
        
        # --- Step 6: Identify cells that ARE STILL FAILING for the cumulative plot ---
        failing_cell_indices_output = pd.Index([]) # Default to empty
        if os.path.exists(cas_csv_path_refinement):
            cas_df_this_level = pd.read_csv(cas_csv_path_refinement)
            
            if args.cas_aggregation_method == 'leiden':
                still_failing_ids = cas_df_this_level[cas_df_this_level['Cluster_Annotation_Score_CAS (%)'] < args.cas_refine_threshold]['Cluster_ID (Leiden)'].astype(str).tolist()
                if still_failing_ids:
                    failing_cell_indices_output = adata_refinement_processed.obs[adata_refinement_processed.obs['leiden'].isin(still_failing_ids)].index
            else: # 'consensus'
                still_failing_types = cas_df_this_level[cas_df_this_level['Cluster_Annotation_Score_CAS (%)'] < args.cas_refine_threshold]['Consensus_Cell_Type'].tolist()
                if still_failing_types:
                    failing_cell_indices_output = adata_refinement_processed.obs[adata_refinement_processed.obs['ctpt_consensus_prediction'].isin(still_failing_types)].index
        
        print(f"--- [Depth {depth}] Found {len(failing_cell_indices_output)} cells that are *still* failing. These will be grey in the cumulative UMAP. ---")

        # --- Step 7: Generate the cumulative UMAP showing the *result* of this depth's analysis ---
        # This plot is saved in the current depth's directory and shows progress.
        greyed_umap_path = os.path.join(stage2_refinement_dir, f"{args.final_run_prefix}_refinement_depth_{depth}_umap_cumulative_result.png")
        _generate_cumulative_refinement_umap(
            adata_full=adata_s2, # Use the main object with updated 'combined_annotation'
            failing_cell_indices=failing_cell_indices_output, # Grey out only the cells that are still failing
            threshold=args.cas_refine_threshold,
            output_path=greyed_umap_path,
            title=f'Refinement Level {depth}: Cumulative Result\n({len(failing_cell_indices_output)} cells remain low-confidence)',
            legend_fontsize=8
        )

        # --- Step 8: Update state for the next iteration ---
        current_cas_csv_path = cas_csv_path_refinement
        adata_to_check = adata_refinement_processed

    # --- FINAL COMBINATION (after loop) ---
    print("\n\n" + "="*80 + "\n### FINALIZING ALL REFINEMENT RESULTS ###\n" + "="*80)
    
    # ==========================================================================
    # === NEW: Generate Final Refined Annotation UMAP (Primary Output) ===
    # ==========================================================================
    print("--- Generating FINAL REFINED ANNOTATION UMAP (Primary Result) ---")
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(adata_s2, color='combined_annotation', palette=sc.pl.palettes.godsnot_102, 
                legend_loc='right margin', legend_fontsize=8, 
                title=f'FINAL Refined Cell Type Annotation\n({adata_s2.obs["combined_annotation"].nunique()} types, n={adata_s2.n_obs} cells)', 
                show=False, size=10)

    refined_umap_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_FINAL_refined_annotation_umap.png")
    _bold_right_margin_legend(refined_umap_path); plt.close()
    print(f"✅ Success! Saved FINAL refined annotation UMAP to: {refined_umap_path}")

    # ==========================================================================
    # === NEW: Generate Comparison UMAP - Before vs After Refinement ===
    # ==========================================================================
    print("--- Generating Before/After Refinement Comparison UMAPs ---")
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Left panel: Original consensus (before refinement)
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(adata_s2, color='ctpt_consensus_prediction', palette=sc.pl.palettes.godsnot_102,
                   ax=axes[0], show=False, size=8, legend_loc=None,
                   title=f'BEFORE Refinement (Original Consensus)\n({adata_s2.obs["ctpt_consensus_prediction"].nunique()} types)')
        
        # Right panel: After refinement (combined_annotation)
        sc.pl.umap(adata_s2, color='combined_annotation', palette=sc.pl.palettes.godsnot_102,
                   ax=axes[1], show=False, size=8, legend_loc=None,
                   title=f'AFTER Refinement (Combined Annotation)\n({adata_s2.obs["combined_annotation"].nunique()} types)')
    
    plt.tight_layout()
    comparison_umap_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_refinement_before_after_comparison.png")
    plt.savefig(comparison_umap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Success! Saved before/after comparison UMAP to: {comparison_umap_path}")

    # ==========================================================================
    # === NEW: Generate UMAP showing cells that CHANGED during refinement ===
    # ==========================================================================
    print("--- Generating UMAP highlighting cells that CHANGED annotation ---")
    
    # Identify cells where annotation changed
    original_labels = adata_s2.obs['ctpt_consensus_prediction'].astype(str)
    refined_labels = adata_s2.obs['combined_annotation'].astype(str)
    changed_mask = (original_labels != refined_labels)
    n_changed = changed_mask.sum()
    
    print(f"       -> {n_changed} cells ({n_changed/len(adata_s2)*100:.2f}%) changed annotation during refinement")
    
    # Create visualization column
    change_viz_col = '_refinement_change_status'
    adata_s2.obs[change_viz_col] = 'Unchanged'
    adata_s2.obs.loc[changed_mask, change_viz_col] = 'Refined/Changed'
    adata_s2.obs[change_viz_col] = adata_s2.obs[change_viz_col].astype('category')
    
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(adata_s2, color=change_viz_col, 
                   palette={'Unchanged': '#cccccc', 'Refined/Changed': '#e74c3c'},
                   legend_loc='right margin', legend_fontsize=12,
                   title=f'Cells Changed During Refinement\n({n_changed} cells refined, {n_changed/len(adata_s2)*100:.1f}%)',
                   show=False, size=10)
    
    changed_umap_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_cells_changed_by_refinement_umap.png")
    _bold_right_margin_legend(changed_umap_path); plt.close()
    print(f"✅ Success! Saved changed cells UMAP to: {changed_umap_path}")
    
    # Clean up temporary column
    del adata_s2.obs[change_viz_col]

    # ==========================================================================
    # === NEW: Generate Inconsistency UMAP based on REFINED annotations ===
    # ==========================================================================
    print("--- Generating UMAP showing inconsistent cells (Individual vs Refined) ---")
    
    individual_labels = adata_s2.obs['ctpt_individual_prediction'].astype(str)
    refined_labels = adata_s2.obs['combined_annotation'].astype(str)
    inconsistent_mask = (individual_labels != refined_labels)
    n_inconsistent = inconsistent_mask.sum()
    
    print(f"       -> {n_inconsistent} cells ({n_inconsistent/len(adata_s2)*100:.2f}%) have Individual != Refined annotation")
    
    # Create visualization column for inconsistency
    inconsist_viz_col = '_refined_inconsistency'
    adata_s2.obs[inconsist_viz_col] = adata_s2.obs['combined_annotation'].astype(str)
    inconsistent_label = f'Inconsistent ({n_inconsistent} cells)'
    adata_s2.obs.loc[inconsistent_mask, inconsist_viz_col] = inconsistent_label
    adata_s2.obs[inconsist_viz_col] = adata_s2.obs[inconsist_viz_col].astype('category')
    
    # Build color palette
    unique_labels = adata_s2.obs[inconsist_viz_col].cat.categories.tolist()
    palette_map = {}
    std_palette = sc.pl.palettes.godsnot_102
    color_idx = 0
    for label in unique_labels:
        if label == inconsistent_label:
            palette_map[label] = '#bbbbbb'  # Grey for inconsistent
        else:
            palette_map[label] = std_palette[color_idx % len(std_palette)]
            color_idx += 1
    
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(adata_s2, color=inconsist_viz_col, palette=palette_map,
                   legend_loc='right margin', legend_fontsize=8,
                   title=f'Refined Annotation with Inconsistent Cells in Grey\n(Individual ≠ Refined: {n_inconsistent} cells, {n_inconsistent/len(adata_s2)*100:.1f}%)',
                   show=False, size=10)
    
    inconsist_refined_umap_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_FINAL_refined_with_inconsistent_grey_umap.png")
    _bold_right_margin_legend(inconsist_refined_umap_path); plt.close()
    print(f"✅ Success! Saved refined annotation with inconsistent cells greyed UMAP to: {inconsist_refined_umap_path}")
    
    # Clean up
    del adata_s2.obs[inconsist_viz_col]

    # ==========================================================================
    # === NEW: Export FINAL Refined Annotations CSV (Comprehensive) ===
    # ==========================================================================
    print("--- Exporting FINAL refined annotations CSV ---")
    
    # Create a comprehensive annotations dataframe
    final_annotations_df = pd.DataFrame({
        'cell_barcode': adata_s2.obs_names,
        'leiden_cluster': adata_s2.obs['leiden'].values,
        'individual_prediction': adata_s2.obs['ctpt_individual_prediction'].values,
        'original_consensus': adata_s2.obs['ctpt_consensus_prediction'].values,
        'FINAL_refined_annotation': adata_s2.obs['combined_annotation'].values,
        'annotation_changed': (adata_s2.obs['ctpt_consensus_prediction'].astype(str) != adata_s2.obs['combined_annotation'].astype(str)).values,
        'individual_matches_refined': (adata_s2.obs['ctpt_individual_prediction'].astype(str) == adata_s2.obs['combined_annotation'].astype(str)).values
    })
    
    # Add confidence if available
    if 'ctpt_confidence' in adata_s2.obs.columns:
        final_annotations_df['prediction_confidence'] = adata_s2.obs['ctpt_confidence'].values
    
    final_annotations_csv_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_FINAL_refined_annotations.csv")
    final_annotations_df.to_csv(final_annotations_csv_path, index=False)
    print(f"✅ Success! Saved FINAL refined annotations CSV to: {final_annotations_csv_path}")
    
    # ==========================================================================
    # === NEW: Generate Summary Statistics for Refined Annotations ===
    # ==========================================================================
    print("--- Generating refined annotation summary statistics ---")
    
    summary_stats = {
        'Total_Cells': len(adata_s2),
        'N_Cell_Types_Original': adata_s2.obs['ctpt_consensus_prediction'].nunique(),
        'N_Cell_Types_Refined': adata_s2.obs['combined_annotation'].nunique(),
        'N_Cells_Changed': int(changed_mask.sum()),
        'Pct_Cells_Changed': f"{changed_mask.sum()/len(adata_s2)*100:.2f}%",
        'N_Cells_Consistent_Individual_vs_Refined': int((~inconsistent_mask).sum()),
        'Pct_Cells_Consistent': f"{(~inconsistent_mask).sum()/len(adata_s2)*100:.2f}%"
    }
    
    # Cell type distribution in refined annotation
    refined_counts = adata_s2.obs['combined_annotation'].value_counts()
    
    summary_lines = [
        "="*60,
        "FINAL REFINED ANNOTATION SUMMARY",
        "="*60,
        f"Total Cells: {summary_stats['Total_Cells']}",
        f"Original Cell Types: {summary_stats['N_Cell_Types_Original']}",
        f"Refined Cell Types: {summary_stats['N_Cell_Types_Refined']}",
        "",
        f"Cells Changed by Refinement: {summary_stats['N_Cells_Changed']} ({summary_stats['Pct_Cells_Changed']})",
        f"Cells Consistent (Individual == Refined): {summary_stats['N_Cells_Consistent_Individual_vs_Refined']} ({summary_stats['Pct_Cells_Consistent']})",
        "",
        "REFINED CELL TYPE DISTRIBUTION:",
        "-"*40
    ]
    
    for ct, count in refined_counts.items():
        summary_lines.append(f"  {ct}: {count} cells ({count/len(adata_s2)*100:.2f}%)")
    
    summary_lines.append("="*60)
    
    summary_txt_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_FINAL_refinement_summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"✅ Success! Saved refinement summary to: {summary_txt_path}")
    
    # Print summary to console as well
    print('\n'.join(summary_lines))

    # ==========================================================================
    # === EXISTING CODE CONTINUES (with minor path rename for clarity) ===
    # ==========================================================================
    print("\n--- Generating legacy combined UMAP plot (for backward compatibility) ---")
    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
        sc.pl.umap(adata_s2, color='combined_annotation', palette=sc.pl.palettes.godsnot_102, 
                legend_loc='right margin', legend_fontsize=8, 
                title='Final Annotation (High-Confidence + All Refined Levels)', 
                show=False, size=10)

    combined_umap_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_umap_combined_annotation_final.png")
    _bold_right_margin_legend(combined_umap_path); plt.close()
    print(f"✅ Success! Saved final combined UMAP to: {combined_umap_path}")

    print("--- Generating final combined CAS score sheet ---")
    passing_cas_df = pd.read_csv(cas_csv_path_s2)
    passing_cas_df = passing_cas_df[passing_cas_df['Cluster_Annotation_Score_CAS (%)'] >= args.cas_refine_threshold]
    passing_cas_df['source_level'] = 'initial_high_confidence'
    all_cas_dfs = [passing_cas_df]
    
    for i, path in enumerate(all_refinement_cas_paths):
        try:
            refinement_cas_df = pd.read_csv(path)
            refinement_cas_df['source_level'] = f'refinement_depth_{i+1}'
            all_cas_dfs.append(refinement_cas_df)
        except FileNotFoundError:
            print(f"[WARNING] Could not find CAS file for refinement level {i+1} at '{path}'. Skipping.")
    
    if len(all_cas_dfs) > 1: # Only save if refinement actually happened
        combined_cas_df = pd.concat(all_cas_dfs, ignore_index=True)
        combined_cas_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_combined_cluster_annotation_scores.csv")
        combined_cas_df.to_csv(combined_cas_path, index=False)
        print(f"✅ Success! Saved combined CAS scores to: {combined_cas_path}")

        # --- START: CALL TO THE NEW JOURNEY SUMMARY FUNCTION ---
        print("--- Generating cell type journey summary report ---")
        journey_summary_output_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_cell_type_journey_summary.csv")
        summarize_annotation_journey(
            input_file=combined_cas_path,
            output_file=journey_summary_output_path
        )
        # --- END: CALL TO THE NEW JOURNEY SUMMARY FUNCTION ---

    print("--- Saving final AnnData object and annotations CSV with refinement results ---")
    final_adata_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_final_processed_with_refinement.h5ad")
    adata_s2.write(final_adata_path)
    print(f"✅ Success! Saved final AnnData object to: {final_adata_path}")
    
    # Save the final annotations to a new CSV, including the new 'combined_annotation' column
    final_csv_path = os.path.join(stage2_output_dir, f"{args.final_run_prefix}_all_annotations_with_refinement.csv")
    cols_to_save = [c for c in adata_s2.obs.columns if c in ['leiden', 'ctpt_individual_prediction', 'ctpt_confidence', 'ctpt_consensus_prediction', 'manual_annotation', 'combined_annotation']]
    adata_s2.obs[cols_to_save].to_csv(final_csv_path)
    print(f"✅ Success! Saved final annotations with refinement column to: {final_csv_path}")

# ==============================================================================
# ==============================================================================
# --- *** PIPELINE ORCHESTRATION *** ---
# ==============================================================================
# ==============================================================================

def run_stage_one_optimization(args, adata_input=None):
    """
    Executes the entire Bayesian optimization pipeline (Stage 1).
    Can load data from disk (if `adata_input` is None) or use a provided
    AnnData object (for refinement runs).
    Returns a dictionary with the best parameters.
    """
    global adata_base, model, RANDOM_SEED, ARGS, CURRENT_OPTIMIZATION_TARGET, CURRENT_STRATEGY_NAME, TRIAL_METADATA

    ARGS = args
    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    sc.settings.verbosity = 0
    sc.logging.print_header()
    os.makedirs(args.output_dir, exist_ok=True)

    # Track if we need batch integration
    needs_integration = False
    integration_key = None

    if adata_input is None:
        if args.data_dir:
            print("--- Loading Data from Single Directory ---")
            adata = load_expression_data(args.data_dir)
            
            # Debug output
            print(f"       -> Loaded {adata.n_obs} cells, {adata.n_vars} genes")
            print(f"       -> Sample barcodes: {adata.obs_names[:3].tolist()}")
            
            # =====================================================================
            # CRITICAL FIX: Detect batch from barcodes BEFORE checking batch_key
            # =====================================================================
            print("\n--- Detecting batch/sample information from barcodes ---")
            adata, batches_detected = detect_batch_from_barcodes(adata)
            
            if batches_detected:
                n_batches = adata.obs['sample'].nunique()
                batch_names = adata.obs['sample'].cat.categories.tolist()
                print(f"       -> ✓ Successfully detected {n_batches} batches from barcodes")
                print(f"       -> Batch names: {batch_names}")
                for batch in batch_names:
                    n_cells = (adata.obs['sample'] == batch).sum()
                    print(f"          - {batch}: {n_cells} cells")
            else:
                print("       -> No batch pattern detected in barcodes")
            # =====================================================================
            
            # === Check for batch information ===
            if args.no_integration or args.integration_method == 'none':
                print("--- Running in SINGLE-SAMPLE Mode (integration disabled by user) ---")
                needs_integration = False
            elif args.batch_key is not None:
                # Now check if the batch_key exists (it should if detect_batch_from_barcodes created 'sample')
                if args.batch_key in adata.obs.columns:
                    n_unique_batches = adata.obs[args.batch_key].nunique()
                    # =========================================================
                    # FIX: Only enable integration if there are 2+ batches
                    # =========================================================
                    if n_unique_batches >= 2:
                        needs_integration = True
                        integration_key = args.batch_key
                        print(f"--- Running in INTEGRATED Mode (using specified '{integration_key}' column) ---")
                        print(f"       -> Found {n_unique_batches} unique batches")
                    else:
                        needs_integration = False
                        print(f"--- Running in SINGLE-SAMPLE Mode ---")
                        print(f"       -> Only {n_unique_batches} batch found in '{args.batch_key}'. Harmony requires 2+ batches.")
                else:
                    # Try to check for other existing batch columns
                    existing_batch, has_batches = check_existing_batch_column(adata, batch_key=args.batch_key)
                    if has_batches:
                        n_unique_batches = adata.obs[existing_batch].nunique()
                        if n_unique_batches >= 2:
                            needs_integration = True
                            integration_key = existing_batch
                            print(f"--- Running in INTEGRATED Mode (using '{integration_key}' column) ---")
                            print(f"       -> Found {n_unique_batches} unique batches")
                        else:
                            needs_integration = False
                            print(f"--- Running in SINGLE-SAMPLE Mode ---")
                            print(f"       -> Only {n_unique_batches} batch found. Harmony requires 2+ batches.")
                    else:
                        print(f"[WARNING] Specified --batch_key '{args.batch_key}' not found. Proceeding without integration.")
            elif batches_detected:
                # Auto-detected batches from barcodes
                n_unique_batches = adata.obs['sample'].nunique()
                # =========================================================
                # FIX: Only enable integration if there are 2+ batches
                # =========================================================
                if n_unique_batches >= 2:
                    needs_integration = True
                    integration_key = 'sample'
                    print(f"--- Running in INTEGRATED Mode (auto-detected from barcodes) ---")
                    print(f"       -> Found {n_unique_batches} unique batches")
                else:
                    needs_integration = False
                    print(f"--- Running in SINGLE-SAMPLE Mode ---")
                    print(f"       -> Only {n_unique_batches} batch detected from barcodes. Harmony requires 2+ batches.")
            else:
                # Check for existing batch columns in metadata
                existing_batch, has_batches = check_existing_batch_column(adata)
                
                if has_batches:
                    n_unique_batches = adata.obs[existing_batch].nunique()
                    if n_unique_batches >= 2:
                        needs_integration = True
                        integration_key = existing_batch
                        print(f"--- Running in INTEGRATED Mode (detected '{integration_key}' column) ---")
                        print(f"       -> Found {n_unique_batches} unique batches")
                    else:
                        needs_integration = False
                        print(f"--- Running in SINGLE-SAMPLE Mode ---")
                        print(f"       -> Only {n_unique_batches} batch found in '{existing_batch}'. Harmony requires 2+ batches.")
                else:
                    # Try to detect batch from barcodes
                    adata, detected_batches = detect_batch_from_barcodes(adata)
                    
                    if detected_batches:
                        n_unique_batches = adata.obs['barcode_batch'].nunique() if 'barcode_batch' in adata.obs.columns else 1
                        if n_unique_batches >= 2:
                            needs_integration = True
                            integration_key = 'barcode_batch'
                            print(f"--- Running in INTEGRATED Mode (detected batches from barcodes) ---")
                            print(f"       -> Found {n_unique_batches} unique batches")
                        else:
                            needs_integration = False
                            print("--- Running in SINGLE-SAMPLE Mode (only 1 batch detected from barcodes) ---")
                    else:
                        print("--- Running in SINGLE-SAMPLE Mode (no batches detected) ---")
            
            # Add 'sample' column for consistency with multi_sample mode
            if needs_integration and integration_key != 'sample':
                adata.obs['sample'] = adata.obs[integration_key].astype(str)
                print(f"       -> Created 'sample' column from '{integration_key}'")
                print(f"       -> Unique samples: {adata.obs['sample'].unique().tolist()}")
            
            adata_merged = adata
            
        elif args.multi_sample:
            print("--- Running in MULTI-SAMPLE (Harmony Integration) Mode ---")
            wt_path, treated_path = args.multi_sample
            adatas = {
                'WT': load_expression_data(wt_path), 
                'Treated': load_expression_data(treated_path)
            }
            for sample_id, adata_sample in adatas.items():
                adata_sample.obs['sample'] = sample_id
            adata_merged = anndata.AnnData.concatenate(*adatas.values(), batch_key='sample', batch_categories=list(adatas.keys()))
            print(f"Combined data: {adata_merged.n_obs} cells, {adata_merged.n_vars} genes")
        else:
            raise ValueError("Invalid arguments. Must provide --data_dir or --multi_sample for the initial run.")
    else:
        print("--- Using provided AnnData object for optimization ---")
        adata_merged = adata_input.copy()

    # Intersect with Spatial Data if provided (Before QC/Processing)
    if args.st_data_dir is not None:
        print(f"\n[INFO] Loading Spatial Data for gene intersection from: {args.st_data_dir}")
        adata_st_temp = load_expression_data(args.st_data_dir)
        
        common_genes = list(set(adata_merged.var_names) & set(adata_st_temp.var_names))
        common_genes.sort()
        print(f"       -> Genes in scRNA-seq: {adata_merged.n_vars}")
        print(f"       -> Genes in Spatial:   {adata_st_temp.n_vars}")
        print(f"       -> Intersection:       {len(common_genes)} common genes.")
        
        if len(common_genes) < 100:
             print("[WARNING] Very few common genes found! Check gene symbol capitalization.")

        # Subset scRNA-seq to common genes
        adata_merged = adata_merged[:, common_genes].copy()
        print(f"       -> scRNA-seq data subsetted to {len(common_genes)} genes.")
        del adata_st_temp # Free memory

    print("\n--- Performing initial QC and normalization ---")
    if 'mt' not in adata_merged.var.columns:
        adata_merged.var['mt'] = adata_merged.var_names.str.contains(MITO_REGEX_PATTERN, regex=True)
        sc.pp.calculate_qc_metrics(adata_merged, qc_vars=['mt'], inplace=True)
        sc.pp.filter_cells(adata_merged, min_genes=args.min_genes)
        sc.pp.filter_cells(adata_merged, max_genes=args.max_genes)
        adata_merged = adata_merged[adata_merged.obs.pct_counts_mt < args.max_pct_mt, :]
        sc.pp.filter_genes(adata_merged, min_cells=args.min_cells)
    
    print(f"Data for this BO run: {adata_merged.n_obs} cells, {adata_merged.n_vars} genes")
    sc.pp.normalize_total(adata_merged, target_sum=1e4); sc.pp.log1p(adata_merged); adata_merged.raw = adata_merged.copy()
    adata_base = adata_merged.copy(); model = models.Model.load(args.model_path)
    print("Initial setup complete. Base AnnData object created for optimization.")

    global MARKER_PRIOR_DICT
    if hasattr(args, 'marker_prior_db') and args.marker_prior_db:
        # Detect species
        detected_species = detect_species_from_model_or_data(args.model_path, adata_merged)
        
        MARKER_PRIOR_DICT = load_marker_prior_database(
            csv_path=args.marker_prior_db,
            species_filter=args.marker_prior_species if hasattr(args, 'marker_prior_species') else 'Human',
            organ_filter=args.marker_prior_organ if hasattr(args, 'marker_prior_organ') else None
        )
        
        # Store species for later use
        args._detected_species = detected_species
    else:
        print("\n--- No Marker Prior Database specified. MPS will be disabled. ---")
        MARKER_PRIOR_DICT = {}

    local_search_space = [dim for dim in search_space]
    if all(p is not None for p in [args.hvg_min_mean, args.hvg_max_mean, args.hvg_min_disp]):
        print("\n--- Two-step HVG mode enabled. Pre-calculating gene filter... ---")
        adata_temp = adata_base.copy()
        sc.pp.highly_variable_genes(adata_temp, min_mean=args.hvg_min_mean, max_mean=args.hvg_max_mean, min_disp=args.hvg_min_disp, batch_key='sample' if 'sample' in adata_base.obs.columns else None)
        n_filtered_genes = adata_temp.var.highly_variable.sum()
        print(f"       -> Found {n_filtered_genes} genes passing thresholds.")
        original_min_hvg = next(dim.low for dim in search_space if dim.name == 'n_hvg')
        if n_filtered_genes < original_min_hvg: print(f"[ERROR] HVG filtering resulted in only {n_filtered_genes} genes, below the minimum search bound of {original_min_hvg}. Please relax your HVG filtering thresholds."); exit(1)
        for i, dim in enumerate(local_search_space):
            if dim.name == 'n_hvg':
                print(f"       -> Adjusting 'n_hvg' search space to [{original_min_hvg}, {n_filtered_genes}].")
                local_search_space[i] = Integer(original_min_hvg, n_filtered_genes, name='n_hvg'); break
    else: print("\n--- Using standard rank-based HVG selection mode. ---")

    param_names = ['n_hvg', 'n_pcs', 'n_neighbors', 'resolution']
    targets_to_run = ['balanced'] if args.target == 'all' else [args.target]
    best_params_for_stage2 = None

    for target in targets_to_run:
        target_name_map = {'weighted_cas': 'WEIGHTED MEAN CAS', 'simple_cas': 'SIMPLE MEAN CAS', 'mcs': 'MEAN MCS', 'balanced': 'BALANCED SCORE (CAS & MCS)'}
        if args.model_type == 'structural': target_name_map['balanced'] = 'BALANCED SCORE (CAS, MCS & SILHOUETTE)'
        elif args.model_type == 'silhouette': target_name_map['balanced'] = 'SILHOUETTE SCORE'
        print("\n\n" + "#"*70 + f"\n### STAGE: OPTIMIZING FOR {target_name_map[target]} ###\n" + "#"*70)
        CURRENT_OPTIMIZATION_TARGET = target
        strategies = {"Exploit": {'acq_func': 'PI', 'xi': 0.01}, "BO-EI": {'acq_func': 'EI', 'xi': 0.01}, "Explore": {'acq_func': 'EI', 'xi': 0.1}}
        output_prefix_model = f"{args.output_prefix}_{args.model_type}"
        results, skopt_file_paths = {}, []
        for name, params in strategies.items():
            print(f"\n--- Running Strategy: {name} ---"); CURRENT_STRATEGY_NAME = name; TRIAL_METADATA.clear()
            result = gp_minimize(func=objective_function, dimensions=local_search_space, n_calls=args.n_calls, random_state=RANDOM_SEED, **params)
            result.trial_metadata = list(TRIAL_METADATA); results[name] = result
            result_path = os.path.join(args.output_dir, f"{output_prefix_model}_{target}_{name.lower().replace('-','_')}_opt_result.skopt")
            dump(result, result_path, store_objective=False); skopt_file_paths.append(result_path); print(f"Saved {name} optimization state to {result_path}")

        generate_yield_csv(results, target, args.output_dir, output_prefix_model)
        plot_optimizer_paths_tsne(results, target, args.output_dir, output_prefix_model, n_points_to_show=args.n_calls)
        plot_optimizer_paths_umap(results, target, args.output_dir, output_prefix_model, n_points_to_show=args.n_calls)
        plot_optimizer_convergence(results, target, args.output_dir, output_prefix_model)
        plot_exact_scores_per_trial(results, target, args.output_dir, output_prefix_model)
        generate_skopt_visualizations(skopt_files=skopt_file_paths, output_prefix_base=os.path.join(args.output_dir, f"{output_prefix_model}_{target}"), target_metric=target)

        best_overall_score, best_result_obj, winning_strategy_name = float('inf'), None, ""
        best_trial_metadata = None  # NEW: Store the metadata from the winning trial

        for name, result in results.items():
            if result.fun < best_overall_score: 
                best_overall_score = result.fun
                best_result_obj = result
                winning_strategy_name = name
                
                # NEW: Find the best trial's metadata from the stored trial_metadata
                if hasattr(result, 'trial_metadata') and result.trial_metadata:
                    # Find the index of the best trial (minimum objective value)
                    best_trial_idx = np.argmin(result.func_vals)
                    if best_trial_idx < len(result.trial_metadata):
                        best_trial_metadata = result.trial_metadata[best_trial_idx]

        best_score_print = -best_overall_score
        format_str = ".3f" if args.model_type == 'silhouette' else ".2f"
        print(f"\n--- Analysis Complete for {target_name_map[target]} ---\nOverall best score ({best_score_print:{format_str}}) was found by the '{winning_strategy_name}' strategy.")

        best_params_for_stage2 = dict(zip(param_names, best_result_obj.x))
        final_metrics, adata_final = evaluate_final_metrics(best_params_for_stage2)
        print_final_report(target, best_params_for_stage2, final_metrics, winning_strategy_name, 
                    stored_trial_metadata=best_trial_metadata)
        txt_path = os.path.join(args.output_dir, f"{output_prefix_model}_{target}_FINAL_best_params.txt")
        h5ad_path = os.path.join(args.output_dir, f"{output_prefix_model}_{target}_FINAL_annotated.h5ad")
        save_results_to_file(txt_path, target, best_params_for_stage2, final_metrics, winning_strategy_name,
                        stored_trial_metadata=best_trial_metadata)
        adata_final.write(h5ad_path)
        print(f"\nFinal optimized results for {target} saved to:\n  - {txt_path}\n  - {h5ad_path}")

    print("\n\n--- Stage 1 (Optimization) Complete ---")
    return_data = {"params": best_params_for_stage2}
    if args.data_dir:
        return_data["data_dir"] = args.data_dir
    elif args.multi_sample:
        return_data["wt_dir"], return_data["treated_dir"] = args.multi_sample
    
    return return_data

def export_consistent_cells(args, adata):
    """
    (New Feature) Exports a subset of the data containing only cells where ALL THREE
    annotation columns are identical:
    - ctpt_individual_prediction
    - ctpt_consensus_prediction  
    - combined_annotation (if refinement was run)
    
    If refinement was NOT run, requires match between individual and consensus only.
    
    Outputs a CSV and a UMAP to a new subdirectory.
    """
    print("\n\n" + "="*80 + "\n### EXPORTING CONSISTENT CELL POPULATIONS ###\n" + "="*80)
    
    # 1. Setup Directory
    out_dir = os.path.join(args.output_dir, "consistent_cells_subset")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Exporting consistent cell data to: {out_dir}")

    ind_col = 'ctpt_individual_prediction'
    cons_col = 'ctpt_consensus_prediction'
    comb_col = 'combined_annotation'
    
    if comb_col in adata.obs.columns:
        print("\n   [INFO] Generating exports based on FINAL REFINED annotation (combined_annotation)...")
        
        # Create a refined-specific subdirectory
        refined_out_dir = os.path.join(out_dir, "refined_annotation_exports")
        os.makedirs(refined_out_dir, exist_ok=True)
        
        # 1. UMAP colored by REFINED annotation (combined_annotation)
        try:
            with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
                sc.pl.umap(
                    adata, 
                    color=comb_col, 
                    palette=sc.pl.palettes.godsnot_102, 
                    legend_loc='right margin', 
                    legend_fontsize=8, 
                    title=f'All Cells - REFINED Annotation\n({adata.obs[comb_col].nunique()} types, n={len(adata)})', 
                    show=False, 
                    size=10
                )
            refined_all_umap_path = os.path.join(refined_out_dir, f"{args.final_run_prefix}_all_cells_REFINED_annotation_umap.png")
            _bold_right_margin_legend(refined_all_umap_path)
            plt.close()
            print(f"       -> Saved all cells REFINED annotation UMAP: {refined_all_umap_path}")
        except Exception as e:
            print(f"[WARNING] Could not generate refined annotation UMAP. Reason: {e}")
        
        # 2. Inconsistency UMAP (Individual vs REFINED)
        try:
            print("       -> Generating inconsistency UMAP (Individual vs REFINED)...")
            
            ind_labels = adata.obs[ind_col].astype(str)
            ref_labels = adata.obs[comb_col].astype(str)
            inconsist_mask_refined = (ind_labels != ref_labels)
            n_inconsist_refined = inconsist_mask_refined.sum()
            
            # Create temp column
            temp_col = '_temp_refined_inconsist'
            adata.obs[temp_col] = adata.obs[comb_col].astype(str)
            inconsist_label_ref = f'Inconsistent (n={n_inconsist_refined})'
            adata.obs.loc[inconsist_mask_refined, temp_col] = inconsist_label_ref
            adata.obs[temp_col] = adata.obs[temp_col].astype('category')
            
            # Build palette
            unique_cats = adata.obs[temp_col].cat.categories.tolist()
            pal_map = {}
            std_pal = sc.pl.palettes.godsnot_102
            cidx = 0
            for cat in unique_cats:
                if cat == inconsist_label_ref:
                    pal_map[cat] = '#bbbbbb'
                else:
                    pal_map[cat] = std_pal[cidx % len(std_pal)]
                    cidx += 1
            
            with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
                sc.pl.umap(
                    adata, 
                    color=temp_col, 
                    palette=pal_map,
                    legend_loc='right margin', 
                    legend_fontsize=8, 
                    title=f'REFINED Annotation - Inconsistent Cells Grey\n(Individual ≠ Refined: {n_inconsist_refined} cells, {n_inconsist_refined/len(adata)*100:.1f}%)', 
                    show=False, 
                    size=10
                )
            
            inconsist_ref_umap_path = os.path.join(refined_out_dir, f"{args.final_run_prefix}_REFINED_inconsistent_cells_grey_umap.png")
            _bold_right_margin_legend(inconsist_ref_umap_path)
            plt.close()
            print(f"       -> Saved REFINED inconsistency UMAP: {inconsist_ref_umap_path}")
            
            del adata.obs[temp_col]
            
        except Exception as e:
            print(f"[WARNING] Could not generate refined inconsistency UMAP. Reason: {e}")
        
        # 3. CSV with REFINED annotations
        refined_csv_path = os.path.join(refined_out_dir, f"{args.final_run_prefix}_all_cells_REFINED_annotations.csv")
        refined_export_df = pd.DataFrame({
            'cell_barcode': adata.obs_names,
            'individual_prediction': adata.obs[ind_col].values,
            'original_consensus': adata.obs[cons_col].values,
            'REFINED_annotation': adata.obs[comb_col].values,
            'individual_matches_refined': (adata.obs[ind_col].astype(str) == adata.obs[comb_col].astype(str)).values
        })
        if 'ctpt_confidence' in adata.obs.columns:
            refined_export_df['confidence'] = adata.obs['ctpt_confidence'].values
        if 'leiden' in adata.obs.columns:
            refined_export_df['leiden_cluster'] = adata.obs['leiden'].values
            
        refined_export_df.to_csv(refined_csv_path, index=False)
        print(f"       -> Saved REFINED annotations CSV: {refined_csv_path}")
        
        # 4. Cell type counts summary for REFINED
        refined_counts = adata.obs[comb_col].value_counts()
        refined_counts_df = pd.DataFrame({
            'Cell_Type': refined_counts.index,
            'Cell_Count': refined_counts.values,
            'Percentage': (refined_counts.values / len(adata) * 100).round(2)
        })
        refined_counts_path = os.path.join(refined_out_dir, f"{args.final_run_prefix}_REFINED_cell_type_counts.csv")
        refined_counts_df.to_csv(refined_counts_path, index=False)
        print(f"       -> Saved REFINED cell type counts: {refined_counts_path}")
        
        print(f"       ✅ All REFINED annotation exports saved to: {refined_out_dir}")

        # =========================================================================
        # === NEW: REFINED ANNOTATION DECONVOLUTION EXPORT ===
        # === (sc_counts_REFINED.csv, sc_labels_REFINED.csv, st_counts.csv) ===
        # =========================================================================
        print("\n   [INFO] Generating REFINED annotation-based Deconvolution reference files...")
        
        # Create subdirectory for refined deconvolution files
        refined_deconv_dir = os.path.join(refined_out_dir, "deconvolution_reference_REFINED")
        os.makedirs(refined_deconv_dir, exist_ok=True)
        
        try:
            # Determine which cells to use: consistent cells where Individual == Refined
            # This ensures high-quality reference for deconvolution
            ind_labels_ref = adata.obs[ind_col].astype(str)
            refined_labels_ref = adata.obs[comb_col].astype(str)
            refined_consistent_mask = (ind_labels_ref == refined_labels_ref)
            
            adata_refined_consistent = adata[refined_consistent_mask].copy()
            n_refined_consistent = len(adata_refined_consistent)
            
            print(f"       -> Using {n_refined_consistent} cells where Individual == Refined annotation")
            print(f"       -> ({n_refined_consistent/len(adata)*100:.2f}% of total cells)")
            
            # Apply min_cells_per_type filter if specified
            if args.min_cells_per_type is not None and args.min_cells_per_type > 0:
                print(f"       -> Applying cell type filter: min {args.min_cells_per_type} cells per REFINED type...")
                
                refined_type_counts = adata_refined_consistent.obs[comb_col].value_counts()
                types_to_keep_refined = refined_type_counts[refined_type_counts >= args.min_cells_per_type].index.tolist()
                types_removed_refined = refined_type_counts[refined_type_counts < args.min_cells_per_type].index.tolist()
                
                if types_removed_refined:
                    print(f"       -> Removing {len(types_removed_refined)} REFINED cell type(s) with < {args.min_cells_per_type} cells")
                    for removed_type in types_removed_refined:
                        print(f"          - {removed_type}: {refined_type_counts[removed_type]} cells")
                
                adata_refined_consistent = adata_refined_consistent[
                    adata_refined_consistent.obs[comb_col].isin(types_to_keep_refined)
                ].copy()
                
                # Clean up categories
                adata_refined_consistent.obs[comb_col] = adata_refined_consistent.obs[comb_col].cat.remove_unused_categories()
                
                print(f"       -> After filtering: {len(adata_refined_consistent)} cells, "
                      f"{adata_refined_consistent.obs[comb_col].nunique()} cell types")
            
            # Determine genes to export (intersect with ST if provided)
            if args.st_data_dir is not None:
                print(f"       -> Loading Spatial data from {args.st_data_dir} for gene intersection")
                adata_st_refined = load_expression_data(args.st_data_dir)
                refined_common_genes = sorted(list(
                    set(adata_refined_consistent.var_names) & set(adata_st_refined.var_names)
                ))
                print(f"       -> Gene intersection for REFINED export: {len(refined_common_genes)} genes")
            else:
                refined_common_genes = sorted(list(adata_refined_consistent.var_names))
                adata_st_refined = None
                print(f"       -> No ST data. Exporting all {len(refined_common_genes)} genes.")
            
            if len(refined_common_genes) > 0 and len(adata_refined_consistent) > 0:
                # 1. Export sc_counts.csv (Consistent Cells based on REFINED, Raw Counts)
                print("       -> Exporting 'sc_counts_REFINED.csv'...")
                if "counts" in adata_refined_consistent.layers:
                    X_sc_refined = adata_refined_consistent[:, refined_common_genes].layers["counts"]
                else:
                    X_sc_refined = adata_refined_consistent[:, refined_common_genes].X
                
                if hasattr(X_sc_refined, "toarray"):
                    X_sc_refined = X_sc_refined.toarray()
                
                pd.DataFrame(
                    X_sc_refined, 
                    index=adata_refined_consistent.obs_names, 
                    columns=refined_common_genes
                ).to_csv(os.path.join(refined_deconv_dir, "sc_counts.csv"))
                
                # 2. Export sc_labels.csv (Using REFINED/combined_annotation)
                print("       -> Exporting 'sc_labels_REFINED.csv' (using combined_annotation)...")
                df_labels_refined = adata_refined_consistent.obs[[comb_col]].copy()
                df_labels_refined.columns = ['CellType']
                df_labels_refined.to_csv(os.path.join(refined_deconv_dir, "sc_labels.csv"))
                
                # 3. Also export a comparison file showing both original and refined labels
                print("       -> Exporting 'sc_labels_REFINED_with_comparison.csv'...")
                df_labels_comparison = adata_refined_consistent.obs[[ind_col, cons_col, comb_col]].copy()
                df_labels_comparison.columns = ['Individual_Prediction', 'Original_Consensus', 'REFINED_Annotation']
                df_labels_comparison.to_csv(os.path.join(refined_deconv_dir, "sc_labels_REFINED_with_comparison.csv"))
                
                # 4. Export st_counts.csv (same as original, but in this directory for convenience)
                if adata_st_refined is not None:
                    print("       -> Exporting 'st_counts.csv' to REFINED directory...")
                    X_st_refined = adata_st_refined[:, refined_common_genes].X
                    if hasattr(X_st_refined, "toarray"):
                        X_st_refined = X_st_refined.toarray()
                    pd.DataFrame(
                        X_st_refined, 
                        index=adata_st_refined.obs_names, 
                        columns=refined_common_genes
                    ).to_csv(os.path.join(refined_deconv_dir, "st_counts.csv"))
                else:
                    print("       -> Skipping 'st_counts.csv' (no --st_data_dir provided).")
                
                # 5. Export summary statistics
                print("       -> Exporting 'REFINED_deconvolution_summary.txt'...")
                summary_lines_deconv = [
                    "="*60,
                    "REFINED ANNOTATION DECONVOLUTION REFERENCE SUMMARY",
                    "="*60,
                    f"Total cells in reference: {len(adata_refined_consistent)}",
                    f"Total genes in reference: {len(refined_common_genes)}",
                    f"Number of cell types: {adata_refined_consistent.obs[comb_col].nunique()}",
                    "",
                    "Cell type distribution:",
                    "-"*40
                ]
                for ct, count in adata_refined_consistent.obs[comb_col].value_counts().items():
                    summary_lines_deconv.append(f"  {ct}: {count} cells ({count/len(adata_refined_consistent)*100:.2f}%)")
                summary_lines_deconv.append("="*60)
                
                with open(os.path.join(refined_deconv_dir, "REFINED_deconvolution_summary.txt"), 'w') as f:
                    f.write('\n'.join(summary_lines_deconv))
                
                print(f"       ✅ Successfully exported REFINED Deconvolution reference files to: {refined_deconv_dir}")
            else:
                print("[WARNING] No genes or cells available for REFINED deconvolution export. Skipping.")
                
        except Exception as e:
            print(f"[ERROR] Failed to export REFINED Deconvolution files. Reason: {e}")
            import traceback
            traceback.print_exc()

    # 2. Check required columns exist
    if ind_col not in adata.obs or cons_col not in adata.obs:
        print(f"[ERROR] Required columns '{ind_col}' and/or '{cons_col}' not found in AnnData. Skipping export.")
        return

    # 3. Determine mode and create consistency mask
    # Convert all columns to string to avoid categorical mismatch issues
    ind_labels = adata.obs[ind_col].astype(str)
    cons_labels = adata.obs[cons_col].astype(str)
    
    if args.cas_refine_threshold is not None and comb_col in adata.obs.columns:
        # THREE-WAY CONSISTENCY CHECK (Refinement mode)
        mode_label = "Refinement_ThreeWay"
        comb_labels = adata.obs[comb_col].astype(str)
        
        print(f"       -> Mode: {mode_label}")
        print(f"       -> Requiring ALL THREE to match:")
        print(f"          1. '{ind_col}'")
        print(f"          2. '{cons_col}'")
        print(f"          3. '{comb_col}'")
        
        # All three must be identical
        mask = (ind_labels == cons_labels) & (cons_labels == comb_labels)
        
        # Report breakdown for transparency
        mask_ind_cons = (ind_labels == cons_labels)
        mask_cons_comb = (cons_labels == comb_labels)
        mask_ind_comb = (ind_labels == comb_labels)
        
        print(f"\n       -> Consistency Breakdown:")
        print(f"          Individual == Consensus:  {mask_ind_cons.sum()} cells ({mask_ind_cons.sum()/len(adata)*100:.2f}%)")
        print(f"          Consensus == Combined:    {mask_cons_comb.sum()} cells ({mask_cons_comb.sum()/len(adata)*100:.2f}%)")
        print(f"          Individual == Combined:   {mask_ind_comb.sum()} cells ({mask_ind_comb.sum()/len(adata)*100:.2f}%)")
        print(f"          ALL THREE match:          {mask.sum()} cells ({mask.sum()/len(adata)*100:.2f}%)")
        
        # For downstream use, use combined_annotation as the reference label
        final_label_col = comb_col
        
    else:
        # TWO-WAY CONSISTENCY CHECK (No refinement mode)
        mode_label = "No_Refinement_TwoWay"
        print(f"       -> Mode: {mode_label}")
        print(f"       -> Comparing '{ind_col}' vs '{cons_col}'")
        
        mask = (ind_labels == cons_labels)
        
        # For downstream use, use consensus as the reference label
        final_label_col = cons_col

    # 4. Filter for consistent cells
    adata_consistent = adata[mask].copy()
    
    n_total = len(adata)
    n_consistent = len(adata_consistent)
    print(f"\n       -> Found {n_consistent} fully consistent cells out of {n_total} ({n_consistent/n_total*100:.2f}%)")

    # Apply cell type filtering if threshold is specified
    adata_consistent_filtered = adata_consistent  # Default: no filtering
    if args.min_cells_per_type is not None and args.min_cells_per_type > 0:
        print(f"\n   [INFO] Applying cell type filter: min {args.min_cells_per_type} cells per type...")
        
        # Count cells per type
        cell_type_counts = adata_consistent.obs[cons_col].value_counts()
        print(f"       -> Cell type counts before filtering:")
        for ct, count in cell_type_counts.items():
            status = "✓ KEEP" if count >= args.min_cells_per_type else "✗ REMOVE"
            print(f"          {ct}: {count} cells [{status}]")
        
        # Identify types to keep
        types_to_keep = cell_type_counts[cell_type_counts >= args.min_cells_per_type].index.tolist()
        types_removed = cell_type_counts[cell_type_counts < args.min_cells_per_type].index.tolist()
        
        if types_removed:
            print(f"       -> Removing {len(types_removed)} cell type(s) with < {args.min_cells_per_type} cells: {types_removed}")
        
        # Filter the AnnData
        adata_consistent_filtered = adata_consistent[adata_consistent.obs[cons_col].isin(types_to_keep)].copy()
        # Clean up unused categories
        adata_consistent_filtered.obs[cons_col] = adata_consistent_filtered.obs[cons_col].cat.remove_unused_categories()
        
        n_filtered = len(adata_consistent_filtered)
        n_removed = n_consistent - n_filtered
        print(f"       -> After filtering: {n_filtered} cells ({n_removed} cells removed)")
        print(f"       -> Remaining cell types: {adata_consistent_filtered.obs[cons_col].nunique()}")
        
        # Generate filtered UMAP
        try:
            with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
                sc.pl.umap(
                    adata_consistent_filtered, 
                    color=cons_col, 
                    palette=sc.pl.palettes.godsnot_102, 
                    legend_loc='right margin', 
                    legend_fontsize=8, 
                    title=f'Consistent Cells (Filtered, min {args.min_cells_per_type} cells/type)\n(n={n_filtered}, {adata_consistent_filtered.obs[cons_col].nunique()} types)', 
                    show=False, 
                    size=10
                )
            filtered_umap_filename = f"{args.final_run_prefix}_consistent_cells_{mode_label}_filtered_umap.png"
            filtered_umap_path = os.path.join(out_dir, filtered_umap_filename)
            _bold_right_margin_legend(filtered_umap_path)
            plt.close()
            print(f"       -> Saved filtered consistent cells UMAP: {filtered_umap_path}")
        except Exception as e:
            print(f"[WARNING] Could not generate filtered consistent cells UMAP. Reason: {e}")
        
        # Save filtered CSV
        filtered_csv_filename = f"{args.final_run_prefix}_consistent_cells_{mode_label}_filtered.csv"
        filtered_csv_path = os.path.join(out_dir, filtered_csv_filename)
        adata_consistent_filtered.obs.to_csv(filtered_csv_path)
        print(f"       -> Saved filtered annotations CSV: {filtered_csv_path}")

        # =========================================================================
        print("\n   [INFO] Generating dotplots for filtered consistent cells...")
        
        # Standard dotplot
        dotplot_path = generate_consistent_cells_dotplot(
            adata_filtered=adata_consistent_filtered,
            output_dir=out_dir,
            prefix=args.final_run_prefix,
            annotation_col=cons_col,
            n_top_genes=getattr(args, 'n_top_genes', 5),
            dpi=getattr(args, 'fig_dpi', 300)
        )
        
        # Categorized dotplot with dendrogram
        cat_dotplot_path = generate_consistent_cells_dotplot_by_category(
            adata_filtered=adata_consistent_filtered,
            output_dir=out_dir,
            prefix=args.final_run_prefix,
            annotation_col=cons_col,
            n_top_genes=3,
            dpi=getattr(args, 'fig_dpi', 300)
        )
        
        # =========================================================================
        if MARKER_PRIOR_DICT:
            print(f"\n   [INFO] Running marker-based annotation on filtered consistent cells...")
            print(f"          (This UMAP will be directly comparable to {filtered_umap_filename.replace('.csv', '_umap.png')})")
            
            try:
                marker_annot_result = annotate_filtered_consistent_cells_by_markers(
                    adata_consistent_filtered=adata_consistent_filtered,
                    prior_dict=MARKER_PRIOR_DICT,
                    output_dir=out_dir,
                    prefix=args.final_run_prefix,
                    annotation_col=cons_col,
                    n_top_genes=50,
                    min_overlap_score=0.05,
                    species=getattr(args, '_detected_species', 'human'),
                    show_top_n_matches=5
                )
                
                if marker_annot_result:
                    print(f"       ✅ Marker-based annotation complete for filtered consistent cells")
                    # Update adata_consistent_filtered with new annotation
                    adata_consistent_filtered = marker_annot_result['adata']
            except Exception as e:
                print(f"[WARNING] Marker-based annotation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n   [INFO] Skipping marker-based annotation (no MARKER_PRIOR_DICT available)")

    # 4. Save CSV
    csv_filename = f"{args.final_run_prefix}_consistent_cells_{mode_label}.csv"
    csv_path = os.path.join(out_dir, csv_filename)
    adata_consistent.obs.to_csv(csv_path)
    print(f"       -> Saved filtered annotations CSV: {csv_path}")

    # 5. Generate and Save UMAP (Consistent Subset Only)
    try:
        with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
            sc.pl.umap(
                adata_consistent, 
                color=cons_col, 
                palette=sc.pl.palettes.godsnot_102, 
                legend_loc='right margin', 
                legend_fontsize=8, 
                title=f'Consistent Cells Only ({mode_label})\n(n={n_consistent})', 
                show=False, 
                size=10
            )
        umap_filename = f"{args.final_run_prefix}_consistent_cells_{mode_label}_umap.png"
        umap_path = os.path.join(out_dir, umap_filename)
        _bold_right_margin_legend(umap_path)
        plt.close()
        print(f"       -> Saved consistent cells UMAP: {umap_path}")
    except Exception as e:
        print(f"[WARNING] Could not generate consistent cells UMAP. Reason: {e}")

    # =========================================================================
    # === NEW ADDITION: Consistency Context UMAP (Grey out inconsistent) ===
    # =========================================================================
    try:
        print(f"       -> Generating Consistency Context UMAP (All cells, inconsistent in grey)...")
        
        # Create a temporary column on the FULL adata object for visualization
        context_plot_col = 'consistency_context'
        adata.obs[context_plot_col] = adata.obs[cons_col].astype(str)
        
        # Identify inconsistent cells (where mask is False) and label them
        inconsistent_label = "Inconsistent/Mismatch"
        adata.obs.loc[~mask, context_plot_col] = inconsistent_label
        adata.obs[context_plot_col] = adata.obs[context_plot_col].astype('category')
        
        # Build a color palette: Use standard colors for types, GREY for inconsistent
        unique_labels = adata.obs[context_plot_col].cat.categories.tolist()
        palette_map = {}
        std_palette = sc.pl.palettes.godsnot_102
        
        color_idx = 0
        for label in unique_labels:
            if label == inconsistent_label:
                palette_map[label] = '#bbbbbb' # Medium Grey
            else:
                # Assign colors from the standard palette
                palette_map[label] = std_palette[color_idx % len(std_palette)]
                color_idx += 1
        
        with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'}):
            sc.pl.umap(
                adata, 
                color=context_plot_col, 
                palette=palette_map, 
                legend_loc='right margin', 
                legend_fontsize=8, 
                title=f'Consistency Overview ({mode_label})\n(Grey = Individual != Consensus)', 
                show=False, 
                size=10
            )
        
        context_umap_filename = f"{args.final_run_prefix}_consistency_context_all_cells_umap.png"
        context_umap_path = os.path.join(out_dir, context_umap_filename)
        _bold_right_margin_legend(context_umap_path)
        plt.close()
        print(f"       -> Saved consistency context UMAP: {context_umap_path}")
        
        # Clean up temporary column
        del adata.obs[context_plot_col]

    except Exception as e:
        print(f"[WARNING] Could not generate consistency context UMAP. Reason: {e}")
    
    # =========================================================================
    # === DECONVOLUTION EXPORT (sc_counts, sc_labels, st_counts) ===
    # =========================================================================
    print("\n   [INFO] Generating scRNA-seq reference files for Deconvolution")
    try:
        # Determine genes to export
        if args.st_data_dir is not None:
            # If ST data provided, intersect genes
            print(f"       -> Loading Spatial data from {args.st_data_dir} for gene intersection")
            adata_st = load_expression_data(args.st_data_dir)
            
            # === MODIFICATION: Use filtered data for gene intersection ===
            final_common_genes = sorted(list(set(adata_consistent_filtered.var_names) & set(adata_st.var_names)))
            print(f"       -> Final intersection for export: {len(final_common_genes)} genes")
        else:
            # No ST data - use all genes from consistent subset
            # === MODIFICATION: Use filtered data ===
            final_common_genes = sorted(list(adata_consistent_filtered.var_names))
            adata_st = None
            print(f"       -> No ST data provided. Exporting all {len(final_common_genes)} genes from scRNA-seq.")

        if len(final_common_genes) > 0:
            # 1. Export sc_counts.csv (Consistent Cells, Raw Counts)
            # Use layers['counts'] if available, else X. Ensure dense format for CSV.
            print("       -> Exporting 'sc_counts.csv'...")
            # === MODIFICATION: Use filtered data ===
            if "counts" in adata_consistent_filtered.layers:
                X_sc = adata_consistent_filtered[:, final_common_genes].layers["counts"]
            else:
                X_sc = adata_consistent_filtered[:, final_common_genes].X
            
            if hasattr(X_sc, "toarray"): X_sc = X_sc.toarray()
            # === MODIFICATION: Use filtered data ===
            pd.DataFrame(X_sc, index=adata_consistent_filtered.obs_names, columns=final_common_genes).to_csv(os.path.join(out_dir, "sc_counts.csv"))

            # 2. Export sc_labels.csv (Annotations)
            print("       -> Exporting 'sc_labels.csv'...")
            # Use the column determined earlier (cons_col)
            # === MODIFICATION: Use filtered data ===
            df_labels = adata_consistent_filtered.obs[[cons_col]].copy()
            df_labels.columns = ['CellType']
            df_labels.to_csv(os.path.join(out_dir, "sc_labels.csv"))

            # 3. Export st_counts.csv (Only if ST data was provided)
            if adata_st is not None:
                print("       -> Exporting 'st_counts.csv'...")
                X_st = adata_st[:, final_common_genes].X
                if hasattr(X_st, "toarray"): X_st = X_st.toarray()
                pd.DataFrame(X_st, index=adata_st.obs_names, columns=final_common_genes).to_csv(os.path.join(out_dir, "st_counts.csv"))
            else:
                print("       -> Skipping 'st_counts.csv' (no --st_data_dir provided).")
            
            print(f"       ✅ Successfully exported Deconvolution reference files to {out_dir}")
        else:
            print("[ERROR] No genes available for export. Skipping.")

    except Exception as e:
        print(f"[ERROR] Failed to export Deconvolution files. Reason: {e}")

def main(parsed_args):
    """Main orchestrator for the two-stage pipeline."""
    adata_s2, cas_csv_path_s2 = None, None

    # --- STAGE 1 ---
    print("="*80 + "\n### STARTING STAGE 1: BAYESIAN PARAMETER OPTIMIZATION ###\n" + "="*80)
    stage1_output_dir = os.path.join(parsed_args.output_dir, "stage_1_bayesian_optimization")
    
    original_output_dir = parsed_args.output_dir
    parsed_args.output_dir = stage1_output_dir
    
    optimization_results = run_stage_one_optimization(parsed_args, adata_input=None)
    optimal_params = optimization_results.get("params")
    
    parsed_args.output_dir = original_output_dir 

    # --- STAGE 2 (Conditional) ---
    if optimal_params:
        print("\n\n" + "="*80 + "\n### STARTING STAGE 2: FINAL ANALYSIS ###\n" + "="*80)
        stage2_output_dir = os.path.join(parsed_args.output_dir, "stage_2_final_analysis")
        os.makedirs(stage2_output_dir, exist_ok=True)
        print(f"Stage 2 outputs will be saved to: {os.path.abspath(stage2_output_dir)}")
        
        if parsed_args.data_dir:
            adata_s2, cas_csv_path_s2 = run_stage_two_final_analysis(
                cli_args=parsed_args, optimal_params=optimal_params, output_dir=stage2_output_dir, data_dir=parsed_args.data_dir
            )
        elif parsed_args.multi_sample:
            wt_path, treated_path = parsed_args.multi_sample
            adata_s2, cas_csv_path_s2 = run_stage_two_final_analysis_multi_sample(
                cli_args=parsed_args, optimal_params=optimal_params, output_dir=stage2_output_dir, wt_path=wt_path, treated_path=treated_path
            )
    else:
        print("\n\n" + "="*80 + "\n### SKIPPING STAGE 2 ###\nStage 1 did not complete successfully.\n" + "="*80)
        print("\n--- Integrated pipeline finished with errors. ---")
        return
    # --- CELL TYPE MATCHING DIAGNOSTICS ---
    if adata_s2 is not None and MARKER_PRIOR_DICT:
        print("\n" + "="*80 + "\n### CELL TYPE MATCHING DIAGNOSTICS ###\n" + "="*80)
        stage2_output_dir = os.path.join(parsed_args.output_dir, "stage_2_final_analysis")
        diag_path = os.path.join(stage2_output_dir, f"{parsed_args.final_run_prefix}_celltype_matching_diagnostics.csv")
        
        matching_diagnostics = diagnose_celltype_matching(
            adata=adata_s2,
            prior_dict=MARKER_PRIOR_DICT,
            annotation_col='ctpt_consensus_prediction',
            output_path=diag_path
        )

        # Export detailed cell type marker information
        print("\n" + "="*80 + "\n### EXPORTING CELL TYPE MARKER DETAILS ###\n" + "="*80)
        
        marker_details_dir = os.path.join(stage2_output_dir, "celltype_marker_details")
        
        exported_marker_files = export_celltype_marker_details(
            adata=adata_s2,
            prior_dict=MARKER_PRIOR_DICT,
            output_dir=marker_details_dir,
            prefix=parsed_args.final_run_prefix,
            groupby_key='ctpt_consensus_prediction',
            n_top_genes=50,  # Export top 50 markers per cell type
            deg_ranking_method=getattr(parsed_args, 'deg_ranking_method', 'original'),
            deg_weight_fc=getattr(parsed_args, 'deg_weight_fc', 0.4),
            deg_weight_expr=getattr(parsed_args, 'deg_weight_expr', 0.3),
            deg_weight_pct=getattr(parsed_args, 'deg_weight_pct', 0.3),
            species=getattr(parsed_args, '_detected_species', 'human')
        )
        
        if exported_marker_files:
            print(f"\n    Marker detail files exported:")
            for file_type, file_path in exported_marker_files.items():
                print(f"       - {file_type}: {file_path}")
    
    if adata_s2 is not None and MARKER_PRIOR_DICT:
        print("\n" + "="*80 + "\n### MARKER-BASED AUTOMATIC ANNOTATION ###\n" + "="*80)
        
        stage2_output_dir = os.path.join(parsed_args.output_dir, "stage_2_final_analysis")
        
        # Determine DEG ranking method from arguments
        deg_method = getattr(parsed_args, 'deg_ranking_method', 'original')
        
        # =========================================================================
        # MODIFICATION: Use final consensus cell type clusters instead of Leiden
        # The 'ctpt_consensus_prediction' column contains the final consensus 
        # annotations after CellTypist prediction and majority voting.
        # This ensures marker-based annotation is performed on biologically
        # meaningful cell type clusters rather than raw Leiden clusters.
        # =========================================================================
        marker_annot_result = annotate_celltypes_by_marker_overlap(
            adata=adata_s2,
            prior_dict=MARKER_PRIOR_DICT,
            output_dir=parsed_args.output_dir,
            prefix=parsed_args.final_run_prefix,
            groupby_key='ctpt_consensus_prediction',  # CHANGED: Use final consensus clusters
            n_top_genes=50,
            min_overlap_score=0.05,
            deg_ranking_method=deg_method,
            deg_weight_fc=getattr(parsed_args, 'deg_weight_fc', 0.4),
            deg_weight_expr=getattr(parsed_args, 'deg_weight_expr', 0.3),
            deg_weight_pct=getattr(parsed_args, 'deg_weight_pct', 0.3),
            species=getattr(parsed_args, '_detected_species', 'human'),
            show_top_n_matches=5
        )
        
        if marker_annot_result:
            adata_s2 = marker_annot_result['adata']
            print(f"\n    ✅ Added 'marker_based_annotation' column to adata")
            
            # Optionally run re-annotation for low-confidence clusters
            if parsed_args.cas_refine_threshold is not None and cas_csv_path_s2:
                print("\n    Running marker-based re-annotation for low-confidence clusters...")
                
                reannot_result = run_marker_based_reannotation_for_low_confidence(
                    adata=adata_s2,
                    prior_dict=MARKER_PRIOR_DICT,
                    output_dir=stage2_output_dir,
                    prefix=parsed_args.final_run_prefix,
                    cas_csv_path=cas_csv_path_s2,
                    cas_threshold=parsed_args.cas_refine_threshold,
                    cas_aggregation_method=parsed_args.cas_aggregation_method,
                    n_top_genes=50,
                    deg_ranking_method=deg_method,
                    deg_weight_fc=getattr(parsed_args, 'deg_weight_fc', 0.4),
                    deg_weight_expr=getattr(parsed_args, 'deg_weight_expr', 0.3),
                    deg_weight_pct=getattr(parsed_args, 'deg_weight_pct', 0.3),
                    species=getattr(parsed_args, '_detected_species', 'human')
                )
                
                if reannot_result:
                    adata_s2 = reannot_result['adata']

    # --- STAGE 3 & 4 (OPTIONAL REFINEMENT) ---
    if parsed_args.cas_refine_threshold is not None:
        if adata_s2 is not None and cas_csv_path_s2 is not None and os.path.exists(cas_csv_path_s2):
            run_iterative_refinement_pipeline(
                args=parsed_args, adata_s2=adata_s2, cas_csv_path_s2=cas_csv_path_s2
            )
        else:
            print(f"[WARNING] --cas_refine_threshold was set, but Stage 2 did not produce the necessary outputs to proceed. Skipping refinement.")

    # --- NEW: EXPORT CONSISTENT SUBSET ---
    # This runs regardless of whether refinement was active or not, 
    # but adapts its logic based on the presence of the 'combined_annotation' column.
    if adata_s2 is not None:
        export_consistent_cells(parsed_args, adata_s2)

    print("\n--- Integrated pipeline finished successfully! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Integrated Two-Stage Bayesian Optimization and Final Analysis Pipeline for scRNA-seq.", formatter_class=argparse.RawTextHelpFormatter)

    stage1_group = parser.add_argument_group('Stage 1 & 2: Main I/O and Mode')
    mode_group = stage1_group.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--data_dir', type=str, 
        help='Path to expression data for single-sample analysis.\n'
             'Accepts: 10x directory (with matrix.mtx), .h5 file, or .h5ad file.')
    mode_group.add_argument('--multi_sample', nargs=2, metavar=('WT_PATH', 'TREATED_PATH'), 
        help='Two paths for WT/Control and Treated/Perturbed data for multi-sample integration.\n'
             'Each path can be: 10x directory, .h5 file, or .h5ad file.')
    stage1_group.add_argument('--output_dir', type=str, required=True, help='Path for all output files.')
    stage1_group.add_argument('--model_path', type=str, required=True, help='Path to CellTypist model (.pkl).')
    stage1_group.add_argument('--output_prefix', type=str, default='bayesian_opt', help='Base prefix for Stage 1 output files.')
    stage1_group.add_argument('--st_data_dir', type=str, default=None, 
        help='Path to Spatial Transcriptomics data (for gene intersection and Deconvolution export).\n'
             'Accepts: 10x directory, .h5 file, or .h5ad file.')
    opt_group = parser.add_argument_group('Stage 1: Optimization Parameters')
    opt_group.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility.')
    opt_group.add_argument('--n_calls', type=int, default=50, help='Number of trials for EACH of the three optimization strategies.')
    opt_group.add_argument(
        '--model_type',
        type=str,
        default='biological',
        choices=['biological', 'structural', 'silhouette'],
        help= ("'biological'(default): balances CAS & MCS.\n"
               "'structural' : adds silhouette score to balance biological concordance with cluster quality.\n"
               "'silhouette': optimizes solely to maximize the silhouette score.")
    )
    opt_group.add_argument('--marker_gene_model', type=str, default='non-mitochondrial', choices=['all', 'non-mitochondrial'], help="'all': use all genes. 'non-mitochondrial' (default): exclude mitochondrial genes from MCS markers.")
    opt_group.add_argument('--target', type=str, default='all', choices=['all', 'weighted_cas', 'simple_cas', 'mcs'], help="'all' (default): runs a single, balanced optimization. Other options optimize for that specific metric.")
    
    opt_group.add_argument(
        '--cas_aggregation_method',
        type=str,
        default='leiden',
        choices=['leiden', 'consensus'],
        help=("Method for calculating Simple Mean CAS and for determining refinement candidates.\n"
              "'leiden' (default): Averages the purity of each individual Leiden cluster.\n"
              "'consensus': Merges Leiden clusters with the same consensus label, then averages their purity.")
    )

    hvg_group = parser.add_argument_group('Stage 1 & 2: HVG Selection Method')
    hvg_group.add_argument('--hvg_min_mean', type=float, default=None, help='(Optional) Activates two-step HVG selection. Min mean for initial filtering.')
    hvg_group.add_argument('--hvg_max_mean', type=float, default=None, help='(Optional) Activates two-step HVG selection. Max mean for initial filtering.')
    hvg_group.add_argument('--hvg_min_disp', type=float, default=None, help='(Optional) Activates two-step HVG selection. Min dispersion for initial filtering.')
    integration_group = parser.add_argument_group('Stage 1 & 2: Batch Integration Options')
    integration_group.add_argument(
        '--batch_key', 
        type=str, 
        default=None, 
        help='(Optional) Column name in adata.obs containing batch/sample information.\n'
             'If not specified, the script will auto-detect from:\n'
             '  1. Common metadata columns (batch, sample, orig.ident, etc.)\n'
             '  2. Barcode suffixes (e.g., BARCODE_Br2720)'
    )
    integration_group.add_argument(
        '--no_integration',
        action='store_true',
        default=False,
        help='(Optional) Force single-sample mode even if batches are detected.\n'
             'Skips Harmony integration.'
    )
    integration_group.add_argument(
        '--integration_method',
        type=str,
        default='harmony',
        choices=['harmony', 'none'],
        help="Integration method to use when batches are detected.\n"
             "'harmony' (default): Use Harmony for batch correction.\n"
             "'none': Skip integration (equivalent to --no_integration)."
    )
    qc_group = parser.add_argument_group('Stage 1 & 2: QC & Filtering Parameters')
    qc_group.add_argument('--min_genes', type=int, default=200, help='Min genes per cell.')
    qc_group.add_argument('--max_genes', type=int, default=7000, help='Max genes per cell.')
    qc_group.add_argument('--max_pct_mt', type=float, default=10.0, help='Max mitochondrial percentage.')
    qc_group.add_argument('--min_cells', type=int, default=3, help='Min cells per gene.')

    stage2_group = parser.add_argument_group('Stage 2 & Optional Refinement: Final Run Parameters')
    stage2_group.add_argument('--final_run_prefix', type=str, default='sc_analysis_repro', help='Prefix for all output files in the Stage 2 subdirectory.')
    stage2_group.add_argument('--fig_dpi', default=500, type=int, help='Resolution (DPI) for saved figures in Stage 2.')
    stage2_group.add_argument('--n_pcs_compute', type=int, default=105, help="Number of principal components to COMPUTE in Stage 1 and 2.")
    stage2_group.add_argument('--n_top_genes', type=int, default=5, help="Number of top marker genes to show in plots/tables in Stage 1 and 2.")
    stage2_group.add_argument('--cellmarker_db', type=str, default=None, help="(Optional) Path to a cell marker database (.csv) for manual annotation in Stage 2.")
    stage2_group.add_argument('--n_degs_for_capture', type=int, default=5, help="Number of top DEGs per cluster to use for the Marker Capture Score calculation in Stage 2.")
    stage2_group.add_argument('--cas_refine_threshold', type=float, default=None, help="(Optional) CAS percentage threshold (0-100). If a cluster's CAS is below this, its cells are pooled for a second, refined optimization run.")
    stage2_group.add_argument('--refinement_depth', type=int, default=1, help="(Optional) Maximum number of times to repeat the refinement process on failing cells. Default is 1.")
    stage2_group.add_argument('--min_cells_refinement', type=int, default=100, help="(Optional) Minimum number of failing cells required to trigger a refinement loop. Default is 100.")
    stage2_group.add_argument('--min_cells_per_type', type=int, default=None, 
            help="(Optional) Minimum number of cells required per cell type in the consistent cells export.\n"
                "Cell types with fewer cells will be excluded from sc_counts.csv, sc_labels.csv, and a filtered UMAP will be generated.")
    marker_prior_group = parser.add_argument_group('Marker Prior Score (MPS) Options')
    marker_prior_group.add_argument(
        '--marker_prior_db',
        type=str,
        default=None,
        help="(Optional) Path to external marker gene database CSV.\n"
             "Expected columns: species, organ, cell_type, marker_genes, gene_count\n"
             "The 'marker_genes' column should contain semicolon-separated gene symbols.\n"
             "Example: combined_markers_summary.csv"
    )
    marker_prior_group.add_argument(
        '--marker_prior_species',
        type=str,
        default='Human',
        help="Species filter for marker prior database (default: 'Human')."
    )
    marker_prior_group.add_argument(
        '--marker_prior_organ',
        type=str,
        default=None,
        help="(Optional) Organ/tissue filter for marker prior database.\n"
             "Example: 'Adipose' will match 'Adipose Tissue', 'Abdominal Adipose Tissue', etc."
    )
    marker_prior_group.add_argument(
        '--mps_weight',
        type=float,
        default=1.0,
        help="[DEPRECATED] Use --mps_bonus_weight instead.\n"
             "Weight for MPS in balanced score calculation (default: 1.0).\n"
             "Set to 0 to disable MPS contribution while still calculating it."
    )
    marker_prior_group.add_argument(
        '--mps_bonus_weight',
        type=float,
        default=0.2,
        help="Maximum bonus that MPS can add to the base score (default: 0.2 = 20%%).\n"
             "Uses ADDITIVE BONUS SYSTEM:\n"
             "  Final Score = Base Score + (mps_bonus_weight × MPS)\n"
             "\n"
             "Examples with default 0.2:\n"
             "  - MPS=0%%   -> +0%%  bonus\n"
             "  - MPS=50%%  -> +10%% bonus\n"
             "  - MPS=100%% -> +20%% bonus\n"
             "\n"
             "Set to 0 to disable MPS bonus while still calculating MPS."
    )
    marker_prior_group.add_argument(
        '--n_degs_for_mps',
        type=int,
        default=50,
        help="Number of top DEGs to consider when calculating MPS recall (default: 50)."
    )
    marker_prior_group.add_argument(
        '--protect_canonical_markers',
        action='store_true',
        default=False,
        help="If set, ensures canonical markers from the prior database are included\n"
            "in HVG selection even if they have low variance. Up to 10%% of n_hvg\n"
            "slots may be reserved for canonical markers."
    )

    marker_prior_group.add_argument(
        '--penalize_unmatched_clusters',
        action='store_true',
        default=True,
        help="If True (default), clusters with no match in the marker prior database\n"
            "receive MPS=0 and ARE included in the mean MPS calculation.\n"
            "If False, unmatched clusters are excluded from MPS averaging."
    )
    marker_prior_group.add_argument(
        '--deg_ranking_method',
        type=str,
        default='original',
        choices=['original', 'composite'],
        help="Method for ranking DEGs in MPS calculation.\n"
             "'original' (default): Rank by log2FC only (Scanpy default).\n"
             "'composite': Rank by weighted combination of log2FC, expression, and pct_diff."
    )
    marker_prior_group.add_argument(
        '--deg_weight_fc',
        type=float,
        default=0.4,
        help="Weight for log2 fold change in composite DEG ranking (default: 0.4)."
    )
    marker_prior_group.add_argument(
        '--deg_weight_expr',
        type=float,
        default=0.3,
        help="Weight for mean expression in composite DEG ranking (default: 0.3)."
    )
    marker_prior_group.add_argument(
        '--deg_weight_pct',
        type=float,
        default=0.3,
        help="Weight for pct difference (pct.1 - pct.2) in composite DEG ranking (default: 0.3)."
    )
    marker_prior_group.add_argument(
        '--mps_similarity_threshold',
        type=float,
        default=0.6,
        help="Minimum similarity score for fuzzy cell type matching (0-1). Default: 0.6"
    )

    marker_prior_group.add_argument(
        '--mps_verbose_matching',
        action='store_true',
        default=False,
        help="Print detailed cell type matching information for debugging."
    )
    marker_prior_group.add_argument(
        '--mps_min_cells_per_group',
        type=int,
        default=5,
        help="Minimum number of cells required in a cluster to calculate MPS.\n"
            "Clusters with fewer cells will receive MPS=0. Default: 5."
    )
    parsed_args = parser.parse_args()

    if parsed_args.multi_sample and "harmony" not in parsed_args.output_prefix:
        parsed_args.output_prefix += "_harmony"
    
    # Call the main orchestrator function
    main(parsed_args)