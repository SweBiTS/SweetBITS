"""
sweetbits.taxmath
High-performance, vectorized math operations for taxonomy data.
"""

import numpy as np
import polars as pl
from joltax import JolTree
from sweetbits.utils import UNCLASSIFIED_TID

def calc_clade_sum(
    df: pl.DataFrame, 
    tree: JolTree, 
    min_reads: int = 0, 
    min_observed: int = 0
) -> pl.DataFrame:
    """
    Computes hierarchical clade counts dynamically using the 'Recursive Failing-Node Push-Up' algorithm.
    
    This function processes the taxonomic tree layer-by-layer from leaves to root.
    At each layer, nodes evaluate their clade counts against quality thresholds. 
    Surviving nodes are retained. 
    Failed nodes are purged, and their direct `taxon_reads` are pushed up to their 
    parent's `taxon_reads`, ensuring 100% data retention (mass balance) while solving
    database fragmentation bias.

    Args:
        df           : Long-format DataFrame with 't_id', 'sample_id', and 'taxon_reads'.
        tree         : The loaded JolTree taxonomy cache.
        min_reads    : Minimum maximum read count across all samples to survive.
        min_observed : Minimum number of non-zero samples to survive.

    Returns:
        A new DataFrame with updated 'taxon_reads' (orphans pushed up) and new 'clade_reads'.
    """
    # 1. Pivot the input data into a dense matrix (t_id x samples)
    matrix_df = df.pivot(
        values="taxon_reads", 
        index="t_id", 
        on="sample_id", 
        aggregate_function="sum"
    ).fill_null(0)
    
    sample_cols = [c for c in matrix_df.columns if c != "t_id"]
    if not sample_cols:
        return df.with_columns(pl.col("taxon_reads").alias("clade_reads"))

    # Extract initial IDs
    initial_tids = matrix_df["t_id"].to_numpy()
    
    # 1.5 Expand Matrix to Include All Ancestors
    # In sparse datasets (only taxon_reads > 0 are present), internal nodes might be missing.
    # We must add them with 0 reads so they can receive "votes" from their children.
    initial_indices = tree._get_indices(initial_tids)
    valid_initial = initial_indices[initial_indices != -1]
    
    # Fast vectorized ancestor discovery
    num_total_nodes = len(tree._index_to_id)
    is_active = np.zeros(num_total_nodes, dtype=bool)
    is_active[valid_initial] = True
    
    current_layer = valid_initial
    while len(current_layer) > 0:
        parents = tree.parents[current_layer]
        valid_parents = parents[(parents != -1) & (parents != current_layer)]
        new_parents = valid_parents[~is_active[valid_parents]]
        if len(new_parents) == 0:
            break
        is_active[new_parents] = True
        current_layer = new_parents
        
    all_active_indices = np.where(is_active)[0]
    all_active_tids = np.array([int(tree._index_to_id[idx]) for idx in all_active_indices], dtype=np.uint32)
    
    # Find which tids are new and need to be appended with 0s
    existing_set = set(initial_tids)
    new_tids = [tid for tid in all_active_tids if tid not in existing_set]
    
    if new_tids:
        # Create a zero-filled dataframe for missing ancestors
        zero_data = {"t_id": new_tids}
        for col in sample_cols:
            zero_data[col] = [0] * len(new_tids)
        zero_df = pl.DataFrame(zero_data, schema=matrix_df.schema)
        matrix_df = pl.concat([matrix_df, zero_df])

    # Extract NumPy arrays for fast processing (use .copy() to ensure writability)
    tids = matrix_df["t_id"].to_numpy()
    taxon_reads = matrix_df.select(sample_cols).to_numpy().copy()
    clade_reads = taxon_reads.copy()
    
    # Map input TaxIDs to internal tree indices
    indices = tree._get_indices(tids)
    
    # Handle taxa not found in the tree (e.g., Unclassified, outdated IDs)
    valid_mask = indices != -1
    valid_indices = indices[valid_mask]
    
    # Reverse lookup to map tree operations back to matrix rows
    tree_to_matrix = np.full(num_total_nodes, -1, dtype=np.int32)
    tree_to_matrix[valid_indices] = np.where(valid_mask)[0]
    
    # 2. The Failing-Node Push-Up Algorithm
    max_depth = int(np.max(tree.depths)) if len(tree.depths) > 0 else 0
    
    # Root node is exempt from filtering to act as the final dustbin
    root_idx = tree._get_indices(np.array([1]))[0]
    
    for d in range(max_depth, -1, -1):
        # Find all valid matrix rows corresponding to nodes exactly at the current depth
        active_matrix_rows = np.where(valid_mask & (tree.depths[indices] == d))[0]
        if len(active_matrix_rows) == 0:
            continue
            
        tree_indices_for_layer = indices[active_matrix_rows]
        
        # Step A: ALL nodes at this depth pass their current clade_reads up to their parent's clade_reads.
        # This mathematically ensures that parent clade_reads are perfectly cumulative and stable.
        parents = tree.parents[tree_indices_for_layer]
        valid_parents_mask = (parents != -1) & (parents != tree_indices_for_layer)
        
        target_parents = parents[valid_parents_mask]
        source_rows_for_vote = active_matrix_rows[valid_parents_mask]
        target_matrix_rows = tree_to_matrix[target_parents]
        
        valid_targets_mask = target_matrix_rows != -1
        final_targets = target_matrix_rows[valid_targets_mask]
        final_sources = source_rows_for_vote[valid_targets_mask]
        
        if len(final_targets) > 0:
            np.add.at(clade_reads, final_targets, clade_reads[final_sources])
            
        # Step B: Evaluate Survival based on the node's fully formed clade_reads
        layer_clade = clade_reads[active_matrix_rows, :]
        max_reads = np.max(layer_clade, axis=1)
        observed = np.sum(layer_clade > 0, axis=1)
        
        is_root = (tree_indices_for_layer == root_idx)
        survivors = (max_reads >= min_reads) & (observed >= min_observed) | is_root
        
        failed_rows = active_matrix_rows[~survivors]
        
        # Step C: The Push-Up. Failed nodes surrender their taxon_reads to their parents.
        if len(failed_rows) > 0:
            failed_tree_indices = indices[failed_rows]
            f_parents = tree.parents[failed_tree_indices]
            
            # Map parents to matrix rows
            f_target_matrix_rows = tree_to_matrix[f_parents]
            
            # A parent is only valid to receive pushed reads if it is not -1, not self, 
            # AND physically exists in the active dataset matrix.
            can_push_mask = (
                (f_parents != -1) & 
                (f_parents != failed_tree_indices) & 
                (f_target_matrix_rows != -1)
            )
            
            # Nodes that successfully found a parent in the dataset to push to
            pushable_rows = failed_rows[can_push_mask]
            target_matrix_rows_for_push = f_target_matrix_rows[can_push_mask]
            
            if len(pushable_rows) > 0:
                np.add.at(taxon_reads, target_matrix_rows_for_push, taxon_reads[pushable_rows])
                
                # Only erase the node from the matrix if it successfully transferred its mass.
                # Nodes that fail but have no valid parent (e.g., they are the top of a --clade filter)
                # act as "Effective Roots" and survive by default to preserve mass balance.
                taxon_reads[pushable_rows, :] = 0
                clade_reads[pushable_rows, :] = 0

    # 3. Reconstruct the Dataframe directly to long format
    # We must keep cells where clade_reads > 0 OR it's the unclassified node 
    unclass_row_mask = tids == UNCLASSIFIED_TID
    
    keep_mask = (clade_reads > 0)
    
    if np.any(unclass_row_mask):
        keep_mask[unclass_row_mask, :] = True
        
    flat_taxon = taxon_reads[keep_mask]
    flat_clade = clade_reads[keep_mask]
    
    row_idx, col_idx = np.where(keep_mask)
    tids_filtered = tids[row_idx]
    
    samples_arr = np.array(sample_cols)
    samples_filtered = samples_arr[col_idx]
    
    result_df = pl.DataFrame({
        "t_id": pl.Series(tids_filtered, dtype=matrix_df.schema["t_id"]),
        "sample_id": pl.Series(samples_filtered, dtype=pl.Categorical),
        "taxon_reads": pl.Series(flat_taxon, dtype=pl.UInt32),
        "clade_reads": pl.Series(flat_clade, dtype=pl.UInt32)
    })
    
    return result_df
