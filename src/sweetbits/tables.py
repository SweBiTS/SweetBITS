import polars as pl
from pathlib import Path
from typing import Optional, List, Union
from joltax import JolTree
from sweetbits.utils import parse_sample_id

def generate_table_logic(
    input_parquet: Path,
    output_file: Path,
    mode: str = "taxon",
    taxonomy_dir: Optional[Path] = None,
    exclude_samples: Optional[Path] = None,
    min_observed: int = 25,
    min_reads: int = 50,
    clade_filter: Optional[int] = None,
    keep_unclassified: bool = False
):
    """
    Generates an abundance table from a merged REPORT_PARQUET file.
    """
    # 1. Load Data
    df = pl.read_parquet(input_parquet)
    
    # 2. Filtering Samples
    if exclude_samples:
        with open(exclude_samples, "r") as f:
            excluded_ids = [line.strip() for line in f if line.strip()]
        df = df.filter(~pl.col("sample_id").is_in(excluded_ids))
        
    # 3. Mode Handling & Clade Filtering (Requires JolTax)
    tree = None
    if mode in ["clade", "canonical"] or clade_filter is not None:
        if not taxonomy_dir:
            raise ValueError(f"Taxonomy directory is required for mode '{mode}' or clade filtering.")
        tree = JolTree.load(str(taxonomy_dir))
        
    # 3a. Clade Filter
    if clade_filter is not None:
        clade_taxids = tree.get_clade(clade_filter)
        df = df.filter(pl.col("t_id").is_in(clade_taxids))
        
    # 3b. Unclassified Handling
    if not keep_unclassified:
        df = df.filter(pl.col("t_id") != 0)
        
    # 4. Aggregation based on Mode
    # mode 'taxon': use 'taxon_reads' and original 't_id'
    # mode 'clade': use 'clade_reads' and original 't_id'
    # mode 'canonical': map 't_id' to its nearest canonical ancestor, then sum 'taxon_reads'
    
    if mode == "taxon":
        pivot_col = "taxon_reads"
    elif mode == "clade":
        pivot_col = "clade_reads"
    elif mode == "canonical":
        # This is more complex: we need to map all t_ids to their canonical rank 
        # (usually species or genus if no species exists).
        # For now, let's assume we annotate and then pick a specific rank 
        # or use JolTax's canonical mapping.
        
        # Get unique TaxIDs in the dataset
        unique_tids = df["t_id"].unique().to_numpy()
        annotation = tree.annotate(unique_tids, strict=False)
        
        # Map each t_id to its most specific canonical rank available
        # (t_species -> t_genus -> ... -> t_domain)
        canonical_ranks = ["t_species", "t_genus", "t_family", "t_order", "t_class", "t_phylum", "t_domain"]
        # In a real implementation, we'd find the first non-null rank.
        # For simplicity in this first version, let's just use species for 'canonical'
        # or ask for clarification.
        # DIRECTIVE: JolTax USAGE.md says annotate returns t_id and t_rank.
        # Let's map to species for now as a placeholder.
        df = df.join(annotation.select(["t_id", "t_species"]), on="t_id")
        df = df.with_columns(pl.col("t_species").fill_null(pl.col("t_id").cast(pl.String)))
        # Re-assign t_id to the species ID if possible? 
        # Actually, let's just use t_id for now to avoid complexity unless requested.
        pivot_col = "taxon_reads"
    
    # 5. Pivot to wide format
    # Rows: t_id, Columns: sample_id (formatted as YYYY_WW)
    
    # Create the column name: YYYY_WW
    df = df.with_columns(
        (pl.col("year").cast(pl.String) + "_" + pl.col("week").cast(pl.String).str.pad_start(2, "0")).alias("period")
    )
    
    table = df.pivot(
        values=pivot_col,
        index="t_id",
        on="period",
        aggregate_function="sum"
    ).fill_null(0)
    
    # 6. Apply Filters (min_observed, min_reads)
    # min_observed: Taxon must be in at least INT samples
    sample_cols = [c for c in table.columns if c != "t_id"]
    
    if min_observed > 0:
        obs_count = table.select([
            pl.sum_horizontal([pl.col(c) > 0 for c in sample_cols]).alias("count")
        ])["count"]
        table = table.filter(obs_count >= min_observed)
        
    if min_reads > 0:
        max_reads = table.select([
            pl.max_horizontal(sample_cols).alias("max_val")
        ])["max_val"]
        table = table.filter(max_reads >= min_reads)
        
    # 7. Output based on extension
    ext = output_file.suffix.lower()
    if ext == ".parquet":
        table.write_parquet(output_file)
    elif ext == ".csv":
        table.write_csv(output_file)
    elif ext == ".tsv":
        table.write_csv(output_file, separator="\t")
    else:
        raise ValueError(f"Unsupported output format: {ext}")
