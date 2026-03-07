"""
tests.test_canonical_complex
Stress tests for canonical remainder logic with complex taxonomic structures.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.tables import generate_table_logic
from sweetbits.metadata import write_parquet_with_metadata, get_standard_metadata
from joltax import JolTree

@pytest.fixture
def complex_taxonomy(tmp_path):
    """Creates a deep and broad taxonomy with nested non-canonical ranks."""
    tax_dir = tmp_path / "complex_tax"
    tax_dir.mkdir()
    
    # Format: TID | ParentTID | Rank
    nodes = [
        "1\t|\t1\t|\tno rank\t|",
        "2\t|\t1\t|\tsuperkingdom\t|",
        "10\t|\t2\t|\tphylum\t|",
        "100\t|\t10\t|\tclass\t|",
        "1000\t|\t100\t|\tsubclass\t|",      # Non-canonical
        "10000\t|\t1000\t|\torder\t|",
        "100000\t|\t10000\t|\tfamily\t|",
        "1000000\t|\t100000\t|\tgenus\t|",    # Direct child of Family
        "10000000\t|\t1000000\t|\tspecies\t|",
        "10000001\t|\t10000000\t|\tsubspecies\t|", # Non-canonical child of Species
        
        "20\t|\t2\t|\tphylum\t|",
        "200\t|\t20\t|\tno rank\t|",           # Non-canonical parent
        "2000\t|\t200\t|\tgenus\t|",           # Child of non-canonical
        "3000\t|\t200\t|\tgenus\t|",           # Sibling Genus
    ]
    
    names = [f"{n.split()[0]}\t|\tName_{n.split()[0]}\t|\t\t|\tscientific name\t|" for n in nodes]
    
    with open(tax_dir / "nodes.dmp", "w") as f:
        for l in nodes: f.write(l + "\n")
    with open(tax_dir / "names.dmp", "w") as f:
        for l in names: f.write(l + "\n")
        
    tree = JolTree(tax_dir=str(tax_dir))
    cache_dir = tmp_path / "complex_cache"
    tree.save(str(cache_dir))
    return cache_dir

def test_canonical_deep_nesting(complex_taxonomy, tmp_path):
    """Tests remainder calculation through deep non-canonical gaps."""
    # We put reads in a subspecies (10000001). 
    # They should stay in the parent species (10000000) but be subtracted from Genus (1000000).
    
    # Subspecies reads: 50
    # Species reads: 100 (Total clade) -> Species Remainder: 50
    # Genus reads: 150 (Total clade) -> Genus Remainder: 50
    # Family reads: 200 (Total clade) -> Family Remainder: 50
    
    data = pl.DataFrame({
        "sample_id": ["S1"] * 4,
        "year": [2022] * 4, "week": [1] * 4,
        "t_id": [100000, 1000000, 10000000, 10000001],
        "clade_reads": [200, 150, 100, 50], # Strictly consistent nesting
        "taxon_reads": [50, 50, 50, 50],
        "mm_tot": [0]*4, "mm_uniq": [0]*4, "source_file": ["f"]*4
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "deep.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    write_parquet_with_metadata(data, report_parquet, meta)
    
    out = tmp_path / "out.tsv"
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=complex_taxonomy, min_observed=0, min_reads=0)
    
    res = {row["t_id"]: row["2022_01"] for row in pl.read_csv(out, separator="\t").to_dicts()}
    
    # Subspecies 10000001 should be pushed up to Species 10000000
    # Species Remainder = Clade(Species) - Sum(Canonical Children) = 100 - 0 = 100
    # NOTE: Our algorithm sees 10000001 as non-canonical, so it doesn't subtract it from Species.
    # Instead, 10000001 maps to Species. 
    # Wait, 10000001 is a CHILD of Species. 
    # If 10000001 is non-canonical, its reads are essentially part of the Species remainder.
    
    assert 10000001 not in res
    assert res[10000000] == 100 # Species (Includes its own 50 + subspecies 50)
    assert res[1000000] == 50   # Genus (150 - 100 from species)
    assert res[100000] == 50    # Family (200 - 150 from genus)

def test_canonical_broad_non_canonical_parent(complex_taxonomy, tmp_path):
    """Tests if multiple canonical children correctly subtract from a distant canonical ancestor."""
    # Phylum_B (20) -> NonCanonical (200) -> [Genus_B (2000), Genus_C (3000)]
    
    # Genus B: 100 reads
    # Genus C: 100 reads
    # Phylum B: 250 reads
    # Expected: Genus B=100, Genus C=100, Phylum B=50.
    
    data = pl.DataFrame({
        "sample_id": ["S1"] * 3,
        "year": [2022] * 3, "week": [1] * 3,
        "t_id": [20, 2000, 3000],
        "clade_reads": [250, 100, 100],
        "taxon_reads": [50, 100, 100],
        "mm_tot": [0]*3, "mm_uniq": [0]*3, "source_file": ["f"]*3
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "broad.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    write_parquet_with_metadata(data, report_parquet, meta)
    
    out = tmp_path / "out_broad.tsv"
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=complex_taxonomy, min_observed=0, min_reads=0)
    
    res = {row["t_id"]: row["2022_01"] for row in pl.read_csv(out, separator="\t").to_dicts()}
    
    assert res[2000] == 100
    assert res[3000] == 100
    assert res[20] == 50
    assert 200 not in res # Non-canonical node skipped

def test_canonical_adjacent_ranks(complex_taxonomy, tmp_path):
    """Tests behavior when canonical ranks are direct parents/children."""
    # Family_A (100000) -> Genus_A (1000000)
    # Family has 100 reads, Genus has 100 reads.
    # Family should have 0 remainder.
    
    data = pl.DataFrame({
        "sample_id": ["S1", "S1"],
        "year": [2022, 2022], "week": [1, 1],
        "t_id": [100000, 1000000],
        "clade_reads": [100, 100],
        "taxon_reads": [0, 100],
        "mm_tot": [0, 0], "mm_uniq": [0, 0], "source_file": ["f", "f"]
    }).with_columns([pl.col("year").cast(pl.UInt16), pl.col("week").cast(pl.UInt8), pl.col("t_id").cast(pl.UInt32)])
    
    report_parquet = tmp_path / "adjacent.parquet"
    meta = get_standard_metadata("REPORT_PARQUET", source_path=tmp_path, data_standard="SWEBITS")
    write_parquet_with_metadata(data, report_parquet, meta)
    
    out = tmp_path / "out_adj.tsv"
    generate_table_logic(report_parquet, out, mode="canonical", taxonomy_dir=complex_taxonomy, min_observed=0, min_reads=0)
    
    res = {row["t_id"]: row["2022_01"] for row in pl.read_csv(out, separator="\t").to_dicts()}
    
    assert res[1000000] == 100
    assert res[100000] == 0 # All reads accounted for by genus
