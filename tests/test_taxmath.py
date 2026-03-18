import pytest
import polars as pl
import numpy as np
from pathlib import Path
from sweetbits.taxmath import calc_clade_sum
from joltax import JolTree

def test_calc_clade_sum_basic(tmp_path):
    tax_dir = Path("test_data/joltax_cache")
    if not tax_dir.exists():
        pytest.skip("JolTax cache not found")
        
    tree = JolTree.load(str(tax_dir))
    
    # Eukaryota (9606) has depth ~8, Bacteria (2) has depth 1. 
    # Root (1).
    # Let's use some IDs we know: 
    # 2 (Bacteria), 2759 (Eukaryota), 9606 (Homo sapiens), 1 (root)
    
    df = pl.DataFrame({
        "sample_id": ["S1", "S1", "S1"],
        "t_id": [9606, 2, 1],
        "taxon_reads": [10, 20, 0]
    })
    
    result = calc_clade_sum(df, tree, min_reads=0, min_observed=0)
    
    res_dict = {row["t_id"]: row["clade_reads"] for row in result.to_dicts()}
    
    # 9606 clade should be 10
    assert res_dict.get(9606, 0) == 10
    # 2 clade should be 20
    assert res_dict.get(2, 0) == 20
    # 1 (root) should get votes from 9606 and 2 (so 10 + 20 = 30)
    assert res_dict.get(1, 0) == 30
    
def test_calc_clade_sum_filtering(tmp_path):
    tax_dir = Path("test_data/joltax_cache")
    if not tax_dir.exists():
        pytest.skip("JolTax cache not found")
        
    tree = JolTree.load(str(tax_dir))
    
    df = pl.DataFrame({
        "sample_id": ["S1", "S1", "S1"],
        "t_id": [9606, 2, 1],
        "taxon_reads": [10, 20, 0]
    })
    
    # Set min_reads = 15. 9606 (10) should fail. 2 (20) should pass.
    # 9606 (Homo sapiens) will fail and its 10 reads should be pushed to its parent.
    # The parent of 9606 is 9605 (Homo). 
    # Let's just check that total taxon reads remain 30!
    
    result = calc_clade_sum(df, tree, min_reads=15, min_observed=1)
    
    res_clade = {row["t_id"]: row["clade_reads"] for row in result.to_dicts()}
    res_taxon = {row["t_id"]: row["taxon_reads"] for row in result.to_dicts()}
    
    assert 9606 not in res_clade  # Failed, should be pruned
    assert res_clade.get(2, 0) == 20    # Passed
    
    # Root clade reads should still be exactly 30 because the 10 reads from 9606
    # were pushed up to 9605 (and then potentially up to Root if 9605 fails)
    assert res_clade.get(1, 0) == 30    
    
    # Total taxon reads must perfectly match mass balance
    total_taxon_in = df["taxon_reads"].sum()
    total_taxon_out = result["taxon_reads"].sum()
    assert total_taxon_in == total_taxon_out
