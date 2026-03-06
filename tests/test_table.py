import pytest
import polars as pl
from pathlib import Path
from sweetbits.testing import generate_mock_report_parquet, generate_mock_taxonomy
from sweetbits.tables import generate_table_logic
from joltax import JolTree

@pytest.fixture
def mock_data(tmp_path):
    report_parquet = tmp_path / "reports.parquet"
    sample_ids = ["Lj-2022_20_001", "Lj-2022_20_002"]
    generate_mock_report_parquet(sample_ids, report_parquet)
    
    taxonomy_dir = tmp_path / "taxonomy"
    generate_mock_taxonomy(taxonomy_dir)
    
    # JolTax needs to build the cache first
    tree = JolTree(tax_dir=str(taxonomy_dir))
    cache_dir = tmp_path / "joltax_cache"
    tree.save(str(cache_dir))
    
    return {
        "parquet": report_parquet,
        "taxonomy": cache_dir,
        "samples": sample_ids
    }

def test_table_taxon_mode(mock_data, tmp_path):
    output_csv = tmp_path / "table.csv"
    generate_table_logic(
        input_parquet=mock_data["parquet"],
        output_file=output_csv,
        mode="taxon",
        min_observed=1,
        min_reads=1
    )
    
    assert output_csv.exists()
    df = pl.read_csv(output_csv)
    
    # Columns should be t_id and the periods (2022_20)
    assert "t_id" in df.columns
    assert "2022_20" in df.columns
    # Since both samples are 2022_20, they should be summed into one column
    assert df.width == 2 

def test_table_clade_filter(mock_data, tmp_path):
    output_csv = tmp_path / "clade_table.csv"
    # Filter for Bacteria (TaxID 2)
    # Our mock taxonomy has 5000001 and 5000002 under Bacteria (via 5000000)
    generate_table_logic(
        input_parquet=mock_data["parquet"],
        output_file=output_csv,
        mode="taxon",
        taxonomy_dir=mock_data["taxonomy"],
        clade_filter=2,
        min_observed=1,
        min_reads=1
    )
    
    df = pl.read_csv(output_csv)
    # Should only have Bacteria descendants
    # Mock ids: 5000000, 5000001, 5000002
    assert all(tid in [5000000, 5000001, 5000002] for tid in df["t_id"])
    assert 9606 not in df["t_id"].to_list()

def test_table_exclude_samples(mock_data, tmp_path):
    exclude_file = tmp_path / "exclude.txt"
    with open(exclude_file, "w") as f:
        f.write("Lj-2022_20_001\n")
        
    output_csv = tmp_path / "filtered_table.csv"
    # This should leave only Lj-2022_20_002
    # In our mock, both are week 20, so the 'period' column name remains the same
    # but the values should change.
    generate_table_logic(
        input_parquet=mock_data["parquet"],
        output_file=output_csv,
        exclude_samples=exclude_file,
        min_observed=1,
        min_reads=1
    )
    
    assert output_csv.exists()

def test_table_min_observed(mock_data, tmp_path):
    # If we have two samples in the same period, they sum up.
    # To test min_observed, we need samples in different periods.
    report_parquet = tmp_path / "multi_period.parquet"
    sample_ids = ["Lj-2022_20_001", "Lj-2022_21_001"]
    generate_mock_report_parquet(sample_ids, report_parquet)
    
    output_csv = tmp_path / "obs_table.csv"
    generate_table_logic(
        input_parquet=report_parquet,
        output_file=output_csv,
        min_observed=2, # Must be in both 2022_20 and 2022_21
        min_reads=1
    )
    
    df = pl.read_csv(output_csv)
    # Our mock generator puts all taxids in all samples, so they should all remain
    assert df.height > 0
    assert "2022_20" in df.columns
    assert "2022_21" in df.columns
