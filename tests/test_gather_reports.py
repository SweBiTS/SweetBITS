import pytest
import polars as pl
from pathlib import Path
from sweetbits.testing import generate_mock_kraken_report_file
from sweetbits.reports import gather_reports_logic

def test_gather_reports_logic(tmp_path):
    # Setup: Create a directory with two mock reports
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    
    report1 = report_dir / "Lj-2022_20_001.report"
    report2 = report_dir / "Lj-2022_20_002.report"
    
    generate_mock_kraken_report_file(report1)
    generate_mock_kraken_report_file(report2)
    
    # Nested report
    nested_dir = report_dir / "nested"
    nested_dir.mkdir()
    report3 = nested_dir / "Ki-1974_02_001.report"
    generate_mock_kraken_report_file(report3)
    
    output_parquet = tmp_path / "merged.parquet"
    
    # Run logic
    gather_reports_logic(
        input_dir=report_dir,
        output_file=output_parquet,
        recursive=True,
        include_pattern="*.report"
    )
    
    assert output_parquet.exists()
    
    df = pl.read_parquet(output_parquet)
    
    # Check shape: 3 reports * 5 rows each = 15 rows
    assert df.height == 15
    
    # Check columns
    expected_cols = ["sample_id", "year", "week", "t_id", "clade_reads", "taxon_reads", "mm_tot", "mm_uniq", "source_file"]
    assert all(col in df.columns for col in expected_cols)
    
    # Check sample IDs
    sample_ids = df["sample_id"].unique().to_list()
    assert "Lj-2022_20_001" in sample_ids
    assert "Lj-2022_20_002" in sample_ids
    assert "Ki-1974_02_001" in sample_ids
    
    # Check sorting (year, week, sample_id, t_id)
    # 1974 should come before 2022
    assert df["year"].to_list() == sorted(df["year"].to_list())
    
    # Check provenance
    assert "Lj-2022_20_001.report" in df.filter(pl.col("sample_id") == "Lj-2022_20_001")["source_file"][0]

def test_gather_reports_invalid_sample_id(tmp_path):
    report_dir = tmp_path / "reports_invalid"
    report_dir.mkdir()
    invalid_report = report_dir / "InvalidName.report"
    generate_mock_kraken_report_file(invalid_report)
    
    output_parquet = tmp_path / "merged_fail.parquet"
    
    # Should raise ValueError due to invalid sample ID format in filename
    with pytest.raises(ValueError, match="Invalid sample ID format"):
        gather_reports_logic(
            input_dir=report_dir,
            output_file=output_parquet,
            recursive=False,
            include_pattern="*.report"
        )
