from pathlib import Path
from sweetbits.testing import generate_mock_kraken_parquet, generate_mock_report_parquet, generate_mock_taxonomy
from joltax import JolTree

def main():
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Generate mock taxonomy
    taxonomy_dir = test_data_dir / "taxonomy"
    print(f"Generating mock taxonomy in {taxonomy_dir}...")
    generate_mock_taxonomy(taxonomy_dir)
    
    # Build JolTax cache
    joltax_cache = test_data_dir / "joltax_cache"
    print(f"Building JolTax cache in {joltax_cache}...")
    tree = JolTree(tax_dir=str(taxonomy_dir))
    tree.save(str(joltax_cache))
    
    sample_id = "Lj-2022_20_001"
    
    print(f"Generating mock Kraken Parquet for {sample_id}...")
    generate_mock_kraken_parquet(
        sample_id=sample_id,
        num_reads=100,
        output_path=test_data_dir / f"{sample_id}.kraken.parquet"
    )
    
    print("Generating mock Report Parquet...")
    generate_mock_report_parquet(
        sample_ids=[sample_id, "Lj-2022_20_002"],
        output_path=test_data_dir / "merged_reports.parquet"
    )
    
    print(f"Done. Test data generated in {test_data_dir}/")

if __name__ == "__main__":
    main()
