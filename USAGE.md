# SweetBITS Usage Guide

This guide provides practical examples for the SweetBITS command-line tools.

## 1. Merging Kraken Reports (`gather-reports`)

Merge multiple `.report` files from a directory into a single optimized Parquet file.

```bash
# Basic usage
sweetbits gather-reports /path/to/reports --output merged_reports.parquet

# Recursive search with specific pattern
sweetbits gather-reports /path/to/data --output results.parquet --recursive --include "*.kraken.report"
```

## 2. Generating Abundance Tables (`table`)

Create a wide-format matrix from merged reports.

```bash
# Default (Clade mode, SWEBITS period grouping)
sweetbits table merged_reports.parquet --output abundance_table.tsv

# Taxon mode with specific filtering
sweetbits table merged_reports.parquet \
    --output taxon_table.csv \
    --mode taxon \
    --min-observed 50 \
    --min-reads 100

# Canonical Remainders (Requires JolTax)
sweetbits table merged_reports.parquet \
    --output canonical_table.tsv \
    --mode canonical \
    --taxonomy /path/to/joltax_cache

# Filter for a specific clade (e.g., Bacteria = 2)
sweetbits table merged_reports.parquet \
    --output bacteria_only.tsv \
    --taxonomy /path/to/joltax_cache \
    --clade 2
```

## 3. Extracting Reads (`extract-reads`)

Stream reads from annotated Parquet files back to FASTQ.gz format.

```bash
# Extract specific TaxIDs
sweetbits extract-reads /path/to/parquet_dir \
    --taxonomy /path/to/joltax_cache \
    --tax-id "9606,10090" \
    --output-dir ./extracted_reads

# Combine all samples into one file per TaxID with temporal filtering
sweetbits extract-reads /path/to/data \
    --taxonomy /path/to/joltax_cache \
    --tax-id "2" \
    --combine-samples \
    --year-start 2020 --year-end 2022 \
    --output-dir ./bacteria_reads
```

## 4. Inspecting Metadata (`inspect`)

View the provenance and configuration of any SweetBITS Parquet file.

```bash
sweetbits inspect merged_reports.parquet
```
