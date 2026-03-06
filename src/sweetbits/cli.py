import click
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from sweetbits import __version__
from sweetbits.reports import gather_reports_logic
from sweetbits.tables import generate_table_logic

def print_header(ctx):
    click.echo(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"Toolkit Version: {__version__}")
    click.echo(f"CWD: {os.getcwd()}")
    click.echo(f"Command: sweetbits {' '.join(sys.argv[1:])}")
    click.echo("-" * 40)

def print_footer(start_time, summary=""):
    elapsed = time.time() - start_time
    click.echo("-" * 40)
    if summary:
        click.echo(summary)
    click.echo(f"Time elapsed: {elapsed:.2f}s")
    click.echo(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """
    SweetBITS: Bioinformatics command-line tools for the Swedish Biodiversity 
    in Time and Space (SweBITS) project.
    """
    pass

@main.command(short_help="Merge Kraken reports into a single Parquet file.")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output Parquet file.")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories (Default: True).")
@click.option("--include", "-i", default="*.report", help="Pattern to match report files (Default: *.report).")
def gather_reports(directory, output, recursive, include):
    """
    Finds and merges multiple 8-column Kraken reports into a single Polars-optimized 
    Parquet file, including provenance and temporal metadata.
    """
    start_time = time.time()
    print_header(click.get_current_context())
    
    try:
        gather_reports_logic(
            input_dir=directory,
            output_file=output,
            recursive=recursive,
            include_pattern=include
        )
        print_footer(start_time, f"Successfully merged reports into {output}")
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

@main.command(short_help="Generate abundance tables from merged reports.")
@click.argument("input_parquet", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output file (.csv, .tsv, .parquet).")
@click.option("--mode", "-m", type=click.Choice(["taxon", "clade", "canonical"]), default="taxon", help="Abundance mode (Default: taxon).")
@click.option("--taxonomy", "-t", type=click.Path(path_type=Path), help="JolTax cache directory.")
@click.option("--exclude-samples", type=click.Path(exists=True, path_type=Path), help="File with sample IDs to exclude.")
@click.option("--min-observed", type=int, default=25, help="Minimum samples taxon must be in (Default: 25).")
@click.option("--min-reads", type=int, default=50, help="Minimum max reads across samples (Default: 50).")
@click.option("--clade", type=int, help="Filter for taxa rooted at this TaxID.")
@click.option("--keep-unclassified", is_flag=True, help="Keep TaxID 0 (unclassified).")
def table(input_parquet, output, mode, taxonomy, exclude_samples, min_observed, min_reads, clade, keep_unclassified):
    """
    Outputs abundance tables with TaxIDs as rows and samples (YYYY_WW) as columns.
    Supports filtering by clade, minimum occupancy, and read depth.
    """
    start_time = time.time()
    print_header(click.get_current_context())
    
    try:
        generate_table_logic(
            input_parquet=input_parquet,
            output_file=output,
            mode=mode,
            taxonomy_dir=taxonomy,
            exclude_samples=exclude_samples,
            min_observed=min_observed,
            min_reads=min_reads,
            clade_filter=clade,
            keep_unclassified=keep_unclassified
        )
        print_footer(start_time, f"Successfully generated table: {output}")
    except Exception as e:
        click.secho(f"Error: {str(e)}", fg="red", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
