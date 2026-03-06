import click
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from sweetbits import __version__
from sweetbits.reports import gather_reports_logic

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

@click.group()
def main():
    """SweetBITS CLI toolkit."""
    pass

@main.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Path to output Parquet file.")
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories (Default: True).")
@click.option("--include", "-i", default="*.report", help="Pattern to match report files (Default: *.report).")
def gather_reports(directory, output, recursive, include):
    """Merges multiple 8-column Kraken reports into a single Parquet file."""
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

if __name__ == "__main__":
    main()
