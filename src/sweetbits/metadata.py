import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from datetime import datetime
import sys
import os
from typing import Dict, Any, Optional
from sweetbits import __version__

def get_standard_metadata(file_type: str, source_path: Optional[Path] = None) -> Dict[str, str]:
    """Generates the standard metadata dictionary for SweetBITS parquet files."""
    # Try to detect if we're running via the 'sweetbits' entry point
    args = sys.argv[1:]
    command_str = " ".join(args)
    
    # If running via python script, include the script name
    if not sys.argv[0].endswith("sweetbits"):
        invocation = f"python {sys.argv[0]} {command_str}".strip()
    else:
        invocation = f"sweetbits {command_str}".strip()

    metadata = {
        "sweetbits_version": __version__,
        "file_type": file_type,
        "execution_command": invocation,
        "creation_time": datetime.now().isoformat(),
        "source_path_abs": str(source_path.resolve()) if source_path else "Unknown"
    }
    return metadata

def write_parquet_with_metadata(df: 'pl.DataFrame', output_path: Path, metadata: Dict[str, str], **kwargs):
    """Writes a Polars DataFrame to Parquet with custom file-level metadata."""
    # Convert Polars to Arrow
    table = df.to_arrow()
    
    # Existing metadata
    existing_meta = table.schema.metadata or {}
    
    # Merge with our metadata (must be bytes)
    merged_meta = {**existing_meta}
    for k, v in metadata.items():
        merged_meta[k.encode()] = str(v).encode()
        
    # Create new schema with metadata
    new_schema = table.schema.with_metadata(merged_meta)
    table = table.cast(new_schema)
    
    # Write using pyarrow to ensure metadata is preserved
    pq.write_table(table, output_path, **kwargs)

def read_parquet_metadata(file_path: Path) -> Dict[str, str]:
    """Reads the custom metadata from a Parquet file header."""
    schema = pq.read_schema(file_path)
    if not schema.metadata:
        return {}
    
    # Convert bytes back to strings
    return {k.decode(): v.decode() for k, v in schema.metadata.items()}
