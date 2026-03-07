"""
sweetbits.utils
Utility functions for parsing and validating SweBITS data structures.
"""

import re
from typing import Dict, Any

def parse_sample_id(sample_id: str) -> Dict[str, Any]:
    """
    Parses a SweBITS sample ID into its components and validates the format.
    
    Supported formats:
    - Ki-YYYY_WW_ZZZ (Kiruna)
    - Lj-YYYY_WW_ZZZ (Ljungbyhed)
    
    The format strictly requires:
    - Site: 'Ki' or 'Lj'
    - Year: 4 digits
    - Week: 1 or 2 digits (validated to range 1-53)
    - Suffix: Exactly 3 digits
    
    Args:
        sample_id: The string ID to parse (e.g., 'Ki-2022_20_001').

    Returns:
        A dictionary containing:
        - site: 'Kiruna' or 'Ljungbyhed'
        - year: int
        - week: int
        - suffix: str (the 3-digit ZZZ part)
        - sample_id: original sample ID
    
    Raises:
        ValueError: If the sample_id format is invalid or week is out of range.
    """
    # Regex breakdown:
    # ^(Ki|Lj)      : Starts with Ki or Lj
    # [-_]          : Followed by a hyphen or underscore
    # (\d{4})       : Exactly 4 digits for the year
    # _             : Underscore separator
    # (\d{1,2})     : 1 or 2 digits for the week
    # _             : Underscore separator
    # (\d{3})       : Exactly 3 digits for the filter/replicate suffix
    # $             : End of string
    pattern = r"^(Ki|Lj)[-_](\d{4})_(\d{1,2})_(\d{3})$"
    match = re.match(pattern, sample_id)
    
    if not match:
        raise ValueError(
            f"Invalid sample ID format: '{sample_id}'. "
            "Expected 'Ki-YYYY_WW_ZZZ' or 'Lj-YYYY_WW_ZZZ' with numeric components."
        )
    
    site_code, year_str, week_str, suffix = match.groups()
    
    year = int(year_str)
    week = int(week_str)
    
    # Basic ISO week validation
    if not (1 <= week <= 53):
        raise ValueError(f"Invalid ISO week in sample ID: {week}. Must be between 1 and 53.")
    
    site_map = {
        "Ki": "Kiruna",
        "Lj": "Ljungbyhed"
    }
    
    return {
        "site": site_map[site_code],
        "year": year,
        "week": week,
        "suffix": suffix,
        "sample_id": sample_id
    }
