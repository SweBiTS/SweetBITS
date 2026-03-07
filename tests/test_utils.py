import pytest
from sweetbits.utils import parse_sample_id

def test_parse_sample_id_kiruna():
    sample_id = "Ki-1974_02_001"
    result = parse_sample_id(sample_id)
    assert result["site"] == "Kiruna"
    assert result["year"] == 1974
    assert result["week"] == 2
    assert result["suffix"] == "001"
    assert result["sample_id"] == sample_id

def test_parse_sample_id_ljungbyhed():
    sample_id = "Lj-2022_52_999"
    result = parse_sample_id(sample_id)
    assert result["site"] == "Ljungbyhed"
    assert result["year"] == 2022
    assert result["week"] == 52
    assert result["suffix"] == "999"
    assert result["sample_id"] == sample_id

def test_parse_sample_id_underscore():
    sample_id = "Lj_2013_1_142"
    result = parse_sample_id(sample_id)
    assert result["site"] == "Ljungbyhed"
    assert result["year"] == 2013
    assert result["week"] == 1
    assert result["suffix"] == "142"

def test_parse_sample_id_invalid():
    with pytest.raises(ValueError, match="Invalid sample ID format"):
        parse_sample_id("Ki-1974_02_ABC") # Non-numeric suffix
    
    with pytest.raises(ValueError, match="Invalid sample ID format"):
        parse_sample_id("XX-2022_01_001") # Invalid site
        
    with pytest.raises(ValueError, match="Invalid sample ID format"):
        parse_sample_id("Ki-22_01_001") # Invalid year

def test_parse_sample_id_invalid_week():
    with pytest.raises(ValueError, match="Invalid ISO week"):
        parse_sample_id("Ki-2022_60_001") # Week > 53
    
    with pytest.raises(ValueError, match="Invalid ISO week"):
        parse_sample_id("Ki-2022_0_001") # Week < 1
