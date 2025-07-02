"""
Data loader utility for the solar prediction project.
Handles loading the full dataset or fallback to sample data.
"""

import pandas as pd
import os
import logging
from pathlib import Path

def load_solar_dataset(data_dir: str = None, prefer_sample: bool = False) -> pd.DataFrame:
    """
    Load the solar dataset with fallback logic.
    
    Args:
        data_dir: Directory containing the data files. If None, uses project default.
        prefer_sample: If True, prefer sample data over full dataset.
        
    Returns:
        pandas.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If no data files are found
    """
    if data_dir is None:
        # Default to project data directory
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
    else:
        data_dir = Path(data_dir)
    
    full_data_path = data_dir / "SolarPrediction.csv" 
    sample_data_path = data_dir / "sample" / "SolarPrediction_sample.csv"
    
    # Define the preferred order based on prefer_sample flag
    if prefer_sample:
        paths_to_try = [sample_data_path, full_data_path]
        data_types = ["sample", "full"]
    else:
        paths_to_try = [full_data_path, sample_data_path]
        data_types = ["full", "sample"]
    
    for path, data_type in zip(paths_to_try, data_types):
        if path.exists():
            logging.info(f"Loading {data_type} dataset from: {path}")
            try:
                df = pd.read_csv(path)
                logging.info(f"Successfully loaded {data_type} dataset with {len(df)} records")
                return df
            except Exception as e:
                logging.warning(f"Failed to load {path}: {e}")
                continue
    
    # If we get here, no files were found or loaded successfully
    raise FileNotFoundError(f"No solar dataset found. Tried: {[str(p) for p in paths_to_try]}")


def get_dataset_info(data_dir: str = None) -> dict:
    """
    Get information about available datasets.
    
    Args:
        data_dir: Directory containing the data files. If None, uses project default.
        
    Returns:
        dict: Information about available datasets
    """
    if data_dir is None:
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
    else:
        data_dir = Path(data_dir)
    
    full_data_path = data_dir / "SolarPrediction.csv"
    sample_data_path = data_dir / "sample" / "SolarPrediction_sample.csv"
    
    info = {
        "full_dataset": {
            "path": str(full_data_path),
            "exists": full_data_path.exists(),
            "size_mb": None,
            "records": None
        },
        "sample_dataset": {
            "path": str(sample_data_path),
            "exists": sample_data_path.exists(),
            "size_mb": None,
            "records": None
        }
    }
    
    # Get file sizes and record counts
    for dataset_type in ["full_dataset", "sample_dataset"]:
        path = Path(info[dataset_type]["path"])
        if path.exists():
            try:
                # Get file size
                size_bytes = path.stat().st_size
                info[dataset_type]["size_mb"] = round(size_bytes / (1024 * 1024), 2)
                
                # Get record count (quick method - count lines minus header)
                with open(path, 'r') as f:
                    line_count = sum(1 for _ in f) - 1  # Subtract header
                info[dataset_type]["records"] = line_count
                
            except Exception as e:
                logging.warning(f"Could not get info for {path}: {e}")
    
    return info
