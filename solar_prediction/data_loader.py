"""
Data loader utility for the solar prediction project.
Handles loading the full dataset or fallback to sample data.
Now includes chunked reading for large CSV files and on-the-fly feature engineering.
"""

import pandas as pd
import os
import logging
from pathlib import Path
from typing import Iterator, Optional, Callable, Dict, Any, Union
import numpy as np
from dataclasses import dataclass

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
    
    full_data_path = data_dir / "solar_weather.csv" 
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
    
    full_data_path = data_dir / "solar_weather.csv"
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


@dataclass
class ChunkedDataLoaderConfig:
    """Configuration for chunked data loading."""
    chunksize: int = 1_000_000  # 1M rows per chunk
    apply_feature_engineering: bool = True
    memory_efficient: bool = True
    skip_empty_chunks: bool = True
    
    
class ChunkedDataLoader:
    """Lightweight data loader for large CSVs with chunked reading and on-the-fly feature engineering."""
    
    def __init__(self, config: ChunkedDataLoaderConfig = None):
        self.config = config or ChunkedDataLoaderConfig()
        self.logger = logging.getLogger(__name__)
        
    def load_chunks(self, file_path: str, **read_csv_kwargs) -> Iterator[pd.DataFrame]:
        """Load CSV file in chunks with optional feature engineering.
        
        Args:
            file_path: Path to the CSV file
            **read_csv_kwargs: Additional arguments for pd.read_csv
            
        Yields:
            pd.DataFrame: Processed chunks
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        self.logger.info(f"Starting chunked reading of {file_path} with chunksize={self.config.chunksize}")
        
        # Set up read_csv parameters
        read_params = {
            'chunksize': self.config.chunksize,
            'low_memory': self.config.memory_efficient,
            **read_csv_kwargs
        }
        
        chunk_count = 0
        total_rows = 0
        
        try:
            for chunk in pd.read_csv(file_path, **read_params):
                chunk_count += 1
                
                # Skip empty chunks if configured
                if self.config.skip_empty_chunks and chunk.empty:
                    self.logger.debug(f"Skipping empty chunk {chunk_count}")
                    continue
                    
                # Apply feature engineering if enabled
                if self.config.apply_feature_engineering:
                    chunk = self._apply_basic_feature_engineering(chunk)
                    
                total_rows += len(chunk)
                self.logger.debug(f"Processed chunk {chunk_count} with {len(chunk)} rows")
                
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Error processing chunks: {e}")
            raise
            
        self.logger.info(f"Completed chunked reading: {chunk_count} chunks, {total_rows} total rows")
    
    def _apply_basic_feature_engineering(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply basic feature engineering to a chunk.
        
        Args:
            chunk: Input chunk DataFrame
            
        Returns:
            pd.DataFrame: Processed chunk with engineered features
        """
        # Work with a copy to avoid modifying the original
        processed_chunk = chunk.copy()
        
        # Basic time-based feature engineering
        if 'Timestamp' in processed_chunk.columns:
            try:
                processed_chunk['Timestamp'] = pd.to_datetime(processed_chunk['Timestamp'], cache=True)
                processed_chunk['HourOfDay'] = processed_chunk['Timestamp'].dt.hour
                processed_chunk['DayOfYear'] = processed_chunk['Timestamp'].dt.dayofyear
                processed_chunk['Month'] = processed_chunk['Timestamp'].dt.month
                
                # Cyclical features
                processed_chunk['HourSin'] = np.sin(2 * np.pi * processed_chunk['HourOfDay'] / 24)
                processed_chunk['HourCos'] = np.cos(2 * np.pi * processed_chunk['HourOfDay'] / 24)
                processed_chunk['DayOfYearSin'] = np.sin(2 * np.pi * processed_chunk['DayOfYear'] / 365)
                processed_chunk['DayOfYearCos'] = np.cos(2 * np.pi * processed_chunk['DayOfYear'] / 365)
                
            except Exception as e:
                self.logger.warning(f"Failed to process timestamp features: {e}")
                
        # Basic weather feature engineering
        if 'Temperature' in processed_chunk.columns:
            # Temperature-based features
            processed_chunk['TempSquared'] = processed_chunk['Temperature'] ** 2
            
        if 'Humidity' in processed_chunk.columns:
            # Humidity-based features
            processed_chunk['HumiditySquared'] = processed_chunk['Humidity'] ** 2
            
        if 'WindSpeed' in processed_chunk.columns:
            # Wind speed features
            processed_chunk['WindSpeedSquared'] = processed_chunk['WindSpeed'] ** 2
            
        # Interaction features
        if 'Temperature' in processed_chunk.columns and 'Humidity' in processed_chunk.columns:
            processed_chunk['TempHumidityInteraction'] = processed_chunk['Temperature'] * processed_chunk['Humidity']
            
        return processed_chunk
    
    def load_and_combine_chunks(self, file_path: str, max_chunks: Optional[int] = None, **read_csv_kwargs) -> pd.DataFrame:
        """Load chunks and combine them into a single DataFrame.
        
        Args:
            file_path: Path to the CSV file
            max_chunks: Maximum number of chunks to load (None for all)
            **read_csv_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Combined DataFrame
        """
        chunks = []
        
        for i, chunk in enumerate(self.load_chunks(file_path, **read_csv_kwargs)):
            chunks.append(chunk)
            
            if max_chunks and i + 1 >= max_chunks:
                self.logger.info(f"Reached maximum chunk limit: {max_chunks}")
                break
                
        if not chunks:
            raise ValueError("No chunks were loaded")
            
        combined_df = pd.concat(chunks, ignore_index=True)
        self.logger.info(f"Combined {len(chunks)} chunks into DataFrame with {len(combined_df)} rows")
        
        return combined_df
    
    def get_chunk_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about the chunks without loading the full data.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            dict: Information about chunks
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Estimate number of chunks by reading a small sample
        try:
            sample_chunk = next(pd.read_csv(file_path, chunksize=1000, nrows=1000))
            estimated_total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
            estimated_chunks = max(1, estimated_total_rows // self.config.chunksize)
            
            info = {
                'file_path': str(file_path),
                'file_size_mb': round(file_size_mb, 2),
                'chunksize': self.config.chunksize,
                'estimated_total_rows': estimated_total_rows,
                'estimated_chunks': estimated_chunks,
                'sample_columns': list(sample_chunk.columns),
                'sample_dtypes': sample_chunk.dtypes.to_dict()
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting chunk info: {e}")
            return {'error': str(e)}


class DataIteratorAdapter:
    """Adapter to make chunked data compatible with existing prepare_weather_data function."""
    
    def __init__(self, chunked_loader: ChunkedDataLoader, file_path: str, **read_csv_kwargs):
        self.chunked_loader = chunked_loader
        self.file_path = file_path
        self.read_csv_kwargs = read_csv_kwargs
        self.logger = logging.getLogger(__name__)
        
    def __iter__(self):
        """Make this object iterable."""
        return self.chunked_loader.load_chunks(self.file_path, **self.read_csv_kwargs)
    
    def to_dataframe(self, max_chunks: Optional[int] = None) -> pd.DataFrame:
        """Convert iterator to DataFrame for compatibility with existing code.
        
        Args:
            max_chunks: Maximum number of chunks to load
            
        Returns:
            pd.DataFrame: Combined DataFrame
        """
        return self.chunked_loader.load_and_combine_chunks(
            self.file_path, max_chunks=max_chunks, **self.read_csv_kwargs
        )


def create_chunked_loader(config: ChunkedDataLoaderConfig = None) -> ChunkedDataLoader:
    """Factory function to create a chunked data loader.
    
    Args:
        config: Configuration for the chunked loader
        
    Returns:
        ChunkedDataLoader: Configured chunked data loader
    """
    return ChunkedDataLoader(config)


def prepare_weather_data_chunked(
    data_source: Union[str, Path, Iterator[pd.DataFrame], pd.DataFrame],
    input_cfg,
    transform_cfg,
    feature_cfg,
    scaling_cfg,
    sequence_cfg,
    chunked_config: ChunkedDataLoaderConfig = None,
    max_chunks: Optional[int] = None
):
    """Unified API for prepare_weather_data that accepts various data sources.
    
    Args:
        data_source: Can be:
            - str/Path: Path to CSV file (will be loaded with chunking)
            - Iterator[pd.DataFrame]: Iterator of DataFrame chunks
            - pd.DataFrame: Regular DataFrame
        input_cfg: Data input configuration
        transform_cfg: Data transformation configuration
        feature_cfg: Feature engineering configuration
        scaling_cfg: Scaling configuration
        sequence_cfg: Sequence configuration
        chunked_config: Configuration for chunked loading
        max_chunks: Maximum number of chunks to process
        
    Returns:
        Same as prepare_weather_data function
    """
    # Import here to avoid circular imports
    from .data_prep import prepare_weather_data
    
    logger = logging.getLogger(__name__)
    
    # Handle different data source types
    if isinstance(data_source, (str, Path)):
        # File path - use chunked loading
        logger.info(f"Loading data from file with chunked reading: {data_source}")
        chunked_loader = create_chunked_loader(chunked_config)
        adapter = DataIteratorAdapter(chunked_loader, str(data_source))
        df = adapter.to_dataframe(max_chunks=max_chunks)
        
    elif hasattr(data_source, '__iter__') and not isinstance(data_source, pd.DataFrame):
        # Iterator of chunks
        logger.info("Processing data from iterator")
        chunks = []
        for i, chunk in enumerate(data_source):
            chunks.append(chunk)
            if max_chunks and i + 1 >= max_chunks:
                logger.info(f"Reached maximum chunk limit: {max_chunks}")
                break
                
        if not chunks:
            raise ValueError("No data chunks provided")
            
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Combined {len(chunks)} chunks into DataFrame with {len(df)} rows")
        
    elif isinstance(data_source, pd.DataFrame):
        # Regular DataFrame
        logger.info("Using provided DataFrame directly")
        df = data_source
        
    else:
        raise ValueError(f"Unsupported data source type: {type(data_source)}")
    
    # Call the original prepare_weather_data function
    return prepare_weather_data(df, input_cfg, transform_cfg, feature_cfg, scaling_cfg, sequence_cfg)
