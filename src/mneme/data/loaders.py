"""Data loading utilities for various data formats."""

import numpy as np
import h5py
from typing import Dict, Any, Optional, List, Iterator, Tuple, Union
from pathlib import Path
import warnings
from dataclasses import dataclass

from ..types import Field, BioelectricMeasurement, FieldData


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    n_samples: int
    shape: Tuple[int, ...]
    dtype: str
    metadata: Dict[str, Any]


class BaseDataLoader:
    """Base class for data loaders."""
    
    def __init__(self, data_dir: Union[str, Path], lazy_load: bool = True):
        """
        Initialize data loader.
        
        Parameters
        ----------
        data_dir : str or Path
            Directory containing data files
        lazy_load : bool
            Whether to load data on demand
        """
        self.data_dir = Path(data_dir)
        self.lazy_load = lazy_load
        self.file_list = []
        self.metadata = {}
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self._scan_directory()
    
    def _scan_directory(self):
        """Scan directory for data files."""
        raise NotImplementedError("Subclasses must implement _scan_directory")
    
    def __len__(self) -> int:
        """Return number of data files."""
        return len(self.file_list)
    
    def __iter__(self) -> Iterator:
        """Iterate over data files."""
        for file_path in self.file_list:
            yield self.load_file(file_path)
    
    def load_file(self, file_path: Union[str, Path]) -> Any:
        """Load a single file."""
        raise NotImplementedError("Subclasses must implement load_file")
    
    def get_info(self) -> DatasetInfo:
        """Get dataset information."""
        if not self.file_list:
            return DatasetInfo("empty", 0, (), "unknown", {})
        
        # Load first file to get shape and dtype info
        sample = self.load_file(self.file_list[0])
        
        if hasattr(sample, 'shape'):
            shape = sample.shape
            dtype = str(sample.dtype)
        elif hasattr(sample, 'data'):
            shape = sample.data.shape
            dtype = str(sample.data.dtype)
        else:
            shape = ()
            dtype = "unknown"
        
        return DatasetInfo(
            name=self.data_dir.name,
            n_samples=len(self.file_list),
            shape=shape,
            dtype=dtype,
            metadata=self.metadata
        )


class HDF5DataLoader(BaseDataLoader):
    """Load data from HDF5 files."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        file_pattern: str = "*.h5",
        data_key: str = "data",
        lazy_load: bool = True
    ):
        """
        Initialize HDF5 data loader.
        
        Parameters
        ----------
        data_dir : str or Path
            Directory containing HDF5 files
        file_pattern : str
            File pattern to match
        data_key : str
            Key for data in HDF5 files
        lazy_load : bool
            Whether to load data on demand
        """
        self.file_pattern = file_pattern
        self.data_key = data_key
        super().__init__(data_dir, lazy_load)
    
    def _scan_directory(self):
        """Scan directory for HDF5 files."""
        self.file_list = list(self.data_dir.glob(self.file_pattern))
        
        if not self.file_list:
            warnings.warn(f"No files matching pattern '{self.file_pattern}' found in {self.data_dir}")
    
    def load_file(self, file_path: Union[str, Path]) -> Field:
        """Load a single HDF5 file."""
        file_path = Path(file_path)
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Load data
                if self.data_key in f:
                    data = f[self.data_key][:]
                else:
                    # Try to find data automatically
                    keys = list(f.keys())
                    if len(keys) == 1:
                        data = f[keys[0]][:]
                    else:
                        raise KeyError(f"Data key '{self.data_key}' not found in {file_path}")
                
                # Load metadata
                metadata = {}
                for key in f.attrs.keys():
                    metadata[key] = f.attrs[key]
                
                # Try to load additional datasets as metadata
                for key in f.keys():
                    if key != self.data_key and key not in ['data', 'field']:
                        try:
                            metadata[key] = f[key][:]
                        except:
                            pass
                
                return Field(
                    data=data,
                    metadata=metadata
                )
        
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")


class BioelectricDataLoader(BaseDataLoader):
    """Load bioelectric imaging data."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        file_pattern: str = "*.h5",
        voltage_key: str = "voltage_field",
        time_key: str = "timestamps",
        lazy_load: bool = True
    ):
        """
        Initialize bioelectric data loader.
        
        Parameters
        ----------
        data_dir : str or Path
            Directory containing bioelectric data
        file_pattern : str
            File pattern to match
        voltage_key : str
            Key for voltage field data
        time_key : str
            Key for timestamp data
        lazy_load : bool
            Whether to load data on demand
        """
        self.file_pattern = file_pattern
        self.voltage_key = voltage_key
        self.time_key = time_key
        super().__init__(data_dir, lazy_load)
    
    def _scan_directory(self):
        """Scan directory for bioelectric files."""
        self.file_list = list(self.data_dir.glob(self.file_pattern))
        
        if not self.file_list:
            warnings.warn(f"No bioelectric files found in {self.data_dir}")
    
    def load_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a single bioelectric file."""
        file_path = Path(file_path)
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Load voltage field
                if self.voltage_key in f:
                    voltage_field = f[self.voltage_key][:]
                else:
                    raise KeyError(f"Voltage key '{self.voltage_key}' not found in {file_path}")
                
                # Load timestamps
                timestamps = None
                if self.time_key in f:
                    timestamps = f[self.time_key][:]
                
                # Load metadata
                metadata = {}
                for key in f.attrs.keys():
                    metadata[key] = f.attrs[key]
                
                # Load additional datasets
                additional_data = {}
                for key in f.keys():
                    if key not in [self.voltage_key, self.time_key]:
                        try:
                            additional_data[key] = f[key][:]
                        except:
                            pass
                
                return {
                    'voltage_field': voltage_field,
                    'timestamps': timestamps,
                    'metadata': metadata,
                    'additional_data': additional_data,
                    'file_path': file_path
                }
        
        except Exception as e:
            raise RuntimeError(f"Error loading bioelectric data from {file_path}: {e}")


class CSVDataLoader(BaseDataLoader):
    """Load data from CSV files."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        file_pattern: str = "*.csv",
        lazy_load: bool = True
    ):
        """
        Initialize CSV data loader.
        
        Parameters
        ----------
        data_dir : str or Path
            Directory containing CSV files
        file_pattern : str
            File pattern to match
        lazy_load : bool
            Whether to load data on demand
        """
        self.file_pattern = file_pattern
        super().__init__(data_dir, lazy_load)
    
    def _scan_directory(self):
        """Scan directory for CSV files."""
        self.file_list = list(self.data_dir.glob(self.file_pattern))
        
        if not self.file_list:
            warnings.warn(f"No CSV files found in {self.data_dir}")
    
    def load_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load a single CSV file."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.values
        except ImportError:
            # Fallback to numpy
            return np.loadtxt(file_path, delimiter=',')


class NumpyDataLoader(BaseDataLoader):
    """Load data from numpy files."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        file_pattern: str = "*.npy",
        lazy_load: bool = True
    ):
        """
        Initialize numpy data loader.
        
        Parameters
        ----------
        data_dir : str or Path
            Directory containing numpy files
        file_pattern : str
            File pattern to match
        lazy_load : bool
            Whether to load data on demand
        """
        self.file_pattern = file_pattern
        super().__init__(data_dir, lazy_load)
    
    def _scan_directory(self):
        """Scan directory for numpy files."""
        self.file_list = list(self.data_dir.glob(self.file_pattern))
        
        if not self.file_list:
            warnings.warn(f"No numpy files found in {self.data_dir}")
    
    def load_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load a single numpy file."""
        try:
            if file_path.suffix == '.npz':
                # Load npz file
                loaded = np.load(file_path)
                if len(loaded.files) == 1:
                    return loaded[loaded.files[0]]
                else:
                    # Return dictionary of arrays
                    return {key: loaded[key] for key in loaded.files}
            else:
                # Load npy file
                return np.load(file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading numpy file {file_path}: {e}")


class SyntheticDataLoader:
    """Load synthetic data generated by the generators module."""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize synthetic data loader.
        
        Parameters
        ----------
        data_path : str or Path
            Path to synthetic data file
        """
        self.data_path = Path(data_path)
        self.data = None
        self.metadata = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Synthetic data file not found: {self.data_path}")
        
        self._load_data()
    
    def _load_data(self):
        """Load synthetic data."""
        if self.data_path.suffix == '.npz':
            loaded = np.load(self.data_path, allow_pickle=True)
            self.data = {}
            self.metadata = {}
            
            for key in loaded.files:
                if key == 'metadata':
                    self.metadata = loaded[key].item()
                else:
                    self.data[key] = loaded[key]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def get_field_types(self) -> List[str]:
        """Get available field types."""
        return list(self.data.keys())
    
    def get_field(self, field_type: str) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Get field data by type."""
        if field_type not in self.data:
            raise KeyError(f"Field type '{field_type}' not found")
        
        return self.data[field_type]
    
    def get_metadata(self, field_type: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for field type or all metadata."""
        if field_type is None:
            return self.metadata
        elif field_type in self.metadata:
            return self.metadata[field_type]
        else:
            return {}


def load_bioelectric_measurements(
    file_path: Union[str, Path],
    voltage_key: str = "voltage",
    position_key: str = "positions",
    time_key: str = "timestamps"
) -> List[BioelectricMeasurement]:
    """
    Load bioelectric measurements from file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to data file
    voltage_key : str
        Key for voltage data
    position_key : str
        Key for position data
    time_key : str
        Key for timestamp data
        
    Returns
    -------
    measurements : List[BioelectricMeasurement]
        List of bioelectric measurements
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.h5':
        with h5py.File(file_path, 'r') as f:
            voltages = f[voltage_key][:]
            positions = f[position_key][:]
            timestamps = f[time_key][:]
            
            # Load metadata if available
            metadata = {}
            for key in f.attrs.keys():
                metadata[key] = f.attrs[key]
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Create measurements
    measurements = []
    for i in range(len(voltages)):
        measurement = BioelectricMeasurement(
            voltage=voltages[i],
            position=tuple(positions[i]),
            timestamp=timestamps[i],
            metadata=metadata
        )
        measurements.append(measurement)
    
    return measurements


def create_data_loader(
    data_dir: Union[str, Path],
    loader_type: str = "auto",
    **kwargs
) -> BaseDataLoader:
    """
    Create appropriate data loader based on data type.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing data
    loader_type : str
        Type of loader ('auto', 'hdf5', 'bioelectric', 'csv', 'numpy')
    **kwargs
        Additional arguments for loader
        
    Returns
    -------
    loader : BaseDataLoader
        Appropriate data loader
    """
    data_dir = Path(data_dir)
    
    if loader_type == "auto":
        # Auto-detect based on file extensions
        if any(data_dir.glob("*.h5")):
            # Check if it's bioelectric data
            sample_file = next(data_dir.glob("*.h5"))
            try:
                with h5py.File(sample_file, 'r') as f:
                    if 'voltage_field' in f or 'voltage' in f:
                        loader_type = "bioelectric"
                    else:
                        loader_type = "hdf5"
            except:
                loader_type = "hdf5"
        elif any(data_dir.glob("*.npy")) or any(data_dir.glob("*.npz")):
            loader_type = "numpy"
        elif any(data_dir.glob("*.csv")):
            loader_type = "csv"
        else:
            raise ValueError(f"Cannot auto-detect data type in {data_dir}")
    
    # Create loader
    if loader_type == "hdf5":
        return HDF5DataLoader(data_dir, **kwargs)
    elif loader_type == "bioelectric":
        return BioelectricDataLoader(data_dir, **kwargs)
    elif loader_type == "csv":
        return CSVDataLoader(data_dir, **kwargs)
    elif loader_type == "numpy":
        return NumpyDataLoader(data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")


def save_field_data(
    field: Field,
    file_path: Union[str, Path],
    format: str = "hdf5"
) -> None:
    """
    Save field data to file.
    
    Parameters
    ----------
    field : Field
        Field data to save
    file_path : str or Path
        Output file path
    format : str
        Output format ('hdf5', 'numpy')
    """
    file_path = Path(file_path)
    
    if format == "hdf5":
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=field.data)
            
            if field.coordinates is not None:
                f.create_dataset('coordinates', data=field.coordinates)
            
            if field.metadata is not None:
                for key, value in field.metadata.items():
                    try:
                        f.attrs[key] = value
                    except:
                        # Skip non-serializable metadata
                        pass
    
    elif format == "numpy":
        if field.metadata is not None:
            np.savez(file_path, data=field.data, metadata=field.metadata)
        else:
            np.save(file_path, field.data)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_field_data(file_path: Union[str, Path]) -> Field:
    """
    Load field data from file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to field data file
        
    Returns
    -------
    field : Field
        Loaded field data
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.h5':
        with h5py.File(file_path, 'r') as f:
            data = f['data'][:]
            
            coordinates = None
            if 'coordinates' in f:
                coordinates = f['coordinates'][:]
            
            metadata = {}
            for key in f.attrs.keys():
                metadata[key] = f.attrs[key]
            
            return Field(
                data=data,
                coordinates=coordinates,
                metadata=metadata
            )
    
    elif file_path.suffix == '.npz':
        loaded = np.load(file_path, allow_pickle=True)
        data = loaded['data']
        metadata = loaded['metadata'].item() if 'metadata' in loaded else {}
        
        return Field(data=data, metadata=metadata)
    
    elif file_path.suffix == '.npy':
        data = np.load(file_path)
        return Field(data=data)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")