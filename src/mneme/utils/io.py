"""Input/Output utilities for Mneme."""

import numpy as np
import h5py
import pickle
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import warnings

from ..types import (
    Field,
    AnalysisResult,
    ReconstructionResult,
    TopologyResult,
    PersistenceDiagram,
    Attractor,
    AttractorType,
    ReconstructionMethod,
)


def save_results(
    results: Union[AnalysisResult, Dict[str, Any]],
    output_path: Union[str, Path],
    format: str = 'hdf5',
    compression: Optional[str] = None
) -> None:
    """
    Save analysis results to file.
    
    Parameters
    ----------
    results : AnalysisResult or Dict[str, Any]
        Results to save
    output_path : str or Path
        Output file path
    format : str
        Output format ('hdf5', 'pickle', 'json')
    compression : str, optional
        Compression method
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'hdf5':
        _save_results_hdf5(results, output_path, compression)
    elif format == 'pickle':
        _save_results_pickle(results, output_path)
    elif format == 'json':
        _save_results_json(results, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_results_hdf5(
    results: Union[AnalysisResult, Dict[str, Any]],
    output_path: Path,
    compression: Optional[str] = None
) -> None:
    """Save results to HDF5 format."""
    with h5py.File(output_path, 'w') as f:
        if isinstance(results, AnalysisResult):
            # Save AnalysisResult
            f.attrs['experiment_id'] = results.experiment_id
            f.attrs['timestamp'] = results.timestamp
            
            # Save raw data
            if results.raw_data is not None:
                _save_field_hdf5(results.raw_data, f.create_group('raw_data'), compression)
            
            # Save processed data
            if results.processed_data is not None:
                _save_field_hdf5(results.processed_data, f.create_group('processed_data'), compression)
            
            # Save reconstruction
            if results.reconstruction is not None:
                recon_group = f.create_group('reconstruction')
                _save_field_hdf5(results.reconstruction.field, recon_group.create_group('field'), compression)
                
                if results.reconstruction.uncertainty is not None:
                    recon_group.create_dataset('uncertainty', data=results.reconstruction.uncertainty, compression=compression)
                
                if results.reconstruction.method is not None:
                    recon_group.attrs['method'] = results.reconstruction.method.value
                
                if results.reconstruction.computation_time is not None:
                    recon_group.attrs['computation_time'] = results.reconstruction.computation_time
            
            # Save topology
            if results.topology is not None:
                topo_group = f.create_group('topology')
                
                # Save diagrams
                if results.topology.diagrams:
                    diag_group = topo_group.create_group('diagrams')
                    for i, diagram in enumerate(results.topology.diagrams):
                        diag_subgroup = diag_group.create_group(f'diagram_{i}')
                        diag_subgroup.create_dataset('points', data=diagram.points, compression=compression)
                        diag_subgroup.attrs['dimension'] = diagram.dimension
                        if diagram.threshold is not None:
                            diag_subgroup.attrs['threshold'] = diagram.threshold
                
                # Save features
                if results.topology.features is not None:
                    topo_group.create_dataset('features', data=results.topology.features, compression=compression)
            
            # Save attractors
            if results.attractors:
                attr_group = f.create_group('attractors')
                for i, attractor in enumerate(results.attractors):
                    attr_subgroup = attr_group.create_group(f'attractor_{i}')
                    attr_subgroup.attrs['type'] = attractor.type.value
                    attr_subgroup.create_dataset('center', data=attractor.center, compression=compression)
                    attr_subgroup.attrs['basin_size'] = attractor.basin_size
                    
                    if attractor.dimension is not None:
                        attr_subgroup.attrs['dimension'] = attractor.dimension
                    
                    if attractor.lyapunov_exponents is not None:
                        attr_subgroup.create_dataset('lyapunov_exponents', data=attractor.lyapunov_exponents, compression=compression)
                    
                    if attractor.trajectory_indices is not None:
                        attr_subgroup.create_dataset('trajectory_indices', data=attractor.trajectory_indices, compression=compression)
            
            # Save metadata
            if results.metadata is not None:
                _save_metadata_hdf5(results.metadata, f.create_group('metadata'))
        
        else:
            # Save dictionary
            _save_dict_hdf5(results, f, compression)


def _save_field_hdf5(field: Field, group: h5py.Group, compression: Optional[str] = None) -> None:
    """Save Field object to HDF5 group."""
    group.create_dataset('data', data=field.data, compression=compression)
    
    if field.coordinates is not None:
        group.create_dataset('coordinates', data=field.coordinates, compression=compression)
    
    if field.resolution is not None:
        group.attrs['resolution'] = field.resolution
    
    if field.bounds is not None:
        group.attrs['bounds'] = field.bounds
    
    if field.metadata is not None:
        _save_metadata_hdf5(field.metadata, group.create_group('metadata'))


def _save_metadata_hdf5(metadata: Dict[str, Any], group: h5py.Group) -> None:
    """Save metadata dictionary to HDF5 group."""
    for key, value in metadata.items():
        try:
            if isinstance(value, (str, int, float, bool)):
                group.attrs[key] = value
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, dict):
                _save_metadata_hdf5(value, group.create_group(key))
            elif isinstance(value, list):
                # Try to save as numpy array
                try:
                    group.create_dataset(key, data=np.array(value))
                except:
                    # Save as string if array conversion fails
                    group.attrs[key] = str(value)
            else:
                group.attrs[key] = str(value)
        except Exception as e:
            warnings.warn(f"Could not save metadata key '{key}': {e}")


def _save_dict_hdf5(data: Dict[str, Any], group: h5py.Group, compression: Optional[str] = None) -> None:
    """Save dictionary to HDF5 group."""
    for key, value in data.items():
        try:
            if isinstance(value, np.ndarray):
                group.create_dataset(key, data=value, compression=compression)
            elif isinstance(value, dict):
                _save_dict_hdf5(value, group.create_group(key), compression)
            elif isinstance(value, (str, int, float, bool)):
                group.attrs[key] = value
            else:
                group.attrs[key] = str(value)
        except Exception as e:
            warnings.warn(f"Could not save key '{key}': {e}")


def _save_results_pickle(results: Union[AnalysisResult, Dict[str, Any]], output_path: Path) -> None:
    """Save results to pickle format."""
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)


def _save_results_json(results: Union[AnalysisResult, Dict[str, Any]], output_path: Path) -> None:
    """Save results to JSON format (metadata only)."""
    if isinstance(results, AnalysisResult):
        # Extract JSON-serializable data
        json_data = {
            'experiment_id': results.experiment_id,
            'timestamp': results.timestamp,
            'metadata': results.metadata
        }
        
        # Add summary statistics
        if results.raw_data is not None:
            json_data['raw_data_shape'] = results.raw_data.shape
        
        if results.topology is not None and results.topology.diagrams:
            json_data['topology_summary'] = {
                'n_diagrams': len(results.topology.diagrams),
                'total_features': sum(len(d.points) for d in results.topology.diagrams)
            }
        
        if results.attractors:
            json_data['attractors_summary'] = {
                'n_attractors': len(results.attractors),
                'types': [a.type.value for a in results.attractors]
            }
    
    else:
        # Try to serialize dictionary
        json_data = _make_json_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)


def _make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return _make_json_serializable(obj.__dict__)
    else:
        return str(obj)


def load_results(
    input_path: Union[str, Path],
    format: Optional[str] = None
) -> Union[AnalysisResult, Dict[str, Any]]:
    """
    Load analysis results from file.
    
    Parameters
    ----------
    input_path : str or Path
        Input file path
    format : str, optional
        Input format (auto-detected if None)
        
    Returns
    -------
    results : AnalysisResult or Dict[str, Any]
        Loaded results
    """
    input_path = Path(input_path)
    
    if format is None:
        # Auto-detect format (support both .h5 and .hdf5)
        if input_path.suffix.lower() in ('.h5', '.hdf5'):
            format = 'hdf5'
        elif input_path.suffix.lower() in ('.pkl', '.pickle'):
            format = 'pickle'
        elif input_path.suffix.lower() == '.json':
            format = 'json'
        else:
            raise ValueError(f"Cannot auto-detect format for {input_path}")
    
    if format == 'hdf5':
        return _load_results_hdf5(input_path)
    elif format == 'pickle':
        return _load_results_pickle(input_path)
    elif format == 'json':
        return _load_results_json(input_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_results_hdf5(input_path: Path) -> AnalysisResult:
    """Load results from HDF5 format and return an AnalysisResult instance."""
    with h5py.File(input_path, 'r') as f:
        # Top-level attributes
        experiment_id = str(f.attrs.get('experiment_id', 'unknown'))
        timestamp = str(f.attrs.get('timestamp', ''))

        # Raw data
        raw_data: Optional[Field] = None
        if 'raw_data' in f:
            raw_group = f['raw_data']
            raw_data = _hdf5_group_to_field(raw_group)

        # Processed data
        processed_data: Optional[Field] = None
        if 'processed_data' in f:
            proc_group = f['processed_data']
            processed_data = _hdf5_group_to_field(proc_group)

        # Reconstruction
        reconstruction: Optional[ReconstructionResult] = None
        if 'reconstruction' in f:
            recon_group = f['reconstruction']
            field_group = recon_group.get('field')
            recon_field = _hdf5_group_to_field(field_group) if field_group is not None else None
            uncertainty = recon_group.get('uncertainty')
            uncertainty_arr = uncertainty[()] if uncertainty is not None else None
            method_attr = recon_group.attrs.get('method')
            method = None
            if isinstance(method_attr, (bytes, str)):
                try:
                    method = ReconstructionMethod(method_attr.decode() if isinstance(method_attr, bytes) else method_attr)
                except Exception:
                    method = None
            comp_time = recon_group.attrs.get('computation_time')
            if recon_field is not None:
                reconstruction = ReconstructionResult(
                    field=recon_field,
                    uncertainty=uncertainty_arr,
                    method=method,
                    parameters=None,
                    computation_time=float(comp_time) if comp_time is not None else None,
                )

        # Topology
        topology: Optional[TopologyResult] = None
        if 'topology' in f:
            topo_group = f['topology']
            diagrams = []
            if 'diagrams' in topo_group:
                diags_group = topo_group['diagrams']
                for key in sorted(diags_group.keys()):
                    dgrp = diags_group[key]
                    points = dgrp['points'][()]
                    dim = int(dgrp.attrs.get('dimension', 0))
                    thr = dgrp.attrs.get('threshold')
                    diag = PersistenceDiagram(points=points, dimension=dim, threshold=float(thr) if thr is not None else None)
                    diagrams.append(diag)
            features = topo_group['features'][()] if 'features' in topo_group else None
            topology = TopologyResult(diagrams=diagrams, features=features)

        # Attractors
        attractors: Optional[List[Attractor]] = None
        if 'attractors' in f:
            attr_group = f['attractors']
            attractors = []
            for key in sorted(attr_group.keys()):
                ag = attr_group[key]
                atype = str(ag.attrs.get('type', 'fixed_point'))
                center = ag['center'][()]
                basin = float(ag.attrs.get('basin_size', 0.0))
                dim = ag.attrs.get('dimension')
                lyap = ag.get('lyapunov_exponents')
                lyap_arr = lyap[()] if lyap is not None else None
                idx = ag.get('trajectory_indices')
                idx_list = idx[()].tolist() if idx is not None else None
                attractors.append(Attractor(
                    type=AttractorType(atype),
                    center=center,
                    basin_size=basin,
                    dimension=float(dim) if dim is not None else None,
                    lyapunov_exponents=lyap_arr,
                    trajectory_indices=idx_list,
                ))
            if not attractors:
                attractors = None

        # Metadata (load as nested dict)
        metadata = None
        if 'metadata' in f:
            metadata = _load_hdf5_item(f['metadata'])

        return AnalysisResult(
            experiment_id=experiment_id,
            timestamp=timestamp,
            raw_data=raw_data if raw_data is not None else Field(data=np.array([])),
            processed_data=processed_data,
            reconstruction=reconstruction,
            topology=topology,
            attractors=attractors,
            metadata=metadata,
        )

def _hdf5_group_to_field(group: h5py.Group) -> Field:
    """Convert HDF5 group with 'data' and optional metadata into a Field."""
    data = group['data'][()]
    coords = group.get('coordinates')
    coords_arr = coords[()] if coords is not None else None
    resolution = group.attrs.get('resolution')
    bounds = group.attrs.get('bounds')
    meta = None
    if 'metadata' in group:
        meta = _load_hdf5_item(group['metadata'])
    return Field(
        data=data,
        coordinates=coords_arr,
        resolution=tuple(resolution) if resolution is not None else None,
        bounds=bounds if bounds is not None else None,
        metadata=meta,
    )


def _load_hdf5_item(item: Union[h5py.Dataset, h5py.Group]) -> Any:
    """Load item from HDF5."""
    if isinstance(item, h5py.Dataset):
        return item[:]
    elif isinstance(item, h5py.Group):
        result = {}
        
        # Load attributes
        for key in item.attrs.keys():
            result[key] = item.attrs[key]
        
        # Load sub-items
        for key in item.keys():
            result[key] = _load_hdf5_item(item[key])
        
        return result
    else:
        return item


def _load_results_pickle(input_path: Path) -> Union[AnalysisResult, Dict[str, Any]]:
    """Load results from pickle format."""
    with open(input_path, 'rb') as f:
        return pickle.load(f)


def _load_results_json(input_path: Path) -> Dict[str, Any]:
    """Load results from JSON format."""
    with open(input_path, 'r') as f:
        return json.load(f)


def create_experiment_directory(
    base_dir: Union[str, Path],
    experiment_id: str,
    create_subdirs: bool = True
) -> Path:
    """
    Create directory structure for experiment.
    
    Parameters
    ----------
    base_dir : str or Path
        Base directory for experiments
    experiment_id : str
        Experiment identifier
    create_subdirs : bool
        Whether to create subdirectories
        
    Returns
    -------
    exp_dir : Path
        Experiment directory
    """
    base_dir = Path(base_dir)
    exp_dir = base_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    if create_subdirs:
        # Create standard subdirectories
        subdirs = ['data', 'results', 'plots', 'logs', 'configs']
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(exist_ok=True)
    
    return exp_dir


def save_experiment_config(
    config: Dict[str, Any],
    experiment_dir: Path,
    filename: str = 'config.yaml'
) -> None:
    """
    Save experiment configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    experiment_dir : Path
        Experiment directory
    filename : str
        Configuration filename
    """
    import yaml
    
    config_path = experiment_dir / 'configs' / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_experiment_config(
    experiment_dir: Path,
    filename: str = 'config.yaml'
) -> Dict[str, Any]:
    """
    Load experiment configuration.
    
    Parameters
    ----------
    experiment_dir : Path
        Experiment directory
    filename : str
        Configuration filename
        
    Returns
    -------
    config : Dict[str, Any]
        Configuration dictionary
    """
    import yaml
    
    config_path = experiment_dir / 'configs' / filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def list_experiments(base_dir: Union[str, Path]) -> List[str]:
    """
    List available experiments.
    
    Parameters
    ----------
    base_dir : str or Path
        Base directory for experiments
        
    Returns
    -------
    experiments : List[str]
        List of experiment IDs
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        return []
    
    experiments = []
    for item in base_dir.iterdir():
        if item.is_dir():
            experiments.append(item.name)
    
    return sorted(experiments)


def cleanup_experiment(
    experiment_dir: Path,
    keep_results: bool = True,
    keep_configs: bool = True
) -> None:
    """
    Clean up experiment directory.
    
    Parameters
    ----------
    experiment_dir : Path
        Experiment directory
    keep_results : bool
        Whether to keep results
    keep_configs : bool
        Whether to keep configurations
    """
    import shutil
    
    if not experiment_dir.exists():
        return
    
    # Remove temporary files
    for pattern in ['*.tmp', '*.temp', '*~']:
        for file in experiment_dir.glob(pattern):
            file.unlink()
    
    # Remove logs if requested
    logs_dir = experiment_dir / 'logs'
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
    
    # Remove data if requested
    if not keep_results:
        for subdir in ['data', 'results', 'plots']:
            subdir_path = experiment_dir / subdir
            if subdir_path.exists():
                shutil.rmtree(subdir_path)
    
    # Remove configs if requested
    if not keep_configs:
        configs_dir = experiment_dir / 'configs'
        if configs_dir.exists():
            shutil.rmtree(configs_dir)