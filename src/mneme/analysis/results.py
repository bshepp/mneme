"""Result management and export utilities."""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from dataclasses import asdict

from ..types import AnalysisResult, PersistenceDiagram, Attractor, Field
from ..utils.io import save_results, load_results


class ResultManager:
    """Manage and organize analysis results."""
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize result manager.
        
        Parameters
        ----------
        base_dir : str or Path
            Base directory for storing results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_dir / 'experiments').mkdir(exist_ok=True)
        (self.base_dir / 'summaries').mkdir(exist_ok=True)
        (self.base_dir / 'exports').mkdir(exist_ok=True)
    
    def save_result(
        self,
        result: AnalysisResult,
        experiment_name: Optional[str] = None,
        format: str = 'hdf5'
    ) -> Path:
        """
        Save analysis result.
        
        Parameters
        ----------
        result : AnalysisResult
            Analysis result to save
        experiment_name : str, optional
            Experiment name (uses result.experiment_id if None)
        format : str
            Storage format
            
        Returns
        -------
        file_path : Path
            Path to saved file
        """
        if experiment_name is None:
            experiment_name = result.experiment_id
        
        # Create experiment directory
        exp_dir = self.base_dir / 'experiments' / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save result
        file_path = exp_dir / f'result.{format}'
        save_results(result, file_path, format=format)
        
        # Save summary
        self._save_summary(result, exp_dir)
        
        return file_path
    
    def load_result(
        self,
        experiment_name: str,
        format: str = 'hdf5'
    ) -> AnalysisResult:
        """
        Load analysis result.
        
        Parameters
        ----------
        experiment_name : str
            Experiment name
        format : str
            Storage format
            
        Returns
        -------
        result : AnalysisResult
            Loaded analysis result
        """
        file_path = self.base_dir / 'experiments' / experiment_name / f'result.{format}'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Result file not found: {file_path}")
        
        return load_results(file_path, format=format)
    
    def list_experiments(self) -> List[str]:
        """
        List available experiments.
        
        Returns
        -------
        experiments : List[str]
            List of experiment names
        """
        exp_dir = self.base_dir / 'experiments'
        
        if not exp_dir.exists():
            return []
        
        experiments = []
        for item in exp_dir.iterdir():
            if item.is_dir():
                experiments.append(item.name)
        
        return sorted(experiments)
    
    def _save_summary(self, result: AnalysisResult, exp_dir: Path) -> None:
        """Save experiment summary."""
        summary = self._create_summary(result)
        
        summary_path = exp_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _create_summary(self, result: AnalysisResult) -> Dict[str, Any]:
        """Create experiment summary."""
        summary = {
            'experiment_id': result.experiment_id,
            'timestamp': result.timestamp,
            'data_info': {},
            'analysis_summary': {}
        }
        
        # Data information
        if result.raw_data is not None:
            summary['data_info']['raw_shape'] = result.raw_data.shape
            summary['data_info']['raw_dtype'] = str(result.raw_data.data.dtype)
        
        if result.processed_data is not None:
            summary['data_info']['processed_shape'] = result.processed_data.shape
        
        # Analysis summary
        if result.reconstruction is not None:
            summary['analysis_summary']['reconstruction'] = {
                'method': result.reconstruction.method.value if result.reconstruction.method else 'unknown',
                'field_shape': result.reconstruction.field.shape,
                'computation_time': result.reconstruction.computation_time
            }
        
        if result.topology is not None:
            summary['analysis_summary']['topology'] = {
                'n_diagrams': len(result.topology.diagrams) if result.topology.diagrams else 0,
                'total_features': sum(len(d.points) for d in result.topology.diagrams) if result.topology.diagrams else 0
            }
        
        if result.attractors:
            summary['analysis_summary']['attractors'] = {
                'n_attractors': len(result.attractors),
                'types': [a.type.value for a in result.attractors]
            }
        
        return summary
    
    def create_comparative_summary(self, experiment_names: List[str]) -> Dict[str, Any]:
        """
        Create comparative summary of multiple experiments.
        
        Parameters
        ----------
        experiment_names : List[str]
            List of experiment names to compare
            
        Returns
        -------
        summary : Dict[str, Any]
            Comparative summary
        """
        summary = {
            'experiments': experiment_names,
            'comparison': {}
        }
        
        results = []
        for exp_name in experiment_names:
            try:
                result = self.load_result(exp_name)
                results.append((exp_name, result))
            except FileNotFoundError:
                continue
        
        if not results:
            return summary
        
        # Compare reconstruction methods
        reconstruction_methods = []
        for exp_name, result in results:
            if result.reconstruction is not None and result.reconstruction.method is not None:
                reconstruction_methods.append(result.reconstruction.method.value)
            else:
                reconstruction_methods.append('unknown')
        
        summary['comparison']['reconstruction_methods'] = reconstruction_methods
        
        # Compare topology results
        topology_features = []
        for exp_name, result in results:
            if result.topology is not None and result.topology.diagrams:
                total_features = sum(len(d.points) for d in result.topology.diagrams)
                topology_features.append(total_features)
            else:
                topology_features.append(0)
        
        summary['comparison']['topology_features'] = topology_features
        
        # Compare attractors
        attractor_counts = []
        for exp_name, result in results:
            if result.attractors:
                attractor_counts.append(len(result.attractors))
            else:
                attractor_counts.append(0)
        
        summary['comparison']['attractor_counts'] = attractor_counts
        
        return summary
    
    def export_to_csv(self, experiment_name: str, output_path: Path) -> None:
        """
        Export results to CSV format.
        
        Parameters
        ----------
        experiment_name : str
            Experiment name
        output_path : Path
            Output CSV file path
        """
        result = self.load_result(experiment_name)
        
        # Create CSV data
        csv_data = []
        
        # Export topology features
        if result.topology is not None and result.topology.diagrams:
            for i, diagram in enumerate(result.topology.diagrams):
                for j, point in enumerate(diagram.points):
                    csv_data.append({
                        'experiment_id': result.experiment_id,
                        'data_type': 'topology',
                        'dimension': diagram.dimension,
                        'feature_id': j,
                        'birth': point[0],
                        'death': point[1],
                        'persistence': point[1] - point[0]
                    })
        
        # Export attractor data
        if result.attractors:
            for i, attractor in enumerate(result.attractors):
                csv_data.append({
                    'experiment_id': result.experiment_id,
                    'data_type': 'attractor',
                    'attractor_id': i,
                    'type': attractor.type.value,
                    'basin_size': attractor.basin_size,
                    'center_x': attractor.center[0] if len(attractor.center) > 0 else None,
                    'center_y': attractor.center[1] if len(attractor.center) > 1 else None,
                    'dimension': attractor.dimension
                })
        
        # Write CSV
        if csv_data:
            import pandas as pd
            df = pd.DataFrame(csv_data)
            df.to_csv(output_path, index=False)
    
    def export_to_matlab(self, experiment_name: str, output_path: Path) -> None:
        """
        Export results to MATLAB format.
        
        Parameters
        ----------
        experiment_name : str
            Experiment name
        output_path : Path
            Output .mat file path
        """
        try:
            from scipy.io import savemat
        except ImportError:
            raise ImportError("scipy is required for MATLAB export")
        
        result = self.load_result(experiment_name)
        
        # Prepare data for MATLAB
        matlab_data = {
            'experiment_id': result.experiment_id,
            'timestamp': result.timestamp
        }
        
        # Raw data
        if result.raw_data is not None:
            matlab_data['raw_data'] = result.raw_data.data
        
        # Processed data
        if result.processed_data is not None:
            matlab_data['processed_data'] = result.processed_data.data
        
        # Reconstruction
        if result.reconstruction is not None:
            matlab_data['reconstructed_field'] = result.reconstruction.field.data
            if result.reconstruction.uncertainty is not None:
                matlab_data['reconstruction_uncertainty'] = result.reconstruction.uncertainty
        
        # Topology
        if result.topology is not None and result.topology.diagrams:
            for i, diagram in enumerate(result.topology.diagrams):
                matlab_data[f'persistence_diagram_dim_{i}'] = diagram.points
        
        # Attractors
        if result.attractors:
            attractor_data = {
                'types': [a.type.value for a in result.attractors],
                'centers': np.array([a.center for a in result.attractors]),
                'basin_sizes': np.array([a.basin_size for a in result.attractors])
            }
            matlab_data['attractors'] = attractor_data
        
        # Save to MATLAB file
        savemat(output_path, matlab_data)
    
    def cleanup_old_results(self, days_old: int = 30) -> None:
        """
        Clean up old results.
        
        Parameters
        ----------
        days_old : int
            Number of days after which to consider results old
        """
        import time
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(days=days_old)
        
        for exp_name in self.list_experiments():
            exp_dir = self.base_dir / 'experiments' / exp_name
            
            # Check modification time
            mod_time = datetime.fromtimestamp(exp_dir.stat().st_mtime)
            
            if mod_time < cutoff_time:
                # Remove experiment directory
                import shutil
                shutil.rmtree(exp_dir)
                print(f"Removed old experiment: {exp_name}")


def create_results_report(
    results: List[AnalysisResult],
    output_path: Path,
    format: str = 'html'
) -> None:
    """
    Create comprehensive results report.
    
    Parameters
    ----------
    results : List[AnalysisResult]
        List of analysis results
    output_path : Path
        Output file path
    format : str
        Report format ('html', 'pdf', 'markdown')
    """
    if format == 'html':
        _create_html_report(results, output_path)
    elif format == 'markdown':
        _create_markdown_report(results, output_path)
    else:
        raise ValueError(f"Unsupported report format: {format}")


def _create_html_report(results: List[AnalysisResult], output_path: Path) -> None:
    """Create HTML report."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mneme Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .summary { background-color: #f9f9f9; padding: 15px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Mneme Analysis Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Total experiments: {}</p>
            <p>Generated on: {}</p>
        </div>
    """.format(len(results), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Add individual experiment results
    for i, result in enumerate(results):
        html_content += f"""
        <h2>Experiment {i+1}: {result.experiment_id}</h2>
        <div class="summary">
            <p><strong>Timestamp:</strong> {result.timestamp}</p>
        """
        
        if result.raw_data is not None:
            html_content += f"<p><strong>Raw data shape:</strong> {result.raw_data.shape}</p>"
        
        if result.topology is not None and result.topology.diagrams:
            total_features = sum(len(d.points) for d in result.topology.diagrams)
            html_content += f"<p><strong>Topological features:</strong> {total_features}</p>"
        
        if result.attractors:
            html_content += f"<p><strong>Attractors found:</strong> {len(result.attractors)}</p>"
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def _create_markdown_report(results: List[AnalysisResult], output_path: Path) -> None:
    """Create Markdown report."""
    from datetime import datetime
    
    markdown_content = f"""# Mneme Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- Total experiments: {len(results)}

## Experiments

"""
    
    for i, result in enumerate(results):
        markdown_content += f"""### Experiment {i+1}: {result.experiment_id}

- **Timestamp:** {result.timestamp}
"""
        
        if result.raw_data is not None:
            markdown_content += f"- **Raw data shape:** {result.raw_data.shape}\n"
        
        if result.topology is not None and result.topology.diagrams:
            total_features = sum(len(d.points) for d in result.topology.diagrams)
            markdown_content += f"- **Topological features:** {total_features}\n"
        
        if result.attractors:
            markdown_content += f"- **Attractors found:** {len(result.attractors)}\n"
        
        markdown_content += "\n"
    
    with open(output_path, 'w') as f:
        f.write(markdown_content)


def merge_results(results: List[AnalysisResult]) -> Dict[str, Any]:
    """
    Merge multiple analysis results into summary.
    
    Parameters
    ----------
    results : List[AnalysisResult]
        List of analysis results
        
    Returns
    -------
    merged : Dict[str, Any]
        Merged results summary
    """
    merged = {
        'n_experiments': len(results),
        'experiment_ids': [r.experiment_id for r in results],
        'timestamps': [r.timestamp for r in results]
    }
    
    # Merge topology results
    all_diagrams = []
    for result in results:
        if result.topology is not None and result.topology.diagrams:
            all_diagrams.extend(result.topology.diagrams)
    
    if all_diagrams:
        merged['topology'] = {
            'total_diagrams': len(all_diagrams),
            'total_features': sum(len(d.points) for d in all_diagrams)
        }
    
    # Merge attractor results
    all_attractors = []
    for result in results:
        if result.attractors:
            all_attractors.extend(result.attractors)
    
    if all_attractors:
        attractor_types = [a.type.value for a in all_attractors]
        unique_types, counts = np.unique(attractor_types, return_counts=True)
        
        merged['attractors'] = {
            'total_attractors': len(all_attractors),
            'types': dict(zip(unique_types, counts))
        }
    
    return merged