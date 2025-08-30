"""Command-line interface for Mneme."""

import click
import numpy as np
from pathlib import Path
import sys
import yaml

from .analysis.pipeline import create_standard_pipeline, create_bioelectric_pipeline
from .data.generators import SyntheticFieldGenerator, generate_planarian_bioelectric_sequence
from .data.loaders import create_data_loader
from .utils.config import Config
from .utils.logging import setup_logging
from .utils.io import save_results, create_experiment_directory


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Mneme: Field Memory Analysis Tool."""
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(level=log_level, console=True)
    
    # Load configuration
    if config:
        ctx.obj = Config.from_yaml(config)
    else:
        ctx.obj = Config()


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='synthetic_data.npz',
              help='Output file path')
@click.option('--type', '-t', type=click.Choice(['gaussian_blob', 'sinusoidal', 'bioelectric']),
              default='bioelectric', help='Field type to generate')
@click.option('--shape', '-s', type=str, default='256,128',
              help='Field shape (height,width)')
@click.option('--timesteps', type=int, default=100,
              help='Number of timesteps for dynamic fields')
@click.option('--seed', type=int, default=42,
              help='Random seed')
def generate(output, type, shape, timesteps, seed):
    """Generate synthetic field data."""
    click.echo(f"Generating {type} field data...")
    
    # Parse shape
    try:
        shape = tuple(map(int, shape.split(',')))
    except ValueError:
        click.echo("Error: Invalid shape format. Use 'height,width'")
        return
    
    if type == 'bioelectric':
        # Generate bioelectric sequence
        data = generate_planarian_bioelectric_sequence(
            shape=shape,
            timesteps=timesteps,
            seed=seed
        )
    else:
        # Generate static field
        generator = SyntheticFieldGenerator(field_type=type, seed=seed)
        if type == 'gaussian_blob':
            params = {'n_centers': 3, 'sigma': 20}
        elif type == 'sinusoidal':
            params = {'frequency': 0.1, 'amplitude': 1.0}
        else:
            params = {}
        
        data = generator.generate_static(shape, params)
    
    # Save data
    np.savez(output, data=data, metadata={'type': type, 'shape': shape, 'seed': seed})
    click.echo(f"Saved to {output}")


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='results',
              help='Output directory')
@click.option('--pipeline', '-p', type=click.Choice(['standard', 'bioelectric']),
              default='standard', help='Pipeline type')
@click.option('--format', '-f', type=click.Choice(['hdf5', 'pickle']),
              default='hdf5', help='Output format')
@click.option('--topology-backend', type=click.Choice(['cubical', 'rips', 'alpha']),
              default=None, help='Choose topology backend (overrides config)')
@click.option('--attractor-method', type=click.Choice(['none', 'recurrence', 'lyapunov', 'clustering']),
              default=None, help='Choose attractor detection method (overrides config)')
@click.option('--attractor-threshold', type=float, default=None,
              help='Attractor detection threshold (method-specific)')
@click.option('--attractor-min-persistence', type=float, default=None,
              help='Recurrence: minimum persistence fraction')
@click.option('--attractor-embedding-dim', type=int, default=None,
              help='Recurrence/Clustering: embedding dimension for 1D series')
@click.option('--attractor-time-delay', type=int, default=None,
              help='Recurrence/Clustering: time delay for embedding')
@click.option('--attractor-n-neighbors', type=int, default=None,
              help='Lyapunov: number of neighbors')
@click.option('--attractor-evolution-time', type=int, default=None,
              help='Lyapunov: evolution time steps')
@click.option('--attractor-min-samples', type=int, default=None,
              help='Clustering: minimum samples per cluster')
@click.option('--attractor-clustering-method', type=click.Choice(['dbscan', 'kmeans']), default=None,
              help='Clustering: algorithm to use')
@click.pass_context
def analyze(ctx, data_path, output, pipeline, format, topology_backend,
            attractor_method, attractor_threshold, attractor_min_persistence,
            attractor_embedding_dim, attractor_time_delay,
            attractor_n_neighbors, attractor_evolution_time,
            attractor_min_samples, attractor_clustering_method):
    """Analyze field data."""
    click.echo(f"Analyzing data from {data_path}...")
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path(data_path)
    if data_path.is_dir():
        # Load from directory
        loader = create_data_loader(data_path)
        data = next(iter(loader))  # Load first file
    else:
        # Load single file
        if data_path.suffix == '.npz':
            loaded = np.load(data_path)
            data = loaded['data']
        elif data_path.suffix == '.npy':
            data = np.load(data_path)
        else:
            click.echo(f"Error: Unsupported file format: {data_path.suffix}")
            return
    
    # Create pipeline config and apply CLI overrides
    config = ctx.obj.to_dict()
    if topology_backend is not None:
        config.setdefault('topology', {})
        config['topology']['backend'] = topology_backend

    # Attractor overrides
    if attractor_method is not None:
        if attractor_method == 'none':
            # Disable attractors by removing section
            if 'attractors' in config:
                del config['attractors']
        else:
            config.setdefault('attractors', {})
            config['attractors']['method'] = attractor_method
            if attractor_threshold is not None:
                config['attractors']['threshold'] = attractor_threshold
            params = config['attractors'].get('parameters', {})
            if attractor_method == 'recurrence':
                if attractor_min_persistence is not None:
                    params['min_persistence'] = attractor_min_persistence
                if attractor_embedding_dim is not None:
                    params['embedding_dimension'] = attractor_embedding_dim
                if attractor_time_delay is not None:
                    params['time_delay'] = attractor_time_delay
            elif attractor_method == 'lyapunov':
                if attractor_n_neighbors is not None:
                    params['n_neighbors'] = attractor_n_neighbors
                if attractor_evolution_time is not None:
                    params['evolution_time'] = attractor_evolution_time
            elif attractor_method == 'clustering':
                if attractor_min_samples is not None:
                    params['min_samples'] = attractor_min_samples
                if attractor_clustering_method is not None:
                    params['clustering_method'] = attractor_clustering_method
                if attractor_embedding_dim is not None:
                    params['embedding_dimension'] = attractor_embedding_dim
                if attractor_time_delay is not None:
                    params['time_delay'] = attractor_time_delay
            if params:
                config['attractors']['parameters'] = params

    # Create pipeline
    if pipeline == 'bioelectric':
        pipe = create_bioelectric_pipeline(config)
    else:
        pipe = create_standard_pipeline(config)
    
    # Run analysis
    click.echo("Running analysis pipeline...")
    result = pipe.run(data)
    
    if result.success:
        # Save results
        output_file = output_dir / f"analysis_results.{format}"
        save_results(result.analysis_result, output_file, format=format)
        
        click.echo(f"Analysis completed successfully!")
        click.echo(f"Results saved to {output_file}")
        click.echo(f"Execution time: {result.execution_time:.2f}s")
    else:
        click.echo("Analysis failed!")
        if result.errors:
            for error in result.errors:
                click.echo(f"Error: {error}")


@cli.command()
@click.argument('results_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='plots',
              help='Output directory for plots')
@click.option('--format', '-f', type=click.Choice(['png', 'svg', 'pdf']),
              default='png', help='Plot format')
def visualize(results_path, output, format):
    """Visualize analysis results."""
    click.echo(f"Visualizing results from {results_path}...")
    
    # Load results
    from .utils.io import load_results
    results = load_results(results_path)
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    from .analysis.visualization import FieldVisualizer
    visualizer = FieldVisualizer()
    
    try:
        if 'analysis_result' in results or hasattr(results, 'raw_data'):
            # Full analysis result
            analysis_result = results.get('analysis_result', results)
            fig = visualizer.create_analysis_dashboard(analysis_result)
            
            output_file = output_dir / f"dashboard.{format}"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            click.echo(f"Dashboard saved to {output_file}")
        
        else:
            # Individual plots
            if 'field' in results:
                fig = visualizer.plot_field(results['field'])
                output_file = output_dir / f"field.{format}"
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                click.echo(f"Field plot saved to {output_file}")
    
    except Exception as e:
        click.echo(f"Error creating visualizations: {e}")


@cli.command()
@click.argument('experiment_name')
@click.option('--base-dir', '-b', type=click.Path(), default='experiments',
              help='Base directory for experiments')
@click.option('--data-path', '-d', type=click.Path(exists=True), required=True,
              help='Path to input data')
@click.option('--pipeline', '-p', type=click.Choice(['standard', 'bioelectric']),
              default='standard', help='Pipeline type')
@click.pass_context
def experiment(ctx, experiment_name, base_dir, data_path, pipeline):
    """Run complete experiment."""
    click.echo(f"Running experiment: {experiment_name}")
    
    # Create experiment directory
    from .utils.io import create_experiment_directory, save_experiment_config
    exp_dir = create_experiment_directory(base_dir, experiment_name)
    
    # Save configuration
    config_dict = ctx.obj.to_dict()
    config_dict['experiment'] = {
        'name': experiment_name,
        'data_path': str(data_path),
        'pipeline': pipeline
    }
    save_experiment_config(config_dict, exp_dir)
    
    # Load data
    click.echo("Loading data...")
    data_path = Path(data_path)
    if data_path.is_dir():
        loader = create_data_loader(data_path)
        data = next(iter(loader))
    else:
        if data_path.suffix == '.npz':
            loaded = np.load(data_path)
            data = loaded['data']
        elif data_path.suffix == '.npy':
            data = np.load(data_path)
        else:
            click.echo(f"Error: Unsupported file format: {data_path.suffix}")
            return
    
    # Create pipeline
    if pipeline == 'bioelectric':
        pipe = create_bioelectric_pipeline(config_dict)
    else:
        pipe = create_standard_pipeline(config_dict)
    
    # Run analysis
    click.echo("Running analysis...")
    result = pipe.run(data)
    
    if result.success:
        # Save results
        results_file = exp_dir / 'results' / 'analysis_results.h5'
        save_results(result.analysis_result, results_file, format='hdf5')
        
        # Create visualizations
        click.echo("Creating visualizations...")
        from .analysis.visualization import FieldVisualizer
        visualizer = FieldVisualizer()
        
        try:
            fig = visualizer.create_analysis_dashboard(result.analysis_result)
            plots_file = exp_dir / 'plots' / 'dashboard.png'
            fig.savefig(plots_file, dpi=300, bbox_inches='tight')
            
            # Pipeline results
            fig2 = visualizer.plot_pipeline_results(result.stage_results)
            pipeline_file = exp_dir / 'plots' / 'pipeline_results.png'
            fig2.savefig(pipeline_file, dpi=300, bbox_inches='tight')
            
        except Exception as e:
            click.echo(f"Warning: Could not create visualizations: {e}")
        
        click.echo(f"Experiment completed successfully!")
        click.echo(f"Results directory: {exp_dir}")
        click.echo(f"Execution time: {result.execution_time:.2f}s")
        
    else:
        click.echo("Experiment failed!")
        if result.errors:
            for error in result.errors:
                click.echo(f"Error: {error}")


@cli.command()
@click.option('--base-dir', '-b', type=click.Path(), default='experiments',
              help='Base directory for experiments')
def list_experiments(base_dir):
    """List available experiments."""
    from .utils.io import list_experiments
    
    experiments = list_experiments(base_dir)
    
    if experiments:
        click.echo("Available experiments:")
        for exp in experiments:
            click.echo(f"  - {exp}")
    else:
        click.echo("No experiments found.")


@cli.command()
def info():
    """Show system information."""
    import platform
    import torch
    
    click.echo("Mneme System Information")
    click.echo("========================")
    click.echo(f"Python version: {sys.version}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"NumPy version: {np.__version__}")
    
    if torch.cuda.is_available():
        click.echo(f"CUDA available: Yes")
        click.echo(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        click.echo(f"CUDA available: No")
    
    # Check optional dependencies
    optional_deps = ['gudhi', 'pysr', 'h5py', 'scipy', 'sklearn']
    click.echo("\nOptional dependencies:")
    
    for dep in optional_deps:
        try:
            __import__(dep)
            click.echo(f"  - {dep}: ✓")
        except ImportError:
            click.echo(f"  - {dep}: ✗")

    # Show default topology backend if present
    try:
        cfg = ctx.obj.to_dict() if isinstance(ctx.obj, Config) else {}
    except Exception:
        cfg = {}
    topo_backend = None
    if isinstance(cfg, dict):
        topo_backend = cfg.get('topology', {}).get('backend', 'cubical') if cfg.get('topology') else 'cubical'
    click.echo(f"\nDefault topology backend: {topo_backend}")

    # PySR / Julia status
    try:
        import pysr  # type: ignore
        from juliacall import Base  # type: ignore
        click.echo("PySR: ✓ (Julia available)")
    except Exception:
        try:
            import pysr  # type: ignore
            click.echo("PySR: ✓ (Julia will be installed at first use)")
        except Exception:
            click.echo("PySR: ✗")


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()