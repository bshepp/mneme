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
@click.pass_context
def analyze(ctx, data_path, output, pipeline, format):
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
    
    # Create pipeline
    if pipeline == 'bioelectric':
        pipe = create_bioelectric_pipeline(ctx.obj.to_dict())
    else:
        pipe = create_standard_pipeline(ctx.obj.to_dict())
    
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


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()