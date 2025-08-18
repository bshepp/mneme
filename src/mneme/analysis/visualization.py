"""Visualization utilities for field analysis results."""

from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from ..types import Field, PersistenceDiagram, Attractor, AnalysisResult


class FieldVisualizer:
    """Visualize fields and analysis results."""

    def __init__(self, style: str = "publication", figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize field visualizer.

        Parameters
        ----------
        style : str
            Plotting style preset
        figsize : Tuple[int, int]
            Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.setup_style()

    def setup_style(self) -> None:
        """Setup matplotlib style."""
        if self.style == "publication":
            plt.style.use("seaborn-v0_8-whitegrid")
            plt.rcParams.update(
                {
                    "font.size": 12,
                    "axes.labelsize": 14,
                    "axes.titlesize": 16,
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12,
                    "legend.fontsize": 12,
                    "figure.titlesize": 18,
                    "axes.spines.top": False,
                    "axes.spines.right": False,
                    "figure.dpi": 300,
                }
            )
        elif self.style == "presentation":
            plt.style.use("seaborn-v0_8-talk")
            plt.rcParams.update({"font.size": 14, "figure.dpi": 150})

    def plot_field(
        self,
        field: Union[Field, np.ndarray],
        title: Optional[str] = None,
        colormap: str = "viridis",
        show_colorbar: bool = True,
        ax: Optional[plt.Axes] = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """
        Plot 2D field with customizable appearance.

        Parameters
        ----------
        field : Field or np.ndarray
            Field to plot
        title : str, optional
            Plot title
        colormap : str
            Colormap name
        show_colorbar : bool
            Whether to show colorbar
        ax : plt.Axes, optional
            Existing axes to plot on
        **kwargs
            Additional arguments for imshow

        Returns
        -------
        fig : plt.Figure
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        # Extract data
        if isinstance(field, Field):
            data = field.data
            metadata = field.metadata or {}
        else:
            data = field
            metadata = {}

        # Handle 3D data (take first slice)
        if data.ndim == 3:
            data = data[0]
            if title is None:
                title = "Field (first time slice)"

        # Plot field
        im = ax.imshow(data, cmap=colormap, aspect="auto", **kwargs)

        if title:
            ax.set_title(title)

        # Add colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            if "units" in metadata:
                cbar.set_label(metadata["units"])

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        return fig

    def plot_field_sequence(
        self,
        fields: Union[List[Field], np.ndarray],
        fps: int = 10,
        title: Optional[str] = None,
        colormap: str = "viridis",
        **kwargs: Any,
    ) -> animation.FuncAnimation:
        """Create animation of field evolution."""
        if isinstance(fields, np.ndarray):
            if fields.ndim != 3:
                raise ValueError("Field sequence must be 3D (time, height, width)")
            field_data = fields
        else:
            field_data = np.stack([f.data for f in fields])

        fig, ax = plt.subplots(figsize=self.figsize)

        # Set up the plot
        vmin, vmax = np.percentile(field_data, [1, 99])
        im = ax.imshow(field_data[0], cmap=colormap, vmin=vmin, vmax=vmax, **kwargs)

        if title:
            ax.set_title(title)

        plt.colorbar(im, ax=ax)

        def animate(frame: int):
            im.set_array(field_data[frame])
            ax.set_title(f"{title or 'Field Evolution'} - Frame {frame}")
            return [im]

        ani = animation.FuncAnimation(
            fig, animate, frames=len(field_data), interval=1000 // fps, blit=True
        )

        return ani

    def plot_persistence_diagram(
        self,
        diagram: PersistenceDiagram,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """Plot topological persistence diagram."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        if len(diagram.points) == 0:
            ax.text(
                0.5,
                0.5,
                "No persistent features",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(title or f"Persistence Diagram (Dim {diagram.dimension})")
            return fig

        # Plot points
        birth = diagram.points[:, 0]
        death = diagram.points[:, 1]

        # Color by persistence
        persistence = death - birth
        scatter = ax.scatter(birth, death, c=persistence, cmap="viridis", **kwargs)

        # Add diagonal line
        lims = [min(birth.min(), death.min()), max(birth.max(), death.max())]
        ax.plot(lims, lims, "k--", alpha=0.5, label="y=x")

        # Labels and title
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(title or f"Persistence Diagram (Dim {diagram.dimension})")

        # Colorbar
        plt.colorbar(scatter, ax=ax, label="Persistence")

        # Equal aspect ratio
        ax.set_aspect("equal")

        return fig

    def plot_multiple_persistence_diagrams(
        self, diagrams: List[PersistenceDiagram], titles: Optional[List[str]] = None
    ) -> plt.Figure:
        """Plot multiple persistence diagrams."""
        num_diagrams = len(diagrams)
        cols = min(3, num_diagrams)
        rows = (num_diagrams + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

        if num_diagrams == 1:
            axes = [axes]  # type: ignore[assignment]
        elif rows == 1:
            axes = axes.flatten()  # type: ignore[assignment]
        else:
            axes = axes.flatten()  # type: ignore[assignment]

        for i, diagram in enumerate(diagrams):
            t = titles[i] if titles else f"Dimension {diagram.dimension}"
            self.plot_persistence_diagram(diagram, ax=axes[i], title=t)

        # Hide unused subplots
        for i in range(num_diagrams, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_attractor_portrait(
        self,
        trajectory: np.ndarray,
        attractors: List[Attractor],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """Plot phase space with detected attractors."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        # Plot trajectory
        if trajectory.shape[1] >= 2:
            ax.plot(trajectory[:, 0], trajectory[:, 1], "gray", alpha=0.3, linewidth=0.5)
            ax.scatter(trajectory[:, 0], trajectory[:, 1], c="lightgray", s=1, alpha=0.5)

        # Plot attractors
        colors = plt.cm.Set1(np.linspace(0, 1, len(attractors)))

        for i, attractor in enumerate(attractors):
            if attractor.trajectory_indices:
                # Plot attractor points
                attractor_points = trajectory[attractor.trajectory_indices]
                if attractor_points.shape[1] >= 2:
                    ax.scatter(
                        attractor_points[:, 0],
                        attractor_points[:, 1],
                        c=[colors[i]],
                        s=20,
                        alpha=0.8,
                        label=f"{attractor.type.value} (size: {attractor.basin_size:.2f})",
                    )

                    # Mark center
                    if len(attractor.center) >= 2:
                        ax.scatter(
                            attractor.center[0],
                            attractor.center[1],
                            c=[colors[i]],
                            s=100,
                            marker="x",
                            linewidth=3,
                        )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(title or "Phase Space Portrait")

        if attractors:
            ax.legend()

        return fig

    def plot_field_statistics(self, field: Union[Field, np.ndarray], ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Plot field statistics."""
        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig = ax.get_figure()
            ax1 = ax2 = ax

        # Extract data
        if isinstance(field, Field):
            data = field.data
        else:
            data = field

        # Flatten data for histogram
        flat_data = data.flatten()
        finite_data = flat_data[np.isfinite(flat_data)]

        # Histogram
        ax1.hist(finite_data, bins=50, alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Value Distribution")

        # Add statistics text
        stats_text = f"Mean: {np.mean(finite_data):.3f}\n"
        stats_text += f"Std: {np.std(finite_data):.3f}\n"
        stats_text += f"Min: {np.min(finite_data):.3f}\n"
        stats_text += f"Max: {np.max(finite_data):.3f}"

        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Spatial profile (if 2D)
        if data.ndim == 2:
            # Average profiles
            profile_x = np.mean(data, axis=0)
            profile_y = np.mean(data, axis=1)

            ax2.plot(profile_x, label="X profile")
            ax2.plot(profile_y, label="Y profile")
            ax2.set_xlabel("Position")
            ax2.set_ylabel("Average Value")
            ax2.set_title("Spatial Profiles")
            ax2.legend()

        return fig

    def plot_reconstruction_comparison(
        self,
        original: Union[Field, np.ndarray],
        reconstructed: Union[Field, np.ndarray],
        uncertainty: Optional[np.ndarray] = None,
    ) -> plt.Figure:
        """Compare original and reconstructed fields."""
        num_plots = 3 if uncertainty is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

        # Extract data
        orig_data = original.data if isinstance(original, Field) else original
        recon_data = reconstructed.data if isinstance(reconstructed, Field) else reconstructed

        # Plot original
        im1 = axes[0].imshow(orig_data, cmap="viridis", aspect="auto")
        axes[0].set_title("Original")
        plt.colorbar(im1, ax=axes[0])

        # Plot reconstructed
        im2 = axes[1].imshow(recon_data, cmap="viridis", aspect="auto")
        axes[1].set_title("Reconstructed")
        plt.colorbar(im2, ax=axes[1])

        # Plot uncertainty if available
        if uncertainty is not None:
            im3 = axes[2].imshow(uncertainty, cmap="Reds", aspect="auto")
            axes[2].set_title("Uncertainty")
            plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        return fig

    def create_analysis_dashboard(
        self, result: AnalysisResult, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create comprehensive analysis dashboard."""
        fig = plt.figure(figsize=(16, 12))

        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Original field
        ax1 = fig.add_subplot(gs[0, 0])
        if result.raw_data is not None:
            self.plot_field(result.raw_data, title="Original Field", ax=ax1)

        # Processed field
        ax2 = fig.add_subplot(gs[0, 1])
        if result.processed_data is not None:
            self.plot_field(result.processed_data, title="Processed Field", ax=ax2)

        # Reconstructed field
        ax3 = fig.add_subplot(gs[0, 2])
        if result.reconstruction is not None:
            self.plot_field(result.reconstruction.field, title="Reconstructed Field", ax=ax3)

        # Reconstruction uncertainty
        ax4 = fig.add_subplot(gs[0, 3])
        if result.reconstruction is not None and result.reconstruction.uncertainty is not None:
            im = ax4.imshow(result.reconstruction.uncertainty, cmap="Reds", aspect="auto")
            ax4.set_title("Reconstruction Uncertainty")
            plt.colorbar(im, ax=ax4)

        # Persistence diagrams
        if result.topology is not None and result.topology.diagrams:
            for i, diagram in enumerate(result.topology.diagrams[:3]):
                ax = fig.add_subplot(gs[1, i])
                self.plot_persistence_diagram(diagram, ax=ax)

        # Attractor analysis
        if result.attractors:
            ax_attr = fig.add_subplot(gs[2, :2])

            # Create simple trajectory for visualization
            if result.processed_data is not None and result.processed_data.data.ndim == 3:
                # Create trajectory from field evolution
                field_data = result.processed_data.data
                trajectory = field_data.reshape(field_data.shape[0], -1)

                # Subsample for visualization
                if trajectory.shape[1] > 100:
                    indices = np.random.choice(trajectory.shape[1], 100, replace=False)
                    trajectory = trajectory[:, indices]

                self.plot_attractor_portrait(trajectory, result.attractors, ax=ax_attr)

        # Statistics
        ax_stats = fig.add_subplot(gs[2, 2:])
        if result.raw_data is not None:
            self.plot_field_statistics(result.raw_data, ax=ax_stats)

        # Add title
        fig.suptitle(f"Analysis Dashboard - {result.experiment_id}", fontsize=16)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_pipeline_results(self, results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Plot pipeline stage results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Quality metrics
        if "quality_check" in results:
            quality = results["quality_check"]
            metrics = quality.get("metrics", {})

            # Plot quality metrics
            metric_names: List[str] = []
            metric_values: List[float] = []

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_names.append(key)
                    metric_values.append(float(value))

            if metric_names:
                axes[0].bar(range(len(metric_names)), metric_values)
                axes[0].set_xticks(range(len(metric_names)))
                axes[0].set_xticklabels(metric_names, rotation=45, ha="right")
                axes[0].set_title("Quality Metrics")

        # Topology features
        if "topology" in results:
            topo = results["topology"]
            if "feature_vector_length" in topo:
                axes[1].text(
                    0.5,
                    0.5,
                    f"Topological Features: {topo['feature_vector_length']}",
                    transform=axes[1].transAxes,
                    ha="center",
                    va="center",
                )
            axes[1].set_title("Topology Analysis")

        # Attractor summary
        if "attractors" in results:
            attr = results["attractors"]
            if "attractor_types" in attr:
                types = attr["attractor_types"]
                unique_types, counts = np.unique(types, return_counts=True)
                axes[2].bar(unique_types, counts)
                axes[2].set_title("Attractor Types")
                axes[2].set_xlabel("Type")
                axes[2].set_ylabel("Count")

        # Execution times
        times: List[float] = []
        labels: List[str] = []

        for stage_name, stage_result in results.items():
            if isinstance(stage_result, dict) and "computation_time" in stage_result:
                times.append(float(stage_result["computation_time"]))
                labels.append(stage_name)

        if times:
            axes[3].bar(labels, times)
            axes[3].set_title("Stage Execution Times")
            axes[3].set_ylabel("Time (s)")
            axes[3].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


def plot_bioelectric_gradient(field: np.ndarray, title: str = "Bioelectric Field") -> plt.Figure:
    """Plot bioelectric field with appropriate colormap."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use RdBu colormap for voltage data
    im = ax.imshow(field, cmap="RdBu_r", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Lateral Position")
    ax.set_ylabel("Anterior-Posterior Position")

    # Add colorbar with voltage units
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Voltage (mV)")

    return fig


def create_comparison_plot(fields: List[np.ndarray], titles: List[str]) -> plt.Figure:
    """Create comparison plot of multiple fields."""
    num_fields = len(fields)
    cols = min(3, num_fields)
    rows = (num_fields + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    if num_fields == 1:
        axes = [axes]  # type: ignore[assignment]
    elif rows == 1:
        axes = axes.flatten()  # type: ignore[assignment]
    else:
        axes = axes.flatten()  # type: ignore[assignment]

    for i, (field, t) in enumerate(zip(fields, titles)):
        im = axes[i].imshow(field, cmap="viridis", aspect="auto")
        axes[i].set_title(t)
        plt.colorbar(im, ax=axes[i])

    # Hide unused subplots
    for i in range(num_fields, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def save_animation(ani: animation.FuncAnimation, filename: str, fps: int = 10) -> None:
    """Save animation to file."""
    try:
        writer = animation.PillowWriter(fps=fps)
        ani.save(filename, writer=writer)
    except Exception as e:
        warnings.warn(f"Failed to save animation: {e}")
        print("To save animations, install pillow: pip install pillow")
