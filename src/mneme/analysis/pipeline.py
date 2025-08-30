"""Analysis pipeline for field memory detection."""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
import time
from datetime import datetime
import logging
from pathlib import Path

from ..types import (
    Field, AnalysisResult, ReconstructionResult, TopologyResult,
    PipelineStage, ExperimentConfig
)
from ..core.field_theory import FieldReconstructor
from ..core.topology import PersistentHomology
from ..core.topology import RipsComplex, AlphaComplex
from ..core.topology import field_to_point_cloud
from ..core.attractors import AttractorDetector
from ..data.preprocessors import FieldPreprocessor
from ..data.validation import DataValidator, QualityChecker


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    success: bool
    execution_time: float
    stage_results: Dict[str, Any]
    analysis_result: Optional[AnalysisResult] = None
    errors: Optional[List[str]] = None


class MnemePipeline:
    """Complete analysis pipeline for field memory detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Pipeline configuration
        """
        self.config = config
        self.stages = []
        self.stage_mapping = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize pipeline components."""
        # Quality checker
        self.quality_checker = QualityChecker()
        
        # Preprocessor
        if 'preprocessing' in self.config:
            preproc_config = self.config['preprocessing']
            steps = []
            
            if preproc_config.get('denoise', {}).get('enabled', False):
                p = {k: v for k, v in preproc_config['denoise'].items() if k != 'enabled'}
                steps.append(('denoise', p))
            
            if preproc_config.get('normalize', {}).get('enabled', False):
                p = {k: v for k, v in preproc_config['normalize'].items() if k != 'enabled'}
                steps.append(('normalize', p))
            
            if preproc_config.get('register', {}).get('enabled', False):
                p = {k: v for k, v in preproc_config['register'].items() if k != 'enabled'}
                steps.append(('register', p))
            
            if preproc_config.get('interpolate', {}).get('enabled', False):
                p = {k: v for k, v in preproc_config['interpolate'].items() if k != 'enabled'}
                steps.append(('interpolate', p))
            
            if steps:
                self.preprocessor = FieldPreprocessor(steps)
            else:
                self.preprocessor = None
        else:
            self.preprocessor = None
        
        # Field reconstructor
        if 'reconstruction' in self.config:
            recon_config = self.config['reconstruction']
            self.reconstructor = FieldReconstructor(
                method=recon_config.get('method', 'ift'),
                resolution=tuple(recon_config.get('resolution', (256, 256))),
                **recon_config.get('parameters', {})
            )
        else:
            self.reconstructor = None
        
        # Topology analyzer
        if 'topology' in self.config:
            topo_config = self.config['topology']
            backend = topo_config.get('backend', 'cubical')
            if backend == 'cubical':
                self.topology_analyzer = PersistentHomology(
                    max_dimension=topo_config.get('max_dimension', 2),
                    filtration=topo_config.get('filtration', 'sublevel'),
                    persistence_threshold=topo_config.get('persistence_threshold', 0.05)
                )
            elif backend == 'rips':
                self.topology_analyzer = RipsComplex(
                    max_dimension=topo_config.get('max_dimension', 2),
                    max_edge_length=topo_config.get('max_edge_length', np.inf)
                )
            elif backend == 'alpha':
                self.topology_analyzer = AlphaComplex(
                    max_dimension=topo_config.get('max_dimension', 2)
                )
            else:
                raise ValueError(f"Unknown topology backend: {backend}")
        else:
            self.topology_analyzer = None
        
        # Attractor detector
        if 'attractors' in self.config:
            attr_config = self.config['attractors']
            self.attractor_detector = AttractorDetector(
                method=attr_config.get('method', 'recurrence'),
                threshold=attr_config.get('threshold', 0.1),
                **attr_config.get('parameters', {})
            )
        else:
            self.attractor_detector = None
    
    def add_stage(
        self,
        name: str,
        stage_func: Callable,
        inputs: List[str],
        outputs: List[str],
        enabled: bool = True
    ) -> 'MnemePipeline':
        """
        Add processing stage to pipeline.
        
        Parameters
        ----------
        name : str
            Stage name
        stage_func : Callable
            Function to execute
        inputs : List[str]
            Input data keys
        outputs : List[str]  
            Output data keys
        enabled : bool
            Whether stage is enabled
            
        Returns
        -------
        self : MnemePipeline
            Pipeline instance for chaining
        """
        stage = PipelineStage(
            name=name,
            enabled=enabled,
            inputs=inputs,
            outputs=outputs
        )
        
        self.stages.append(stage)
        self.stage_mapping[name] = stage_func
        
        return self
    
    def run(self, data: Union[Dict[str, Any], Field, np.ndarray]) -> PipelineResult:
        """
        Execute full pipeline on data.
        
        Parameters
        ----------
        data : Union[Dict[str, Any], Field, np.ndarray]
            Input data
            
        Returns
        -------
        result : PipelineResult
            Pipeline execution result
        """
        start_time = time.time()
        stage_results = {}
        errors = []
        
        try:
            # Prepare data
            if isinstance(data, np.ndarray):
                data_dict = {'field': Field(data=data)}
            elif isinstance(data, Field):
                data_dict = {'field': data}
            else:
                data_dict = data.copy()
            
            # Run standard pipeline
            if not self.stages:
                # Use default pipeline
                stage_results = self._run_default_pipeline(data_dict)
            else:
                # Run custom pipeline
                stage_results = self._run_custom_pipeline(data_dict)
            
            # Create analysis result
            analysis_result = self._create_analysis_result(data_dict, stage_results)
            
            execution_time = time.time() - start_time
            
            return PipelineResult(
                success=True,
                execution_time=execution_time,
                stage_results=stage_results,
                analysis_result=analysis_result
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            errors.append(str(e))
            self.logger.error(f"Pipeline execution failed: {e}")
            
            return PipelineResult(
                success=False,
                execution_time=execution_time,
                stage_results=stage_results,
                errors=errors
            )
    
    def _run_default_pipeline(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run default pipeline stages."""
        stage_results = {}
        
        # Stage 1: Quality check
        if 'field' in data_dict:
            quality_report = self.quality_checker.check_field(data_dict['field'])
            stage_results['quality_check'] = quality_report
            
            if quality_report['overall_quality'] == 'poor':
                self.logger.warning("Poor data quality detected")
        
        # Stage 2: Preprocessing
        if self.preprocessor is not None and 'field' in data_dict:
            field = data_dict['field']
            processed_data = self.preprocessor.fit_transform(field.data)
            processed_field = Field(
                data=processed_data,
                coordinates=field.coordinates,
                resolution=field.resolution,
                metadata=field.metadata
            )
            data_dict['processed_field'] = processed_field
            stage_results['preprocessing'] = {
                'parameters': self.preprocessor.get_step_parameters(),
                'shape_change': f"{field.data.shape} -> {processed_data.shape}"
            }
        
        return stage_results

    def _run_custom_pipeline(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run user-configured stages in order."""
        stage_results: Dict[str, Any] = {}
        for stage in self.stages:
            if not stage.enabled:
                continue
            stage_func = self.stage_mapping.get(stage.name)
            if stage_func is None:
                continue
            # Prepare inputs for stage
            inputs_payload = {k: data_dict.get(k) for k in stage.inputs}
            try:
                output_value = stage_func(inputs_payload)
            except Exception as exc:
                self.logger.error(f"Stage '{stage.name}' failed: {exc}")
                raise

            # Map outputs
            if isinstance(output_value, dict):
                for key, value in zip(stage.outputs, [output_value.get(o) for o in stage.outputs]):
                    data_dict[key] = value
            else:
                if len(stage.outputs) == 1:
                    data_dict[stage.outputs[0]] = output_value
                else:
                    # If multiple outputs expected but single value returned, store under first key
                    data_dict[stage.outputs[0]] = output_value

            # Record stage summary
            stage_results[stage.name] = {
                'inputs': stage.inputs,
                'outputs': stage.outputs
            }

        return stage_results

    def _create_analysis_result(self, data_dict: Dict[str, Any], stage_results: Dict[str, Any]) -> AnalysisResult:
        """Assemble AnalysisResult from available components.

        This method is deliberately lightweight for MVP: it wraps the raw/processed
        data, performs optional topology on reasonably sized fields, and only runs
        attractor detection when a temporal sequence is provided. Reconstruction
        defaults to identity if no sparse observations are available.
        """
        # Raw field
        raw_field: Field
        raw_input = data_dict.get('field')
        if isinstance(raw_input, Field):
            raw_field = raw_input
        else:
            raw_field = Field(data=np.asarray(raw_input))

        # Processed field if present
        processed_field: Optional[Field] = data_dict.get('processed_field')

        # Choose field for analysis (prefer processed)
        analysis_field = processed_field if processed_field is not None else raw_field

        # Optional topology (downsample for speed if needed)
        topology_result: Optional[TopologyResult] = None
        if self.topology_analyzer is not None and analysis_field.data.ndim == 2:
            field_for_tda = analysis_field.data
            try:
                h, w = field_for_tda.shape
                max_dim = max(h, w)
                backend_name = getattr(self.topology_analyzer.__class__, '__name__', '').lower()
                if 'rips' in backend_name or 'alpha' in backend_name:
                    # Convert field to point cloud for point-cloud backends
                    pc = field_to_point_cloud(field_for_tda, method='peaks', percentile=95.0, max_points=2000)
                    diagrams = self.topology_analyzer.compute_persistence(pc)
                else:
                    if max_dim > 128:
                        stride = int(np.ceil(max_dim / 128))
                        field_for_tda = field_for_tda[::stride, ::stride]
                    diagrams = self.topology_analyzer.compute_persistence(field_for_tda)
                features = self.topology_analyzer.extract_features(diagrams) if diagrams is not None else None
                topology_result = TopologyResult(diagrams=diagrams, features=features)
                stage_results['topology'] = {
                    'n_diagrams': len(diagrams) if diagrams else 0,
                    'total_features': int(sum(len(d.points) for d in diagrams)) if diagrams else 0,
                    'feature_vector_length': int(len(features)) if features is not None else 0,
                    'backend': 'rips' if 'rips' in backend_name else ('alpha' if 'alpha' in backend_name else 'cubical'),
                }
            except Exception as exc:
                self.logger.warning(f"Topology analysis failed: {exc}")

        # Optional attractors (only if temporal data provided)
        attractors = None
        if self.attractor_detector is not None and analysis_field.data.ndim == 3:
            try:
                # Flatten spatial dimensions into features
                t, h, w = analysis_field.data.shape
                trajectory = analysis_field.data.reshape(t, h * w)
                attractors = self.attractor_detector.detect(trajectory)
                stage_results['attractors'] = {
                    'n_attractors': len(attractors),
                    'attractor_types': [a.type.value for a in attractors],
                }
            except Exception as exc:
                self.logger.warning(f"Attractor detection failed: {exc}")

        # Reconstruction (identity fallback to avoid heavy compute without sparse obs)
        reconstruction = None
        try:
            if self.reconstructor is not None and 'observations' in data_dict and 'positions' in data_dict:
                observations = np.asarray(data_dict['observations'])
                positions = np.asarray(data_dict['positions'])
                reconstruction = self.reconstructor.fit_reconstruct(observations, positions)
            else:
                # Identity reconstruction: wrap current analysis field
                reconstruction = ReconstructionResult(
                    field=Field(
                        data=analysis_field.data,
                        coordinates=analysis_field.coordinates,
                        resolution=analysis_field.resolution,
                        metadata=(analysis_field.metadata or {}) | {'note': 'identity_reconstruction'}
                    ),
                    uncertainty=None,
                    method=None,
                    parameters=None,
                    computation_time=0.0,
                )
        except Exception as exc:
            self.logger.warning(f"Reconstruction step failed: {exc}")

        # Compose AnalysisResult
        analysis_result = AnalysisResult(
            experiment_id=f"mneme_{int(time.time())}",
            timestamp=datetime.utcnow().isoformat(),
            raw_data=raw_field,
            processed_data=processed_field,
            reconstruction=reconstruction,
            topology=topology_result,
            attractors=attractors,
            metadata={'stage_results_keys': list(stage_results.keys())}
        )

        return analysis_result


def create_standard_pipeline(config: Optional[Dict[str, Any]] = None) -> MnemePipeline:
    """Create standard analysis pipeline."""
    if config is None:
        config = {
            'preprocessing': {
                'denoise': {'enabled': True, 'method': 'gaussian', 'sigma': 1.0},
                'normalize': {'enabled': True, 'method': 'z_score'},
                'register': {'enabled': False},
                'interpolate': {'enabled': True, 'target_shape': (256, 256)}
            },
            'reconstruction': {
                'method': 'gaussian_process',
                'resolution': (256, 256),
                'parameters': {'kernel': 'rbf', 'length_scale': 10.0}
            },
            'topology': {
                'backend': 'cubical',  # 'cubical' | 'rips' | 'alpha'
                'max_dimension': 2,
                'filtration': 'sublevel',
                'persistence_threshold': 0.05
            },
            'attractors': {
                'method': 'recurrence',
                'threshold': 0.1,
                'parameters': {'min_persistence': 0.1}
            }
        }
    
    return MnemePipeline(config)


def create_bioelectric_pipeline(config: Optional[Dict[str, Any]] = None) -> MnemePipeline:
    """Create a bioelectric-focused analysis pipeline.

    This is a lightweight wrapper around the standard pipeline with
    bioelectric-appropriate defaults. It can be extended later.
    """
    if config is None:
        config = {
            'preprocessing': {
                # Use lightweight, broadly compatible defaults for MVP
                'denoise': {'enabled': True, 'method': 'gaussian', 'sigma': 1.0},
                'normalize': {'enabled': True, 'method': 'z_score', 'per_frame': True},
                # Registration requires temporal (3D) data; disable by default for 2D fields
                'register': {'enabled': False},
                # Linear interpolation is much faster than cubic for MVP
                'interpolate': {'enabled': True, 'target_shape': (256, 256), 'method': 'linear'},
            },
            'reconstruction': {
                'method': 'ift',
                'resolution': (256, 256),
                'parameters': {}
            },
            'topology': {
                'backend': 'cubical',  # 'cubical' | 'rips' | 'alpha'
                'max_dimension': 2,
                'filtration': 'sublevel',
                'persistence_threshold': 0.05
            },
            'attractors': {
                'method': 'recurrence',
                'threshold': 0.1,
                'parameters': {}
            }
        }

    return MnemePipeline(config)


    
    