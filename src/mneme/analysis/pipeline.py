"""Analysis pipeline for field memory detection."""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path

from ..types import (
    Field, AnalysisResult, ReconstructionResult, TopologyResult,
    PipelineStage, ExperimentConfig
)
from ..core.field_theory import FieldReconstructor
from ..core.topology import PersistentHomology
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
                steps.append(('denoise', preproc_config['denoise']))
            
            if preproc_config.get('normalize', {}).get('enabled', False):
                steps.append(('normalize', preproc_config['normalize']))
            
            if preproc_config.get('register', {}).get('enabled', False):
                steps.append(('register', preproc_config['register']))
            
            if preproc_config.get('interpolate', {}).get('enabled', False):
                steps.append(('interpolate', preproc_config['interpolate']))
            
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
            self.topology_analyzer = PersistentHomology(
                max_dimension=topo_config.get('max_dimension', 2),
                filtration=topo_config.get('filtration', 'sublevel'),
                persistence_threshold=topo_config.get('persistence_threshold', 0.05)
            )
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
                'denoise': {'enabled': True, 'method': 'wavelet', 'threshold': 'soft'},
                'normalize': {'enabled': True, 'method': 'z_score', 'per_frame': True},
                'register': {'enabled': True, 'reference': 'first_frame'},
                'interpolate': {'enabled': True, 'target_shape': (256, 256)},
            },
            'reconstruction': {
                'method': 'ift',
                'resolution': (256, 256),
                'parameters': {}
            },
            'topology': {
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