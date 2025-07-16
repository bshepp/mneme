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
            self.logger.error(f\"Pipeline execution failed: {e}\")\n            \n            return PipelineResult(\n                success=False,\n                execution_time=execution_time,\n                stage_results=stage_results,\n                errors=errors\n            )\n    \n    def _run_default_pipeline(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Run default pipeline stages.\"\"\"\n        stage_results = {}\n        \n        # Stage 1: Quality check\n        if 'field' in data_dict:\n            quality_report = self.quality_checker.check_field(data_dict['field'])\n            stage_results['quality_check'] = quality_report\n            \n            if quality_report['overall_quality'] == 'poor':\n                self.logger.warning(\"Poor data quality detected\")\n        \n        # Stage 2: Preprocessing\n        if self.preprocessor is not None and 'field' in data_dict:\n            field = data_dict['field']\n            processed_data = self.preprocessor.fit_transform(field.data)\n            processed_field = Field(\n                data=processed_data,\n                coordinates=field.coordinates,\n                resolution=field.resolution,\n                metadata=field.metadata\n            )\n            data_dict['processed_field'] = processed_field\n            stage_results['preprocessing'] = {\n                'parameters': self.preprocessor.get_step_parameters(),\n                'shape_change': f\"{field.data.shape} -> {processed_data.shape}\"\n            }\n        \n        # Stage 3: Field reconstruction\n        if self.reconstructor is not None:\n            field_key = 'processed_field' if 'processed_field' in data_dict else 'field'\n            field = data_dict[field_key]\n            \n            # For reconstruction, we need observation points\n            # Generate synthetic observations if not provided\n            if 'observations' not in data_dict:\n                observations, positions = self._generate_synthetic_observations(field)\n                data_dict['observations'] = observations\n                data_dict['positions'] = positions\n            \n            reconstruction_result = self.reconstructor.fit_reconstruct(\n                data_dict['observations'],\n                data_dict['positions']\n            )\n            \n            data_dict['reconstruction'] = reconstruction_result\n            stage_results['reconstruction'] = {\n                'method': reconstruction_result.method.value,\n                'computation_time': reconstruction_result.computation_time,\n                'field_shape': reconstruction_result.field.shape\n            }\n        \n        # Stage 4: Topology analysis\n        if self.topology_analyzer is not None:\n            field_key = 'reconstruction' if 'reconstruction' in data_dict else 'processed_field' if 'processed_field' in data_dict else 'field'\n            \n            if field_key == 'reconstruction':\n                field_data = data_dict['reconstruction'].field.data\n            else:\n                field_data = data_dict[field_key].data\n            \n            # Use 2D slice if 3D\n            if field_data.ndim == 3:\n                field_data = field_data[0]  # First time slice\n            \n            diagrams = self.topology_analyzer.compute_persistence(field_data)\n            features = self.topology_analyzer.extract_features(diagrams)\n            \n            topology_result = TopologyResult(\n                diagrams=diagrams,\n                features=features,\n                computation_time=time.time()  # Placeholder\n            )\n            \n            data_dict['topology'] = topology_result\n            stage_results['topology'] = {\n                'n_diagrams': len(diagrams),\n                'feature_vector_length': len(features),\n                'total_features': sum(len(d.points) for d in diagrams)\n            }\n        \n        # Stage 5: Attractor detection\n        if self.attractor_detector is not None:\n            field_key = 'processed_field' if 'processed_field' in data_dict else 'field'\n            field = data_dict[field_key]\n            \n            if field.data.ndim == 3:  # Time series\n                # Create trajectory from field evolution\n                trajectory = self._field_to_trajectory(field.data)\n                attractors = self.attractor_detector.detect(trajectory)\n                \n                data_dict['attractors'] = attractors\n                stage_results['attractors'] = {\n                    'n_attractors': len(attractors),\n                    'attractor_types': [a.type.value for a in attractors]\n                }\n            else:\n                stage_results['attractors'] = {\n                    'n_attractors': 0,\n                    'note': 'Attractor detection requires temporal data'\n                }\n        \n        return stage_results\n    \n    def _run_custom_pipeline(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Run custom pipeline stages.\"\"\"\n        stage_results = {}\n        \n        # Execute stages in order\n        for stage in self.stages:\n            if not stage.enabled:\n                continue\n            \n            # Check inputs\n            for input_key in stage.inputs:\n                if input_key not in data_dict:\n                    raise ValueError(f\"Stage '{stage.name}' requires input '{input_key}'\")\n            \n            # Get stage function\n            stage_func = self.stage_mapping.get(stage.name)\n            if stage_func is None:\n                raise ValueError(f\"No function defined for stage '{stage.name}'\")\n            \n            # Prepare inputs\n            stage_inputs = {key: data_dict[key] for key in stage.inputs}\n            \n            # Execute stage\n            stage_outputs = stage_func(stage_inputs)\n            \n            # Store outputs\n            if isinstance(stage_outputs, dict):\n                for output_key in stage.outputs:\n                    if output_key in stage_outputs:\n                        data_dict[output_key] = stage_outputs[output_key]\n                \n                stage_results[stage.name] = stage_outputs\n            else:\n                # Single output\n                if len(stage.outputs) == 1:\n                    data_dict[stage.outputs[0]] = stage_outputs\n                    stage_results[stage.name] = stage_outputs\n        \n        return stage_results\n    \n    def _generate_synthetic_observations(self, field: Field) -> tuple:\n        \"\"\"Generate synthetic observations from field for reconstruction.\"\"\"\n        # Simple random sampling\n        if field.data.ndim == 2:\n            h, w = field.data.shape\n            n_obs = min(1000, h * w // 10)\n            \n            # Random positions\n            positions = np.random.rand(n_obs, 2)\n            positions[:, 0] *= h\n            positions[:, 1] *= w\n            \n            # Sample observations\n            observations = np.zeros(n_obs)\n            for i, (y, x) in enumerate(positions):\n                y_int, x_int = int(y), int(x)\n                if 0 <= y_int < h and 0 <= x_int < w:\n                    observations[i] = field.data[y_int, x_int]\n            \n            # Normalize positions to [0, 1]\n            positions[:, 0] /= h\n            positions[:, 1] /= w\n            \n            return observations, positions\n        \n        else:\n            raise ValueError(\"Observation generation only supports 2D fields\")\n    \n    def _field_to_trajectory(self, field_data: np.ndarray) -> np.ndarray:\n        \"\"\"Convert field time series to trajectory.\"\"\"\n        if field_data.ndim != 3:\n            raise ValueError(\"Field must be 3D (time, height, width)\")\n        \n        # Simple approach: flatten each time slice and use as trajectory points\n        trajectory = field_data.reshape(field_data.shape[0], -1)\n        \n        # Subsample if too large\n        if trajectory.shape[1] > 100:\n            indices = np.random.choice(trajectory.shape[1], 100, replace=False)\n            trajectory = trajectory[:, indices]\n        \n        return trajectory\n    \n    def _create_analysis_result(self, data_dict: Dict[str, Any], stage_results: Dict[str, Any]) -> AnalysisResult:\n        \"\"\"Create comprehensive analysis result.\"\"\"\n        import datetime\n        \n        # Get original field\n        raw_field = data_dict.get('field')\n        processed_field = data_dict.get('processed_field')\n        reconstruction = data_dict.get('reconstruction')\n        topology = data_dict.get('topology')\n        attractors = data_dict.get('attractors', [])\n        \n        result = AnalysisResult(\n            experiment_id=f\"exp_{int(time.time())}\",\n            timestamp=datetime.datetime.now().isoformat(),\n            raw_data=raw_field,\n            processed_data=processed_field,\n            reconstruction=reconstruction,\n            topology=topology,\n            attractors=attractors,\n            metadata={\n                'pipeline_config': self.config,\n                'stage_results': stage_results\n            }\n        )\n        \n        return result\n    \n    def run_stage(self, stage_name: str, data: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Run specific pipeline stage.\"\"\"\n        stage = None\n        for s in self.stages:\n            if s.name == stage_name:\n                stage = s\n                break\n        \n        if stage is None:\n            raise ValueError(f\"Stage '{stage_name}' not found\")\n        \n        stage_func = self.stage_mapping.get(stage_name)\n        if stage_func is None:\n            raise ValueError(f\"No function defined for stage '{stage_name}'\")\n        \n        # Prepare inputs\n        stage_inputs = {key: data[key] for key in stage.inputs}\n        \n        # Execute stage\n        return stage_func(stage_inputs)\n    \n    def get_stage_info(self) -> List[Dict[str, Any]]:\n        \"\"\"Get information about pipeline stages.\"\"\"\n        info = []\n        for stage in self.stages:\n            info.append({\n                'name': stage.name,\n                'enabled': stage.enabled,\n                'inputs': stage.inputs,\n                'outputs': stage.outputs,\n                'has_function': stage.name in self.stage_mapping\n            })\n        return info\n    \n    def visualize_pipeline(self) -> None:\n        \"\"\"Visualize pipeline structure.\"\"\"\n        print(\"Pipeline Structure:\")\n        print(\"==================\")\n        \n        for i, stage in enumerate(self.stages):\n            status = \"✓\" if stage.enabled else \"✗\"\n            print(f\"{i+1}. {status} {stage.name}\")\n            print(f\"   Inputs: {', '.join(stage.inputs)}\")\n            print(f\"   Outputs: {', '.join(stage.outputs)}\")\n            print()\n\n\ndef create_standard_pipeline(config: Optional[Dict[str, Any]] = None) -> MnemePipeline:\n    \"\"\"Create standard analysis pipeline.\"\"\"\n    if config is None:\n        config = {\n            'preprocessing': {\n                'denoise': {'enabled': True, 'method': 'gaussian', 'sigma': 1.0},\n                'normalize': {'enabled': True, 'method': 'z_score'},\n                'register': {'enabled': False},\n                'interpolate': {'enabled': True, 'target_shape': (256, 256)}\n            },\n            'reconstruction': {\n                'method': 'gaussian_process',\n                'resolution': (256, 256),\n                'parameters': {'kernel': 'rbf', 'length_scale': 10.0}\n            },\n            'topology': {\n                'max_dimension': 2,\n                'filtration': 'sublevel',\n                'persistence_threshold': 0.05\n            },\n            'attractors': {\n                'method': 'recurrence',\n                'threshold': 0.1,\n                'parameters': {'min_persistence': 0.1}\n            }\n        }\n    \n    return MnemePipeline(config)\n\n\ndef create_bioelectric_pipeline(config: Optional[Dict[str, Any]] = None) -> MnemePipeline:\n    \"\"\"Create pipeline optimized for bioelectric data.\"\"\"\n    if config is None:\n        config = {\n            'preprocessing': {\n                'denoise': {'enabled': True, 'method': 'median', 'kernel_size': 3},\n                'normalize': {'enabled': True, 'method': 'robust'},\n                'register': {'enabled': True, 'reference': 'first'},\n                'interpolate': {'enabled': True, 'target_shape': (256, 256)}\n            },\n            'reconstruction': {\n                'method': 'gaussian_process',\n                'resolution': (256, 256),\n                'parameters': {'kernel': 'matern', 'length_scale': 15.0}\n            },\n            'topology': {\n                'max_dimension': 1,  # Focus on connected components and loops\n                'filtration': 'sublevel',\n                'persistence_threshold': 0.1\n            },\n            'attractors': {\n                'method': 'clustering',\n                'threshold': 0.2,\n                'parameters': {'min_samples': 10}\n            }\n        }\n    \n    return MnemePipeline(config)\n\n\ndef create_synthetic_pipeline(config: Optional[Dict[str, Any]] = None) -> MnemePipeline:\n    \"\"\"Create pipeline for synthetic data testing.\"\"\"\n    if config is None:\n        config = {\n            'preprocessing': {\n                'denoise': {'enabled': False},  # Synthetic data is clean\n                'normalize': {'enabled': True, 'method': 'z_score'},\n                'register': {'enabled': False},\n                'interpolate': {'enabled': False}  # Synthetic data has controlled resolution\n            },\n            'reconstruction': {\n                'method': 'ift',\n                'resolution': (128, 128),\n                'parameters': {'correlation_length': 5.0}\n            },\n            'topology': {\n                'max_dimension': 2,\n                'filtration': 'sublevel',\n                'persistence_threshold': 0.01\n            },\n            'attractors': {\n                'method': 'lyapunov',\n                'threshold': 0.05,\n                'parameters': {'n_neighbors': 5}\n            }\n        }\n    \n    return MnemePipeline(config)