"""Configuration management utilities."""

import yaml
import json
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging


class Config:
    """Configuration manager for Mneme."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Parameters
        ----------
        config_dict : Dict[str, Any], optional
            Configuration dictionary
        """
        self._config = config_dict or {}
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to YAML configuration file
            
        Returns
        -------
        config : Config
            Configuration instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from JSON file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to JSON configuration file
            
        Returns
        -------
        config : Config
            Configuration instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = "MNEME_") -> 'Config':
        """
        Load configuration from environment variables.
        
        Parameters
        ----------
        prefix : str
            Prefix for environment variables
            
        Returns
        -------
        config : Config
            Configuration instance
        """
        config_dict = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Try to parse as JSON, otherwise keep as string
                try:
                    config_dict[config_key] = json.loads(value)
                except json.JSONDecodeError:
                    config_dict[config_key] = value
        
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)
        default : Any
            Default value if key not found
            
        Returns
        -------
        value : Any
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)
        value : Any
            Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        config[keys[-1]] = value
    
    def update(self, other: Union['Config', Dict[str, Any]]) -> None:
        """
        Update configuration with another configuration.
        
        Parameters
        ----------
        other : Config or Dict[str, Any]
            Configuration to merge
        """
        if isinstance(other, Config):
            other_dict = other._config
        else:
            other_dict = other
        
        self._config = self._deep_merge(self._config, other_dict)
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save(self, file_path: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        file_path : str or Path
            Output file path
        format : str
            Output format ('yaml' or 'json')
        """
        file_path = Path(file_path)
        
        if format == 'yaml':
            with open(file_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
        elif format == 'json':
            with open(file_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def keys(self) -> list:
        """Return top-level configuration keys."""
        return list(self._config.keys())
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using bracket notation."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"


def load_default_config() -> Config:
    """Load default configuration."""
    # Look for default config file
    config_paths = [
        Path.cwd() / "config" / "default.yaml",
        Path.cwd() / "default.yaml",
        Path(__file__).parent.parent.parent / "config" / "default.yaml"
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            return Config.from_yaml(config_path)
    
    # Return empty config if no default found
    return Config()


def merge_configs(*configs: Union[Config, Dict[str, Any]]) -> Config:
    """
    Merge multiple configurations.
    
    Parameters
    ----------
    *configs : Config or Dict[str, Any]
        Configurations to merge
        
    Returns
    -------
    merged : Config
        Merged configuration
    """
    merged = Config()
    
    for config in configs:
        merged.update(config)
    
    return merged


def validate_config(config: Config, schema: Dict[str, Any]) -> tuple:
    """
    Validate configuration against schema.
    
    Parameters
    ----------
    config : Config
        Configuration to validate
    schema : Dict[str, Any]
        Validation schema
        
    Returns
    -------
    is_valid : bool
        Whether configuration is valid
    errors : List[str]
        Validation errors
    """
    errors = []
    
    def validate_recursive(config_dict: Dict[str, Any], schema_dict: Dict[str, Any], path: str = ""):
        for key, schema_value in schema_dict.items():
            full_key = f"{path}.{key}" if path else key
            
            if key not in config_dict:
                if isinstance(schema_value, dict) and schema_value.get('required', False):
                    errors.append(f"Required key '{full_key}' is missing")
                continue
            
            config_value = config_dict[key]
            
            if isinstance(schema_value, dict):
                if 'type' in schema_value:
                    expected_type = schema_value['type']
                    if not isinstance(config_value, expected_type):
                        errors.append(f"Key '{full_key}' should be of type {expected_type.__name__}")
                
                if 'values' in schema_value:
                    allowed_values = schema_value['values']
                    if config_value not in allowed_values:
                        errors.append(f"Key '{full_key}' must be one of {allowed_values}")
                
                if 'range' in schema_value:
                    min_val, max_val = schema_value['range']
                    if not (min_val <= config_value <= max_val):
                        errors.append(f"Key '{full_key}' must be in range [{min_val}, {max_val}]")
                
                if 'schema' in schema_value:
                    validate_recursive(config_value, schema_value['schema'], full_key)
            
            elif isinstance(schema_value, type):
                if not isinstance(config_value, schema_value):
                    errors.append(f"Key '{full_key}' should be of type {schema_value.__name__}")
    
    validate_recursive(config.to_dict(), schema)
    
    return len(errors) == 0, errors


# Example schema for Mneme configuration
MNEME_CONFIG_SCHEMA = {
    'project': {
        'type': dict,
        'schema': {
            'name': {'type': str, 'required': True},
            'version': {'type': str, 'required': True},
            'random_seed': {'type': int, 'range': (0, 2**32 - 1)}
        }
    },
    'data': {
        'type': dict,
        'schema': {
            'raw_path': {'type': str, 'required': True},
            'processed_path': {'type': str, 'required': True},
            'cache_enabled': {'type': bool}
        }
    },
    'preprocessing': {
        'type': dict,
        'schema': {
            'denoise': {
                'type': dict,
                'schema': {
                    'enabled': {'type': bool},
                    'method': {'type': str, 'values': ['gaussian', 'median', 'wavelet']}
                }
            }
        }
    },
    'reconstruction': {
        'type': dict,
        'schema': {
            'method': {'type': str, 'values': ['ift', 'gaussian_process', 'neural_field']},
            'resolution': {'type': list}
        }
    }
}