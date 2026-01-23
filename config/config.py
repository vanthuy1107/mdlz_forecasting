"""Configuration management module."""
import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import date
import torch


class Config:
    """Configuration class to manage hyperparameters and paths."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default.
        """
        if config_path is None:
            # Default to config/config.yaml relative to this file
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Set dynamic values
        self._update_device()
    
    def _update_device(self):
        """Set device based on config."""
        device_config = self._config['training']['device']
        if device_config == 'auto':
            self._config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config['data']
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._config['model']
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._config['training']
    
    @property
    def window(self) -> Dict[str, Any]:
        """Get window configuration."""
        return self._config['window']
    
    @property
    def output(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self._config['output']
    
    @property
    def inference(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self._config['inference']
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key (e.g., 'data.years')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot-separated key."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with dictionary of values."""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self._config, updates)
    
    def save(self, path: Optional[str] = None):
        """Save configuration to YAML file."""
        if path is None:
            path = self.config_path
        
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file.
    
    Returns:
        Config object.
    """
    return Config(config_path)


def load_holidays(holidays_path: Optional[str] = None, holiday_type: str = "model") -> Dict[int, Dict[str, List[date]]]:
    """
    Load Vietnam holidays from holidays.yaml config file.
    
    Args:
        holidays_path: Path to holidays.yaml file. If None, uses default.
        holiday_type: Type of holidays to load. Either "model" or "business".
    
    Returns:
        Dictionary mapping years to holiday types to lists of dates.
        Format: {year: {holiday_type: [date, ...], ...}, ...}
    """
    if holidays_path is None:
        # Default to config/holidays.yaml relative to this file
        holidays_path = Path(__file__).parent / "holidays.yaml"
    
    holidays_path = Path(holidays_path)
    if not holidays_path.exists():
        raise FileNotFoundError(f"Holidays config file not found: {holidays_path}")
    
    with open(holidays_path, 'r') as f:
        holidays_config = yaml.safe_load(f)
    
    # Allow user-friendly aliases ("model", "business") as well as
    # the raw keys from YAML ("model_holidays", "business_holidays").
    if holiday_type in ("model", "business"):
        yaml_key = f"{holiday_type}_holidays"
    else:
        yaml_key = holiday_type
    
    if yaml_key not in holidays_config:
        raise ValueError(
            f"Holiday type '{holiday_type}' not found in config. "
            f"Available keys: {list(holidays_config.keys())}"
        )
    
    holidays_data = holidays_config[yaml_key]
    
    # Convert YAML format [year, month, day] to date objects
    result = {}
    for year_str, year_holidays in holidays_data.items():
        year = int(year_str)
        result[year] = {}
        for holiday_name, date_list in year_holidays.items():
            result[year][holiday_name] = [
                date(d[0], d[1], d[2]) for d in date_list
            ]
    
    return result

