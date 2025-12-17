"""Configuration parser."""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Merge two configuration dictionaries."""
    merged = base_config.copy()
    merged.update(override_config)
    return merged