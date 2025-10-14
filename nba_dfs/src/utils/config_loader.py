import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and validate experiment configurations"""

    @staticmethod
    def load_experiment(config_path: str) -> Dict[str, Any]:
        """
        Load experiment configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dictionary containing experiment configuration

        Raises:
            FileNotFoundError: If config file does not exist
            yaml.YAMLError: If YAML parsing fails
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Validate experiment configuration structure.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If required keys are missing
        """
        required_keys = ['data', 'features', 'model', 'evaluation']
        missing_keys = [k for k in required_keys if k not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        return True
