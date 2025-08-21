from pathlib import Path
import yaml

# config.yaml is two levels up (project root)
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

def load_config():
    """Load the YAML config file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_value(path_or_list, default=None):
    """
    Retrieve value(s) from the config.
    
    Args:
        path_or_list (str or list[str]): Dot-separated key path(s)
        default: Value to return if key is missing

    Returns:
        value or list of values
    """
    config = load_config()

    def _get_single(path):
        keys = path.split(".")
        value = config
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    if isinstance(path_or_list, str):
        return _get_single(path_or_list)
    elif isinstance(path_or_list, (list, tuple)):
        return [_get_single(p) for p in path_or_list]
    else:
        raise TypeError("path_or_list must be a str or list/tuple of str")
        
