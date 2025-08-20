from pathlib import Path
import yaml

# config.yaml is two levels up (project root)
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

def load_config():
    """Load the YAML config file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
    
    

def get(path, default=None):
    config = load_config()  # Load each time
    keys = path.split(".")
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
