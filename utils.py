import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    output_directory: str
    Nblobs: int
    phase_shift: float
    Nrods: int
    an: str
    bn: str
    nmodes: int
    dt: float
    Nstep: int
    freq: int

def load_config(path="config.yaml") -> Config:
    with open(path, 'r') as file:
        return Config(**yaml.safe_load(file))

def abs_path(directory: str) -> Path:
    script_dir = Path(__file__).parent.resolve()
    abs_dir = (script_dir / directory).resolve()
    return abs_dir