import yaml
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class Config:
    input_directory: str
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
        config = Config(**yaml.safe_load(file))
        config.input_directory = abs_path(config.input_directory)
        config.output_directory = abs_path(config.output_directory)
    return config

def abs_path(directory: str) -> str:
    script_dir = Path(__file__).parent.resolve()
    abs_dir = (script_dir / directory).resolve()
    return str(abs_dir)

def get_sim_folder(sub_folder_name: str, n_rods: int, n_blobs: int):
    rod_folder = f'{n_rods}_Rods'
    blob_folder = f'{n_blobs}_Blobs'
    parent_folder = os.path.join(blob_folder, rod_folder)
    output_folder = os.path.join(parent_folder, sub_folder_name)
    return output_folder