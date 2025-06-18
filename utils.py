import yaml
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class Config:
    input_file_path: str
    output_directory: str
    Nblobs: int
    Nrods: int
    dt: float
    Nstep: int


def load_config(path="config.yaml") -> Config:
    with open(path, 'r') as file:
        config = Config(**yaml.safe_load(file))
        config.input_file_path = abs_path(config.input_file_path)
        config.output_directory = abs_path(config.output_directory)
    return config

def abs_path(directory: str) -> str:
    script_dir = Path(__file__).parent.resolve()
    abs_dir = (script_dir / directory).resolve()
    return str(abs_dir)

def get_sim_folder(output_dir: str, n_rods: int, n_blobs: int):
    root_name = 'bacillaria_'
    rod_folder = f'{n_rods}_Rods'
    blob_folder = f'{n_blobs}_Blobs'
    parent_folder = os.path.join(blob_folder, rod_folder)
    output_folder = os.path.join(parent_folder, root_name + f"{n_blobs}_blobs_{n_blobs}_rods")
    return os.path.join(output_dir, output_folder)