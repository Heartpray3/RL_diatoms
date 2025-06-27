import yaml
from dataclasses import dataclass
from pathlib import Path
import os

from enum import Enum, auto

class RewardMethod(Enum):
    VELOCITY = "VELOCITY"
    CM_DISTANCE = "CM_DISTANCE"
    X_DISPLACEMENT = "X_DISPLACEMENT"
    CIRCULAR_ZONES = "CIRCULAR_ZONES"
    FORWARD_PROGRESS = "FORWARD_PROGRESS"

@dataclass
class Config:
    input_file_path: str
    output_directory: str
    nb_blobs: int
    nb_rods: int
    dt: float
    nb_episodes: int
    nb_step: int
    learning_rate: float
    discount_factor: float
    lookahead_steps: int
    reward_method: RewardMethod
    reward_angle: float


def load_config(path="config.yaml") -> Config:
    with open(path, 'r') as file:
        raw = yaml.safe_load(file)

    raw['input_file_path'] = abs_path(raw['input_file_path'])
    raw['output_directory'] = abs_path(raw['output_directory'])
    raw['reward_method'] = RewardMethod[raw['reward_method'].upper()]

    return Config(**raw)

def abs_path(directory: str) -> str:
    script_dir = Path(__file__).parent.resolve()
    abs_dir = (script_dir / directory).resolve()
    return str(abs_dir)

def get_sim_folder(output_dir: str, n_rods: int, n_blobs: int):
    root_name = 'bacillaria_'
    rod_folder = f'{n_rods}_Rods'
    blob_folder = f'{n_blobs}_Blobs'
    parent_folder = os.path.join(blob_folder, rod_folder)
    output_folder = os.path.join(parent_folder, root_name + f"{n_blobs}_blobs_{n_rods}_rods")
    return os.path.join(output_dir, output_folder)