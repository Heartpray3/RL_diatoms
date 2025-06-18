import shutil
from dataclasses import dataclass
import subprocess
from pathlib import Path
import os
import math
from typing import List
from utils import get_sim_folder
import numpy as np


@dataclass(frozen=True)
class ColonyState:
    gaps: tuple[int, ...]

@dataclass(frozen=True)
class Action:
    n_gap: int
    direction: int

class DiatomEnv:
    def __init__(self,
                 input_file_path: str,
                 output_dir: str,
                 n_rods: int,
                 n_blobs: int,
                 a: float,
                 dt: float):
        self.input_file_sim_path = ''
        self.sim_dir = ''
        self.const_filename = ''
        self.n_rods = n_rods
        self.n_blobs = n_blobs
        self.a = a
        self.dt = dt
        self.state = None
        self.reset()
        self.setup(input_file_path, output_dir)

    def get_available_actions(self, state: ColonyState):
        available_actions: List[Action] = []
        for n_gap, gap in enumerate(state.gaps):
            if abs(gap) >= self.n_blobs - 1:
                available_actions.append(Action(n_gap, -int(math.copysign(1, gap))))
                continue
            available_actions.append(Action(n_gap, +1))
            available_actions.append(Action(n_gap, -1))
        return available_actions


    def step(self, action: Action):
        """
                Exécute une étape avec une action donnée.
                Retourne : next_state, reward, done
                """
        gap_number = action.n_gap
        direction = action.direction

        self.write_move_const(
            moving_rod=gap_number,
            direction=direction
        )

        # Étape 3 : Lancer la simulation
        subprocess.run(
            ["python3",
            "multi_bodies_bacillaria1.py",
            f" --input-file {self.input_file_sim_path}",
            # "--print-residual",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Étape 4 : Lire la sortie
        get_sim_file = lambda prefix, suffix: os.path.join(str(Path(self.input_file_sim_path).parent),
                                       f"{prefix}{Path(self.input_file_sim_path).stem}{suffix}")
        new_coords_file = get_sim_file("run.", ".config")
        clone_file = get_sim_file("", ".clones")
        *_, positions = self.extract_last_infos(clone_file)
        last_cm = self.compute_center_of_mass(positions)
        try:
            n, pos_lines, positions = self.extract_last_infos(new_coords_file)
            with open(clone_file, 'w') as f2:
                f2.writelines([str(n)] + [pos_lines])
            new_cm = self.compute_center_of_mass(positions)
            inst_vel = [(c1 - c2)/self.dt for c1, c2 in zip(new_cm, last_cm)]
        except (FileNotFoundError, ValueError) as e:
            inst_vel = [0] * 3
            print(f"Attention, 0 velocity: \n\t{e}")

        # Étape 6 : Mettre à jour l'état
        new_gaps = list(self.state.gaps)
        new_gaps[gap_number] += direction
        self.state = ColonyState(tuple(new_gaps))

        return self.state, inst_vel

    @staticmethod
    def compute_center_of_mass(positions: List[List[float]]) -> List[float]:
        n = len(positions)
        return [sum(coord) / n for coord in zip(*positions)]

    @staticmethod
    def extract_last_infos(file_path):
        with open(file_path, 'r') as f:
            lines = list(f)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].isdigit():
                    n = int(lines[i])
                    pos_lines = lines[i + 1:i + 1 + n]
                    break
            else:
                raise ValueError(f"No positions found in this file {file_path}")
        positions = [list(map(float, line.split()[:3])) for line in pos_lines]
        return n, pos_lines, positions

    def write_move_const(
            self,
            moving_rod: int,  # indice [0‥Nrods-1]
            direction: int,  # +1 ou -1
            Nconst_per_rod: int = 2,
            offset: float = 0.1
    ) -> str:
        """
        Écrit un fichier .const basé sur un déplacement linéaire de ±2a
        pour le bâtonnet `moving_rod`, pendant dt. Respecte le format de contraintes complet.
        """

        const_path = os.path.join(self.sim_dir, self.const_filename)
        with open(const_path, "w") as fid:
            fid.write(f"{self.n_rods}\n")
            fid.write(f"{(self.n_rods - 1) * Nconst_per_rod}\n")

            sixzeros = '0 0 0 0 0 0'

            for n in range(self.n_rods - 1):
                pos1z, pos2z = ' 0 ', ' 0 '
                pos1y, pos2y = f"{self.a} ", f"{-self.a} "
                vel1z = vel2z = vel1y = vel2y = ' 0 '

                pos1x = pos2x = '0'
                vel1x = vel2x = '0'
                expression = self.a / (2 * self.dt) * direction
                if n == moving_rod:
                    pos1x = f"{expression}*t"
                    pos2x = f"{-expression}*t"
                    vel1x = f"{expression}"
                    vel2x = f"{-expression}"
                elif n + 1 == moving_rod:
                    pos1x = f"{-expression}*t"
                    pos2x = f"{expression}*t"
                    vel1x = f"{-expression}"
                    vel2x = f"{expression}"

                c1 = (
                    f"{n} {n + 1} {sixzeros} "
                    f"{pos1x} {pos1z}{pos1y}"
                    f"{pos2x} {pos2z}{pos2y}"
                    f"{vel1x} {vel1z}{vel1y}"
                    f"{vel2x} {vel2z}{vel2y}"
                )
                fid.write(c1 + '\n')

                if Nconst_per_rod >= 2:
                    c2 = (
                        f"{n} {n + 1} {sixzeros} "
                        f"{1 + offset}*({pos1x}) {pos1z}{(1 + offset) * self.a} "
                        f"{1 - offset}*({pos2x}) {pos2z}{(1 - offset) * -self.a} "
                        f"{1 + offset}*({vel1x}) {vel1z}{vel1y} "
                        f"{1 - offset}*({vel2x}) {vel2z}{vel2y}"
                    )
                    fid.write(c2 + '\n')

        return const_path

    def reset(self) -> ColonyState:
        self.state = ColonyState((0,)*(self.n_rods - 1))

        return self.state

    def setup(self, input_file_path, output_dir):
        # %% Code Generate_const_clone_list_vertex_file_and_execute_Blobs
        X_coef = 7.4209799e-02
        X_step = 2 * X_coef

        Nconst_per_rod = 2
        root_name = 'bacillaria_'

        suffix_nrods = str(self.n_blobs) + '_blobs_'
        filename1 = root_name + suffix_nrods
        # L_rods = self.n_blobs * 0.81 * a
        # L = 0.5 * 0.9 * L_rods

        suffix_nrods = str(self.n_blobs) + '_rods'
        filename = filename1 + suffix_nrods

        foldername = filename  # + suffix_phase_shift #+ suffix_fourier

        self.sim_dir = get_sim_folder(output_dir, self.n_rods, self.n_blobs)
        if os.path.exists(self.sim_dir):
            print("Folder {} already exists!".format(self.sim_dir))
            response = input("Do you want to restart the simulation ? (y/n) : ").strip().lower()
            if response == 'y':
                shutil.rmtree(self.sim_dir)
                print("Simulation restarted.")
            else:
                print("Abort simulation.")
                return ()

        os.makedirs(self.sim_dir)

        ## Generate clones files
        filename_clones = filename + '.clones'

        pos = np.zeros((self.n_rods, 3))
        quat = np.zeros((self.n_rods, 4))

        for n in range(self.n_rods):
            pos[n, 2] = (n - 1) * 2 * self.a
            quat[n, 0] = 1
        mean_pos = np.mean(pos, axis=0)

        for n in range(self.n_rods):
            pos[n, :] = pos[n, :] - mean_pos

        to_save = np.concatenate((pos, quat), axis=1)

        fid = open(os.path.join(self.sim_dir, filename_clones), 'w')
        fid.write(str(self.n_rods) + '\n')
        np.savetxt(fid, to_save)
        fid.close()

        ## Generate Vertex files
        with open(os.path.join(output_dir, f"{self.n_blobs}_Blobs", root_name + str(self.n_blobs) + '_blobs.vertex'),
                  "w") as f:
            f.write(str(self.n_blobs) + "\n")
            Start = -((self.n_blobs / 2 - 1) * X_step + X_coef)
            for i in range(self.n_blobs):
                X = Start + i * X_step
                f.write("{:.7e} {:.7e} {:.7e}\n".format(X, 0, 0))

        ## Generate list vertex files
        filename_vertex = root_name + str(self.n_blobs) + '_blobs.vertex'

        filename_list_vertex = filename + '.list_vertex'
        fid = open(os.path.join(self.sim_dir, filename_list_vertex), 'w')

        for n in range(self.n_rods):
            fid.write(os.path.join(output_dir, f"{self.n_blobs}_Blobs", filename_vertex) + '\n')
        fid.close()

        ## Constraints
        self.const_filename = filename + '.const'

        filename_input = 'inputfile_bacillaria'
        # input_path = os.path.join(input_file_path, filename_input + '.dat')

        # Valeurs à mettre à jour
        updates = {
            'dt': [str(self.dt)],
            'n_steps': [str(1)],  # RL feedback
            'output_name': ['run'],
            'articulated': [filename_list_vertex, filename_clones, self.const_filename]
        }

        # Chemin de sortie
        self.input_file_sim_path = os.path.join(self.sim_dir, f"{filename_input}_{suffix_nrods}.dat")

        # Écriture du nouveau fichier .dat
        self.update_dat(input_file_path, self.input_file_sim_path, updates)
        # return self.input_file_sim

    @staticmethod
    def update_dat(input_file, output_file, updates):
        lines = open(input_file).readlines()
        max_len = max((len(l.split()[0]) for l in lines if l.strip() and not l.strip().startswith('#')), default=0)

        with open(output_file, 'w') as f:
            for line in lines:
                if not line.strip() or line.lstrip().startswith('#'):
                    f.write(line)
                    continue
                key, *vals = line.split()
                new_vals = updates.get(key, vals)
                f.write(f"{key.ljust(max_len + 1)}{' '.join(new_vals)}\n")