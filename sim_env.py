import shutil
import sys
from dataclasses import dataclass
import subprocess
from pathlib import Path
import os
import math
from typing import List

from utils import get_sim_folder
import numpy as np
from itertools import combinations

@dataclass(frozen=True)
class ColonyState:
    gaps: tuple[int, ...]

@dataclass(frozen=True)
class Action:
    n_rod: int
    direction: int

class DiatomEnv:
    def __init__(self,
                 input_file_path: str,
                 output_dir: str,
                 n_rods: int,
                 n_blobs: int,
                 a: float,
                 dt: float):
        self.input_parm = input_file_path
        self.output_param = output_dir
        self.input_file_sim_path = ''
        self.sim_dir = ''
        self.const_filename = ''
        self.n_rods = n_rods
        self.n_blobs = n_blobs
        self.a = a
        self.dt = dt
        self.state = None
        self._step = 0
        self.update_file = ''
        self.setup(input_file_path, output_dir, delete_folder=True)
        self.update_file = ''
        self.const_pos = []
        self.reset(0)

    def get_available_actions(self, state: ColonyState):
        available_actions: List[Action] = []

        for n_rod in range(self.n_rods):
            for direction in [-1, 1]:  # -1: contract, 1: expand
                new_gaps = list(state.gaps)

                for offset, sign in zip([-1, 0], [-1, 1]):
                    idx = n_rod + offset
                    if idx < 0 or idx >= len(new_gaps):
                        continue
                    new_gaps[idx] += direction * sign

                # Vérifier si tous les gaps sont dans la limite autorisée
                if all(abs(gap) <= self.n_blobs - 1 for gap in new_gaps):
                    available_actions.append(Action(n_rod, direction))
        return available_actions

    def step(self, action: Action):
        """
                Exécute une étape avec une action donnée.
                Retourne : next_state, reward, done
                """
        rod_number = action.n_rod
        direction = action.direction

        self.write_move_const(
            moving_rod=rod_number,
            direction=direction,
        )

        # Étape 3 : Lancer la simulation
        subprocess.run(
            [
            "python3",
            "../../../multi_bodies_bacillaria1.py",
            "--input-file",
            self.input_file_sim_path,
            # "--print-residual",
            ],
            cwd=self.sim_dir,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Étape 4 : Lire la sortie
        get_sim_file = lambda prefix, suffix: os.path.join(
            str(self.sim_dir),
            f"{prefix}bacillaria_{self.n_blobs}_blobs_{self.n_rods}_rods{suffix}")
        new_coords_file = get_sim_file("run.", ".config")
        clone_file = get_sim_file("", ".clones")
        *_, positions = self.extract_last_infos(clone_file)
        last_cm = self.compute_center_of_mass(positions)

        try:
            n, pos_lines, positions = self.extract_last_infos(new_coords_file)
            with open(clone_file, 'w') as f2:
                f2.writelines([n] + pos_lines)
            with open(self.update_file, 'a') as f3:
                f3.writelines([n] + pos_lines)
            new_cm = self.compute_center_of_mass(positions)
            inst_vel = [(c1 - c2)/self.dt for c1, c2 in zip(new_cm, last_cm)]
        except (FileNotFoundError, ValueError) as e:
            inst_vel = [0] * 3
            print(f"Attention, 0 velocity: \n\t{e}")

        # Étape 6 : Mettre à jour l'état
        new_gaps = list(self.state.gaps)
        for offset, sign in zip([-1, 0], [-1, 1]):
            idx = rod_number + offset
            if idx < 0 or idx >= len(new_gaps):
                continue
            new_gaps[idx] += direction * sign
        self.state = ColonyState(tuple(new_gaps))

        self._step += 1
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
                if lines[i].removesuffix('\n').isdigit():
                    n = int(lines[i])
                    pos_lines = lines[i + 1:i + 1 + n]
                    n = lines[i]
                    break
            else:
                raise ValueError(f"No positions found in this file {file_path}")
        positions = [list(map(float, line.split()[:3])) for line in pos_lines]
        return n, pos_lines, positions

    def write_move_const(
            self,
            moving_rod: int,
            direction: int,
            Nconst_per_rod: int = 2,
            offset: float = 0.1
    ) -> str:
        """
        Écrit un fichier .const basé sur un déplacement linéaire de ±2a
        pour le bâtonnet `moving_rod`, pendant dt. Respecte le format de contraintes complet.
        """

        const_path = os.path.join(self.sim_dir, self.const_filename)
        expression = self.a / (2 * self.dt) * direction
        sixzeros = ['0'] * 6
        constraints = {}
        last_pos = list(map(self._safe_eval, self.const_pos))

        for n in range(self.n_rods - 1):
            i, j = n, n + 1

            # Par défaut, tout est fixe sauf X
            pos1y, pos1z = "0", f"{self.a}"
            pos2y, pos2z = "0", f"{-self.a}"
            vel1y = vel1z = vel2y = vel2z = "0"

            # Init X
            pos1x, pos2x = f"{last_pos[n]}", f"{-last_pos[n]}"
            vel1x = vel2x = "0"


            # Si ce bâton est le mobile
            if n == moving_rod:
                pos1x = f"{expression}*t" + f"+{last_pos[n]}"
                pos2x = f"{-expression}*t" + f"+{-last_pos[n]}"
                vel1x = f"{expression}"
                vel2x = f"{-expression}"
                self.const_pos[n] = pos1x
                # self.const_pos[n + self.n_rods - 1] = pos2x
            elif n + 1 == moving_rod:
                pos1x = f"{-expression}*t" + f"+{last_pos[n]}"
                pos2x = f"{expression}*t" + f"+{-last_pos[n]}"
                vel1x = f"{-expression}"
                vel2x = f"{expression}"
                self.const_pos[n] = pos1x
                # self.const_pos[n + self.n_rods - 1] = pos2x
            # Contrainte 1 (centrale)
            raw_expr_1 = sixzeros + [
                pos1x, pos1y, pos1z,
                pos2x, pos2y, pos2z,
                vel1x, vel1y, vel1z,
                vel2x, vel2y, vel2z
            ]
            constraints[(i, j)] = {"raw_expr": raw_expr_1}

            # Contrainte 2 (offset)
            if Nconst_per_rod >= 2:
                pos1_off = [f"{1 + offset}*({pos1x})", pos1y, f"{(1 + offset)* self.a}"]
                pos2_off = [f"{1 - offset}*({pos2x})", pos2y, f"{(1 - offset)* -self.a}"]
                vel1_off = [f"{1 + offset}*({vel1x})", vel1y, vel1z]
                vel2_off = [f"{1 - offset}*({vel2x})", vel2y, vel2z]

                raw_expr_2 = sixzeros + pos1_off + pos2_off + vel1_off + vel2_off
                constraints[(i, j, "offset")] = {"raw_expr": raw_expr_2}

        # Structure finale
        data_out = {
            "n_rods": self.n_rods,
            "n_constraints": len(constraints),
            "constraints": constraints
        }

        # Écriture via fonction dédiée
        self.write_constraints_file(data_out, const_path)
        return const_path

    def _safe_eval(self, expr):
        """Convert numerical string or keep expressions like '1.1*(0)' or '3*t'."""
        try:
            # Try evaluating basic constants
            return eval(expr, {"__builtins__": None}, {"t": self.dt})  # dummy eval to allow 't'
        except:
            return expr

    def write_constraints_file(self, data, path):
        with open(path, "w") as f:
            f.write(f"{data['n_rods']}\n")
            f.write(f"{data['n_constraints']}\n")

            for key, constraint in data["constraints"].items():
                i, j = key[0], key[1]
                raw_expr = constraint["raw_expr"]

                if len(raw_expr) != 18:
                    raise ValueError(f"raw_expr for constraint {key} does not have 20 elements")

                line = f"{i} {j} " + " ".join(raw_expr)
                f.write(line + "\n")

    # def parse_constraints_file(self, filepath):
    #     with open(filepath, "r") as f:
    #         lines = f.readlines()
    #
    #     n_rods = int(lines[0])
    #     n_constraints = int(lines[1])
    #     constraint_lines = lines[2:]
    #
    #     constraints = {}
    #
    #     for line in constraint_lines:
    #         parts = line.strip().split()
    #         if len(parts) != 20:
    #             raise ValueError(f"Line does not have 20 elements: {line}")
    #
    #         i, j = int(parts[0]), int(parts[1])
    #         data = {
    #             "params": list(map(self._safe_eval, parts[2:])),
    #             "raw_expr": parts[2:]  # for preserving expressions if needed
    #         }
    #         if constraints.get((i, j)):
    #             constraints[(i, j, "off")] = data
    #         else:
    #             constraints[(i, j)] = data
    #
    #     return {
    #         "n_rods": n_rods,
    #         "n_constraints": n_constraints,
    #         "constraints": constraints
    #     }

    def reset(self, episode_nb: int) -> ColonyState:
            self._step = 0
            self.state = ColonyState((0,) * (self.n_rods - 1))
            self.update_file = f'step_{episode_nb}_update_'
            self.setup(self.input_parm, self.output_param, delete_folder=False)
            self.const_pos = []
            for n in range(self.n_rods - 1):
                self.const_pos.append('0')
            return self.state

    def setup(self, input_file_path, output_dir, delete_folder=True):
        X_coef = 7.4209799e-02
        X_step = 2 * X_coef

        Nconst_per_rod = 2
        root_name = 'bacillaria_'

        suffix_nrods = str(self.n_blobs) + '_blobs_'
        filename1 = root_name + suffix_nrods

        suffix_nrods = str(self.n_rods) + '_rods'
        filename = filename1 + suffix_nrods

        self.sim_dir = get_sim_folder(output_dir, self.n_rods, self.n_blobs)
        if os.path.exists(self.sim_dir):
            if delete_folder:
                print("Folder {} already exists!".format(self.sim_dir))
                response = input("Do you want to restart the simulation ? (y/n) : ").strip().lower()
                if response == 'y':
                    shutil.rmtree(self.sim_dir)
                    print("Simulation restarted.")
                    os.makedirs(self.sim_dir)
                else:
                    print("Abort simulation.")
                    return ()
        else:
            os.makedirs(self.sim_dir)

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
        self.update_file = os.path.join(self.sim_dir, self.update_file + filename + '.config')
        fid = open(self.update_file, 'w')
        fid.write(str(self.n_rods) + '\n')
        np.savetxt(fid, to_save)
        fid.close()

        with open(os.path.join(output_dir, f"{self.n_blobs}_Blobs", root_name + str(self.n_blobs) + '_blobs.vertex'), "w") as f:
            f.write(str(self.n_blobs) + "\n")
            Start = -((self.n_blobs / 2 - 1) * X_step + X_coef)
            for i in range(self.n_blobs):
                X = Start + i * X_step
                f.write("{:.7e} {:.7e} {:.7e}\n".format(X, 0, 0))

        filename_vertex = root_name + str(self.n_blobs) + '_blobs.vertex'
        filename_list_vertex = filename + '.list_vertex'
        fid = open(os.path.join(self.sim_dir, filename_list_vertex), 'w')

        for n in range(self.n_rods):
            fid.write(os.path.join(output_dir, f"{self.n_blobs}_Blobs", filename_vertex) + '\n')
        fid.close()

        self.const_filename = filename + '.const'
        filename_input = 'inputfile_bacillaria'

        updates = {
            'dt': [str(self.dt)],
            'n_steps': [str(1)],
            'output_name': ['run'],
            'articulated': [filename_list_vertex, filename_clones, self.const_filename]
        }

        self.input_file_sim_path = os.path.join(self.sim_dir, f"{filename_input}_{suffix_nrods}.dat")
        self.update_dat(input_file_path, self.input_file_sim_path, updates)

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

    def physical_colony_state(self) -> ColonyState:
        get_sim_file = lambda prefix, suffix: os.path.join(
            str(self.sim_dir),
            f"{prefix}bacillaria_{self.n_blobs}_blobs_{self.n_rods}_rods{suffix}")
        clone_file = get_sim_file("", ".clones")
        with open(clone_file, 'r') as f:
            lines = f.readlines()

        for i in range(len(lines)):
            if lines[i].strip().isdigit():
                n = int(lines[i])
                data = lines[i + 1:i + 1 + n]
                break
        else:
            raise ValueError("Pas de section positions trouvée dans le fichier")

        array = np.array([[float(x) for x in line.strip().split()] for line in data])
        pos = array[:, :3]
        quats = array[:, 3:]

        return self.infer_colony_state_from_positions(pos, quats, self.a)

    @staticmethod
    def infer_colony_state_from_positions(pos: np.ndarray, quats: np.ndarray, a: float) -> ColonyState:
        n_rods = len(pos)
        gaps = []

        for i in range(n_rods - 1):
            p1, p2 = pos[i], pos[i + 1]
            delta = p2 - p1

            axis_1 = quaternion_rotation_matrix(*quats[i])[:, 0]
            axis_1 /= np.linalg.norm(axis_1)

            gap_proj = np.dot(delta, axis_1)
            gap = int(round(gap_proj / a))
            gaps.append(gap)

        return ColonyState(tuple(gaps))

def quaternion_rotation_matrix(q0, q1, q2, q3):
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])
