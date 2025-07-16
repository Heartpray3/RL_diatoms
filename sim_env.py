import random
import shutil
import sys
from dataclasses import dataclass
import subprocess
from os import truncate
from pathlib import Path
import os
import math
from typing import List
import gymnasium
from gymnasium import spaces

from utils import get_sim_folder, RewardMethod, quaternion_rotation_matrix
import numpy as np


@dataclass(frozen=True)
class ColonyState:
    gaps: tuple[int, ...]

@dataclass(frozen=True)
class Action:
    n_rod: int
    direction: int

class DiatomEnv(gymnasium.Env):
    fact_blobs = 0.81
    def __init__(self,
                 input_file_path: str,
                 output_dir: str,
                 n_rods: int,
                 n_blobs: int,
                 reward_method: RewardMethod,
                 nb_steps_per_ep,
                 a: float,
                 dt: float,
                 angle = None):
        self.input_parm = input_file_path
        self.output_param = output_dir
        self.reward_method = reward_method
        self.angle = angle
        self.input_file_sim_path = ''
        self.sim_dir = ''
        self.const_filename = ''
        self.n_rods = n_rods
        self.n_blobs = n_blobs
        self.a = a
        self.dt = dt
        self.state = ColonyState(tuple())
        self._step = 0
        self.update_file = ''
        self.setup(input_file_path, output_dir, delete_folder=True)
        self.initial_cm = None
        self.update_file = ''
        self.episode = 0
        self.nb_steps_per_ep = nb_steps_per_ep
        self.reset()
        self.action_space = spaces.Discrete(self.n_rods * 2)
        self.observation_space = spaces.Box(
            low=-(self.n_blobs - 1),
            high=+(self.n_blobs - 1),
            shape=(self.n_rods - 1,),
            dtype=np.int32
        )

    def get_available_actions(self, state: ColonyState):
        available_actions: List[Action] = []

        for n_rod in range(self.n_rods):
            for direction in [-1, 1]:  # -1: right, 1: left
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

    def encode_action(self, action: Action) -> int:
        dir_index = 0 if action.direction == -1 else 1
        return action.n_rod * 2 + dir_index

    def decode_action(self, action_index: int) -> Action:
        direction = [-1, 1][action_index % 2]
        n_rod = action_index // 2
        return Action(n_rod, direction)

    def _state_to_obs(self, state: ColonyState) -> np.ndarray:
        return np.array(state.gaps, dtype=np.int32)

    def step(self, action_index: int):
        """
            Exécute une étape avec une action donnée.
            Retourne : next_state, reward, done
        """
        # Convertir l'action entière en Action(n_rod, direction)
        action = self.decode_action(action_index)

        # Vérifier si l'action est valide à cet état
        valid_actions = self.get_available_actions(self.state)
        if action not in valid_actions:
            # Action invalide : grosse pénalité + épisode terminé
            reward = -100.0
            done = True
            obs = self._state_to_obs(self.state)
            return obs, reward, done, False, {"invalid_action": True}

        rod_number = action.n_rod
        direction = action.direction

        self.write_move_const(
            moving_rod=rod_number,
            direction=direction,
        )

        # Étape 3 : Lancer la simulation
        sim_script_path = (Path(__file__).parent / "multi_bodies_bacillaria1.py").resolve()
        subprocess.run(
            [
            "python3",
            str(sim_script_path),
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
        *_, last_positions = self.extract_last_infos(clone_file)
        n, pos_lines, new_positions = self.extract_last_infos(new_coords_file)
        try:
            with open(clone_file, 'w') as f2:
                f2.writelines([n] + pos_lines)
            with open(self.update_file, 'a') as f3:
                f3.writelines([n] + pos_lines)
        except (FileNotFoundError, ValueError) as e:
            new_positions = last_positions
            print(f"Attention, didn't write or append in step {self._step}: \n\t{e}")

        # Étape 6 : Mettre à jour l'état
        new_gaps = list(self.state.gaps)
        for offset, sign in zip([-1, 0], [-1, 1]):
            idx = rod_number + offset
            if idx < 0 or idx >= len(new_gaps):
                continue
            new_gaps[idx] += direction * sign
        self.state = ColonyState(tuple(new_gaps))
        reward = self.compute_reward(last_positions, new_positions)
        self._step += 1
        if self._step >= self.nb_steps_per_ep:
            done = True
        else:
            done = False
        # on n’utilise pas de truncated spécifique, donc :
        truncated = False
        return self._state_to_obs(self.state), reward, done, truncated, {}

    def compute_reward(self, pos_before, pos_after):
        cm1 = self.compute_center_of_mass(pos_before)
        cm2 = self.compute_center_of_mass(pos_after)
        delta_cm = np.array(cm2) - np.array(cm1)

        match self.reward_method:
            case RewardMethod.VELOCITY:
                vel = [(c2 - c1) / self.dt for c1, c2 in zip(cm1, cm2)]
                return float(np.linalg.norm(vel))

            case RewardMethod.CM_DISTANCE:
                return float(np.linalg.norm(np.array(cm2) - np.array(cm1)))

            case RewardMethod.X_DISPLACEMENT:
                return float(cm2[0] - cm1[0])

            case RewardMethod.CIRCULAR_ZONES:
                if self.initial_cm is None:
                    self.initial_cm = cm1
                dist = float(np.linalg.norm(np.array(cm2) - np.array(self.initial_cm)))
                return int(dist // 10) - 1  # palier de 10

            case RewardMethod.FORWARD_PROGRESS:
                if self.angle is None:
                    raise ValueError("ANGLE IS REQUIRED FOR THIS METHOD")

                direction = np.array([math.cos(self.angle), 0, math.sin(self.angle)])
                direction /= np.linalg.norm(direction)

                reward = np.dot(delta_cm, direction)
                return float(reward)

            case _:
                raise ValueError("Méthode de reward inconnue")

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
        expression = self.fact_blobs * self.a / (2 * self.dt) * direction
        sixzeros = ['0'] * 6
        constraints = {}
        last_pos = list(map(lambda x: self.fact_blobs * x * self.a / 2, self.state.gaps))

        for n in range(self.n_rods - 1):
            i, j = n, n + 1

            # Par défaut, tout est fixe sauf X
            pos1y, pos1z = "0", f"{self.a}"
            pos2y, pos2z = "0", f"{-self.a}"
            vel1y = vel1z = vel2y = vel2z = "0"

            # Init X
            pos1x, pos2x = f"{last_pos[n]}", f"{-last_pos[n]}"
            vel1x = vel2x = "0"


            # TRUST ME THIS WHOLE BLOCK MAYBE CONFUSING BUT IS CORRECT ABSOLUTLY CORRECT DO NOT TOUCH IT IF YOU DON'T
            # WANT TO DEBUG FOR 3 HOURS.
            if n == moving_rod:
                pos1x = f"{expression}*t" + f"+{last_pos[n]}"
                pos2x = f"{-expression}*t" + f"+{-last_pos[n]}"
                vel1x = f"{expression}"
                vel2x = f"{-expression}"
                # self.const_pos[n] = pos1x
            elif n + 1 == moving_rod:
                pos1x = f"{-expression}*t" + f"+{last_pos[n]}"
                pos2x = f"{expression}*t" + f"+{-last_pos[n]}"
                vel1x = f"{-expression}"
                vel2x = f"{expression}"
                # self.const_pos[n] = pos1x
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
        with open(const_path, "w") as f:
            f.write(f"{data_out['n_rods']}\n")
            f.write(f"{data_out['n_constraints']}\n")

            for key, constraint in data_out["constraints"].items():
                i, j = key[0], key[1]
                raw_expr = constraint["raw_expr"]

                if len(raw_expr) != 18:
                    raise ValueError(f"raw_expr for constraint {key} does not have 20 elements")

                line = f"{i} {j} " + " ".join(raw_expr)
                f.write(line + "\n")
        return const_path

    # def _safe_eval(self, expr):
    #     """Convert numerical string or keep expressions like '1.1*(0)' or '3*t'."""
    #     try:
    #         # Try evaluating basic constants
    #         return eval(expr, {"__builtins__": None}, {"t": self.dt})  # dummy eval to allow 't'
    #     except:
    #         return expr

    # def write_constraints_file(self, data, path):
    #     with open(path, "w") as f:
    #         f.write(f"{data['n_rods']}\n")
    #         f.write(f"{data['n_constraints']}\n")
    #
    #         for key, constraint in data["constraints"].items():
    #             i, j = key[0], key[1]
    #             raw_expr = constraint["raw_expr"]
    #
    #             if len(raw_expr) != 18:
    #                 raise ValueError(f"raw_expr for constraint {key} does not have 20 elements")
    #
    #             line = f"{i} {j} " + " ".join(raw_expr)
    #             f.write(line + "\n")

    def reset(self, *, seed = None, options = None):
            if seed:
                random.seed(seed)
            self._step = 0
            self.state = ColonyState(tuple(random.randint(-(self.n_blobs - 1), self.n_blobs - 1) for _ in range(self.n_rods - 1)))#ColonyState((0,) * (self.n_rods - 1))
            self.update_file = f'epoch_{self.episode}_update_'
            self.setup(self.input_parm, self.output_param, delete_folder=False)
            self.initial_cm = None
            self.episode += 1
            return self._state_to_obs(self.state)

    def setup(self, input_file_path, output_dir, delete_folder=True):
        X_coef = 7.4209799e-02
        X_step = 2 * X_coef

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

        to_save = self.give_abs_state()

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

    def give_abs_state(self, angle=None):
        n_rods = len(self.state.gaps) + 1
        pos = np.zeros((n_rods, 3))
        quat = np.zeros((n_rods, 4))
        quat[0, 0] = 1
        # pos[0, 0] = 0.0
        for n, gap in enumerate(self.state.gaps):
            pos[n + 1, 2] = (n + 1) * 2 * self.a
            pos[n + 1, 0] = gap * self.a * self.fact_blobs  + pos[n, 0]
            quat[n + 1, 0] = 1
        mean_pos = np.mean(pos, axis=0)

        for n in range(n_rods):
            pos[n, :] = pos[n, :] - mean_pos

        to_save = np.concatenate((pos, quat), axis=1)

        return to_save

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
            gap = int(round(gap_proj / (DiatomEnv.fact_blobs * a)))
            gaps.append(gap)
        return ColonyState(tuple(gaps))
