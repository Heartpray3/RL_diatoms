# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt; plt.close('all')
from sim_env import DiatomEnv
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_config, get_sim_folder, quaternion_rotation_matrix

# === PARAMÈTRES ===
run_single_file = True         # ← active la PARTIE 1
run_multiple_episodes = False  # ← active la PARTIE 2
trace_every_n = 100           # pour la PARTIE 2

# === CONFIGURATION ===
config = load_config()
Nrods = config.nb_rods
dt = config.dt
a = 0.183228708092682
working_dir = config.output_directory
root_name = 'bacillaria_'
sim_dir = get_sim_folder(working_dir, config.nb_rods, config.nb_blobs)
sim_path = os.path.join(working_dir, sim_dir)

# === CHARGEMENT VERTEX ===
vertex_file = os.path.join(working_dir, f"{config.nb_blobs}_Blobs", f"{root_name}{config.nb_blobs}_blobs.vertex")
with open(vertex_file, 'r') as f:
    lines = f.read().split()
    nb_blobs = int(lines[0])
    vertex = np.array(lines[1:], dtype=float).reshape((-1, 3))

# === FONCTION DE LECTURE DES POSES ===
def load_cm_positions(filepath, Nrods):
    with open(filepath, 'r') as f:
        lines = [list(map(float, l.strip().split())) for l in f if len(l.strip().split()) == 7]
    data = np.array(lines)
    if len(data) % Nrods != 0:
        print(f"⚠️ {filepath} mal formé.")
        return None
    n_steps = len(data) // Nrods
    positions = data.reshape((n_steps, Nrods, 7))
    return positions

# === CALCUL POSITIONS BLOBS ===
def compute_all_blob_positions(rods_positions, vertex):
    n_steps, Nrods, _ = rods_positions.shape
    blobs = vertex.shape[0]
    pos_all_blobs = np.zeros((n_steps, Nrods * blobs, 3))
    for t in range(n_steps):
        idx = 0
        for r in range(Nrods):
            x, y, z, qx, qy, qz, qw = rods_positions[t, r]
            R = quaternion_rotation_matrix(qx, qy, qz, qw)
            center = np.array([x, y, z])
            rotated_blobs = (R @ vertex.T).T + center
            pos_all_blobs[t, idx:idx+blobs, :] = rotated_blobs
            idx += blobs
    return pos_all_blobs


# === PARTIE 1 ===
if run_single_file:
    print("→ Traitement du fichier step_0")
    file_name = f"step_0_update_{root_name}{config.nb_blobs}_blobs_{config.nb_rods}_rods.config"
    filepath = os.path.join(sim_path, file_name)

    rods_positions = load_cm_positions(filepath, Nrods)
    if rods_positions is not None:
        cm_positions = rods_positions[:, :, 0:3].mean(axis=1)
        cm_velocity = np.diff(cm_positions, axis=0) / dt
        pos_all_blobs = compute_all_blob_positions(rods_positions, vertex)

        n_steps = pos_all_blobs.shape[0]
        L = Nrods * 2 * a
        theta = np.linspace(0, 2 * np.pi, 100)

        xmin = np.min(pos_all_blobs[:, :, 0])
        xmax = np.max(pos_all_blobs[:, :, 0])
        zmin = np.min(pos_all_blobs[:, :, 2])
        zmax = np.max(pos_all_blobs[:, :, 2])
        maxi = max(abs(xmin), xmax, abs(zmin), zmax)
        xmin, xmax = -maxi, maxi
        zmin, zmax = -maxi, maxi

        for i in range(n_steps):
            plt.figure(figsize=(12.8, 9.6))
            plt.xlabel('x/L')
            plt.ylabel('z/L')
            plt.xlim(((xmin - 3 * a) / L, (xmax + 3 * a) / L))
            plt.ylim(((zmin - 3 * a) / L, (zmax + 3 * a) / L))

            for j in range(nb_blobs * Nrods):
                x = a * np.cos(theta) / L + pos_all_blobs[i, j, 0] / L
                z = a * np.sin(theta) / L + pos_all_blobs[i, j, 2] / L
                plt.plot(x, z, '#d2aa0f', zorder=-2)
                plt.fill_between(x, z, facecolor='#d2aa0f')

            # Centre de masse courant (point bleu)
            plt.plot(cm_positions[i, 0] / L, cm_positions[i, 2] / L, 'bo', markersize=8, label='CM actuel')
            # Trace historique du CM (ligne rouge)
            plt.plot(cm_positions[0:i + 1, 0] / L, cm_positions[0:i + 1, 2] / L, 'r-', lw=2, label='Trajectoire CM')
            state = DiatomEnv.infer_colony_state_from_positions(rods_positions[i, :, 0:3], rods_positions[i, :, 3:], a)
            plt.title(f"Colony State {state}")
            plt.legend()
            # plt.show()

            plt.savefig(f'N_{Nrods}_Step_{i}.pdf')
            plt.close()

# === PARTIE 2 : TOUS LES ÉPISODES ===
if run_multiple_episodes:
    print("→ Traitement de tous les fichiers .config")
    all_files = sorted([
        f for f in os.listdir(sim_path)
        if f.startswith("step_") and f.endswith(".config")
    ])
    episode_velocities = []

    for idx, filename in enumerate(all_files):
        path = os.path.join(sim_path, filename)
        cm_positions, cm_velocities = load_cm_positions(path, Nrods)
        if cm_positions is None:
            continue

        vel_norms = np.linalg.norm(cm_velocities, axis=1)
        mean_vel = np.mean(vel_norms)
        episode_velocities.append(mean_vel)

        if idx % trace_every_n == 0:
            plt.figure(figsize=(8, 6))
            plt.plot(cm_positions[:, 0], cm_positions[:, 2], label=f'Épisode {idx}')
            plt.xlabel('x')
            plt.ylabel('z')
            plt.title(f'Trajectoire - épisode {idx}')
            plt.axis('equal')
            plt.grid(True)
            plt.savefig(f"traj_cm_episode_{idx}.png", dpi=300)
            plt.close()

    # Courbe vitesse moyenne
    if episode_velocities:
        plt.figure(figsize=(8, 5))
        plt.plot(episode_velocities, '-o')
        plt.xlabel('Épisode')
        plt.ylabel('Vitesse moyenne CM')
        plt.title('Évolution de la vitesse moyenne')
        plt.grid(True)
        plt.savefig("vitesse_moyenne_par_episode.png", dpi=300)
        plt.show()
