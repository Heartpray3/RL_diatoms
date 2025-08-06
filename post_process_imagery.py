import os
import numpy as np
import matplotlib.pyplot as plt
from sim_env import DiatomEnv
from utils import load_config, quaternion_rotation_matrix
from matplotlib.backends.backend_pdf import PdfPages

# === Chemins vers les fichiers .config à traiter ===
config_files = [
    "./ppo/ppo_3r_5b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_90/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_2309_update_bacillaria_5_blobs_3_rods.config",
    "./ppo/ppo_3r_5b_ep_5000_step_40_meth_CIRCULAR_ZONES_ang_0/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_1632_update_bacillaria_5_blobs_3_rods.config",
    "./ppo/ppo_3r_2b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_0/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_478_update_bacillaria_2_blobs_3_rods.config",
    "./ppo/ppo_3r_10b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_90/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_2203_update_bacillaria_10_blobs_3_rods.config",
    "./ppo/ppo_3r_10b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_0/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_440_update_bacillaria_10_blobs_3_rods.config",
    "./ppo/ppo_3r_10b_ep_5000_step_40_meth_CIRCULAR_ZONES_ang_0/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_1554_update_bacillaria_10_blobs_3_rods.config",
    "./ppo/ppo_3r_5b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_90/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_463_update_bacillaria_5_blobs_3_rods.config",
    "./ppo/ppo_3r_2b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_90/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_2384_update_bacillaria_2_blobs_3_rods.config",
    "./ppo/ppo_3r_5b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_0/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_2311_update_bacillaria_5_blobs_3_rods.config",
    "./ppo/ppo_3r_5b_ep_1000_step_200_meth_CIRCULAR_ZONES_ang_0/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_327_update_bacillaria_5_blobs_3_rods.config",
    "./ppo/ppo_3r_5b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_0/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_463_update_bacillaria_5_blobs_3_rods.config",
    "./ppo/ppo_3r_2b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_90/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_478_update_bacillaria_2_blobs_3_rods.config",
    "./ppo/ppo_3r_2b_ep_1000_step_200_meth_CIRCULAR_ZONES_ang_0/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_336_update_bacillaria_2_blobs_3_rods.config",
    "./ppo/ppo_3r_2b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_0/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_2386_update_bacillaria_2_blobs_3_rods.config",
    "./ppo/ppo_3r_10b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_90/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_439_update_bacillaria_10_blobs_3_rods.config",
    "./ppo/ppo_3r_10b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_0/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_2202_update_bacillaria_10_blobs_3_rods.config",
    "./ppo/ppo_3r_2b_ep_5000_step_40_meth_CIRCULAR_ZONES_ang_0/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_1680_update_bacillaria_2_blobs_3_rods.config",
    "./ppo/ppo_3r_10b_ep_1000_step_200_meth_CIRCULAR_ZONES_ang_0/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_311_update_bacillaria_10_blobs_3_rods.config",
    "./qlearning/qlearning_3r_5b_ep_1000_step_200_meth_CIRCULAR_ZONES_ang_0/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_348_update_bacillaria_5_blobs_3_rods.config",
    "./qlearning/qlearning_3r_5b_ep_5000_step_40_meth_CIRCULAR_ZONES_ang_0/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_1771_update_bacillaria_5_blobs_3_rods.config",
    "./qlearning/qlearning_3r_5b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_90/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_347_update_bacillaria_5_blobs_3_rods.config",
    "./qlearning/qlearning_3r_2b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_90/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_2692_update_bacillaria_2_blobs_3_rods.config",
    "./qlearning/qlearning_3r_10b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_90/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_1567_update_bacillaria_10_blobs_3_rods.config",
    "./qlearning/qlearning_3r_5b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_90/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_1771_update_bacillaria_5_blobs_3_rods.config",
    "./qlearning/qlearning_3r_10b_ep_5000_step_40_meth_CIRCULAR_ZONES_ang_0/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_1567_update_bacillaria_10_blobs_3_rods.config",
    "./qlearning/qlearning_3r_10b_ep_1000_step_200_meth_CIRCULAR_ZONES_ang_0/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_307_update_bacillaria_10_blobs_3_rods.config",
    "./qlearning/qlearning_3r_5b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_0/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_347_update_bacillaria_5_blobs_3_rods.config",
    "./qlearning/qlearning_3r_5b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_0/5_Blobs/3_Rods/bacillaria_5_blobs_3_rods/epoch_1771_update_bacillaria_5_blobs_3_rods.config",
    "./qlearning/qlearning_3r_2b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_0/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_528_update_bacillaria_2_blobs_3_rods.config",
    "./qlearning/qlearning_3r_2b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_0/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_2691_update_bacillaria_2_blobs_3_rods.config",
    "./qlearning/qlearning_3r_10b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_90/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_307_update_bacillaria_10_blobs_3_rods.config",
    "./qlearning/qlearning_3r_2b_ep_1000_step_200_meth_CIRCULAR_ZONES_ang_0/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_528_update_bacillaria_2_blobs_3_rods.config",
    "./qlearning/qlearning_3r_2b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_90/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_528_update_bacillaria_2_blobs_3_rods.config",
    "./qlearning/qlearning_3r_10b_ep_5000_step_40_meth_FORWARD_PROGRESS_ang_0/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_1565_update_bacillaria_10_blobs_3_rods.config",
    "./qlearning/qlearning_3r_2b_ep_5000_step_40_meth_CIRCULAR_ZONES_ang_0/2_Blobs/3_Rods/bacillaria_2_blobs_3_rods/epoch_2693_update_bacillaria_2_blobs_3_rods.config",
    "./qlearning/qlearning_3r_10b_ep_1000_step_200_meth_FORWARD_PROGRESS_ang_0/10_Blobs/3_Rods/bacillaria_10_blobs_3_rods/epoch_307_update_bacillaria_10_blobs_3_rods.config"
]

a = 0.183228708092682
dt = 0.005
theta = np.linspace(0, 2 * np.pi, 100)

def load_cm_positions(filepath, Nrods):
    with open(filepath, 'r') as f:
        lines = [list(map(float, l.strip().split())) for l in f if len(l.strip().split()) == 7]
    data = np.array(lines)
    if len(data) % Nrods != 0:
        print(f"⚠️ {filepath} mal formé.")
        return None
    n_steps = len(data) // Nrods
    return data.reshape((n_steps, Nrods, 7))

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
            rotated = (R @ vertex.T).T + center
            pos_all_blobs[t, idx:idx+blobs, :] = rotated
            idx += blobs
    return pos_all_blobs

for filepath in config_files:
    print(f"▶️ Traitement : {filepath}")

    identifier = filepath.split("/")[2]
    root = os.path.splitext(os.path.basename(filepath))[0]
    parts = root.split("_")
    nb_blobs = int(parts[4])
    nb_rods = int(parts[6])

    # Vertex file = deux niveaux au-dessus
    vertex_dir = os.path.dirname(os.path.dirname(os.path.dirname(filepath)))
    vertex_file = os.path.join(vertex_dir, f"bacillaria_{nb_blobs}_blobs.vertex")
    if not os.path.exists(vertex_file):
        print(f"❌ Pas de vertex file trouvé pour {filepath}")
        continue

    with open(vertex_file, 'r') as f:
        lines = f.read().split()
        nb_blobs_from_file = int(lines[0])
        vertex = np.array(lines[1:], dtype=float).reshape((-1, 3))

    rods_positions = load_cm_positions(filepath, nb_rods)
    if rods_positions is None:
        continue

    cm_positions = rods_positions[:, :, 0:3].mean(axis=1)
    pos_all_blobs = compute_all_blob_positions(rods_positions, vertex)
    n_steps = pos_all_blobs.shape[0]

    # Centrage
    xmin, xmax = np.min(pos_all_blobs[:, :, 0]), np.max(pos_all_blobs[:, :, 0])
    zmin, zmax = np.min(pos_all_blobs[:, :, 2]), np.max(pos_all_blobs[:, :, 2])
    maxi = max(abs(xmin), xmax, abs(zmin), zmax)
    L = 2 * a + (nb_rods - 1) * 0.81 * a
    xmin, xmax = -maxi, maxi
    zmin, zmax = -maxi, maxi

    # Output
    outdir = os.path.join("analyses", identifier)
    os.makedirs(outdir, exist_ok=True)

    pdf_path = os.path.join(outdir, "all_steps.pdf")
    with PdfPages(pdf_path) as pdf:
        for i in range(n_steps):
            fig = plt.figure(figsize=(12.8, 9.6))
            plt.xlabel('x/L')
            plt.ylabel('z/L')
            plt.xlim(((xmin - 3 * a) / L, (xmax + 3 * a) / L))
            plt.ylim(((zmin - 3 * a) / L, (zmax + 3 * a) / L))

            for j in range(nb_blobs_from_file * nb_rods):
                x = a * np.cos(theta) / L + pos_all_blobs[i, j, 0] / L
                z = a * np.sin(theta) / L + pos_all_blobs[i, j, 2] / L
                plt.plot(x, z, '#d2aa0f', zorder=-2)
                plt.fill_between(x, z, facecolor='#d2aa0f')

            plt.plot(cm_positions[i, 0] / L, cm_positions[i, 2] / L, 'bo', label='CM actuel')
            plt.plot(cm_positions[:i + 1, 0] / L, cm_positions[:i + 1, 2] / L, 'r-', lw=2, label='Trajectoire CM')
            state = DiatomEnv.infer_colony_state_from_positions(rods_positions[i, :, 0:3], rods_positions[i, :, 3:], a)
            plt.title(f"Colony State {state}")
            plt.legend()

            pdf.savefig(fig)
            plt.close(fig)
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.plot(cm_positions[:, 0] / L, cm_positions[:, 2] / L, 'r-', lw=2)
    plt.plot(cm_positions[0, 0] / L, cm_positions[0, 2] / L, 'go', label='Start', markersize=8)
    plt.plot(cm_positions[-1, 0] / L, cm_positions[-1, 2] / L, 'bo', label='End', markersize=8)
    plt.xlabel('x / L')
    plt.ylabel('z / L')
    plt.title(f"Trajectory of CM — {root}")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(outdir, "trajectory.png"), dpi=200)
    plt.close()