import argparse
import pickle
from sim_env import DiatomEnv, Action
from utils import load_config
import random


def validate_policy(env: DiatomEnv, Q: dict, episodes: int = 5):
    print("\n=== VALIDATION ===")
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(200):
            actions = env.get_available_actions(state)
            q_vals = [Q.get((state, a), 0.0) for a in actions]
            best_q = max(q_vals)
            best_actions = [a for a in actions if Q.get((state, a), 0.0) == best_q]
            action = random.choice(best_actions)

            next_state, inst_vel = env.step(action)
            reward = sum(v**2 for v in inst_vel)**0.5

            total_reward += reward
            state = next_state

        print(f"[Validation Episode {ep+1}] Total reward: {total_reward:.4f}")


def main(config_path: str, q_table_path: str, num_episodes: int):
    config = load_config(config_path)

    env = DiatomEnv(
        input_file_path=config.input_file_path,
        output_dir=config.output_directory,
        n_rods=config.Nrods,
        n_blobs=config.Nblobs,
        dt=config.dt,
        a=0.183228708092682
    )

    with open(q_table_path, "rb") as f:
        Q = pickle.load(f)

    validate_policy(env, Q, episodes=num_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation de politique Q-learning")
    parser.add_argument("--config", type=str, default="config.yaml", help="Fichier de configuration YAML")
    parser.add_argument("--qtable", type=str, default="q_table.pkl", help="Fichier .pkl de Q-table apprise")
    parser.add_argument("--episodes", type=int, default=5, help="Nombre d'épisodes de validation")

    args = parser.parse_args()

    main(args.config, args.qtable, args.episodes)
