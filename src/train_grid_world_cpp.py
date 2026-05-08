#
# python train_grid_world_cpp.py <train|test|run|curriculum> <dumb|smart> dim obstacles max_steps [total_timesteps]
#

import gymnasium as gym
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

# Importa as duas versões do ambiente usando aliases
from gymnasium_env.grid_world_cpp_dumb import GridWorldCPPEnv as DumbEnv
from gymnasium_env.grid_world_cpp_smart import GridWorldCPPEnv as SmartEnv


def print_action(action: int) -> str:
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(action, "unknown")


# Validação principal dos argumentos
if len(sys.argv) < 3 or sys.argv[1] not in ["train", "test", "run", "curriculum"] or sys.argv[2] not in ["dumb", "smart"]:
    print(
        "Usage: python train_grid_world_cpp.py <train|test|run|curriculum> <dumb|smart> dim obstacles max_steps [total_timesteps]"
    )
    sys.exit(1)

mode = sys.argv[1]
env_type = sys.argv[2]

# Validação da quantidade de argumentos dependendo do modo
if mode in ["train", "curriculum"]:
    if len(sys.argv) != 7:
        print(
            "Usage for training: python train_grid_world_cpp.py train|curriculum <dumb|smart> dim obstacles max_steps total_timesteps"
        )
        sys.exit(1)
elif mode in ["test", "run"]:
    if len(sys.argv) != 6:
        print(
            "Usage for testing/running: python train_grid_world_cpp.py test|run <dumb|smart> dim obstacles max_steps"
        )
        sys.exit(1)

# --- Hyperparameters ---
DIM = int(sys.argv[3])          # 5, 10, 20
OBSTACLES = int(sys.argv[4])    # 3, 12, 48
MAX_STEPS = int(sys.argv[5])    # 200, 500, 1000
TOTAL_TIMESTEPS = int(sys.argv[6]) if len(sys.argv) == 7 else 0
ENTROPY_COEF = 0.05
# -----------------------

# Registro Duplo dos Ambientes
try:
    gym.register(
        id="gymnasium_env/GridWorldCPP-Dumb-v0",
        entry_point=lambda **kwargs: DumbEnv(**kwargs),
    )
    gym.register(
        id="gymnasium_env/GridWorldCPP-Smart-v0",
        entry_point=lambda **kwargs: SmartEnv(**kwargs),
    )
except Exception:
    pass

# Define qual ID o script vai usar baseado no input do usuário
TARGET_ENV_ID = "gymnasium_env/GridWorldCPP-Smart-v0" if env_type == "smart" else "gymnasium_env/GridWorldCPP-Dumb-v0"


if mode == "train":
    print(f"--- Starting CPP Training ({env_type.upper()}) ---")
    env = gym.make(
        TARGET_ENV_ID,
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )
    check_env(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        ent_coef=ENTROPY_COEF,
        gamma=0.999,  # Visão de longo prazo
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        device="cpu",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Adicionamos o env_type no nome para organizar os arquivos gerados
    log_dir = f"log/ppo_cpp_{env_type}_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}"
    model_path = (
        f"data/ppo_cpp_{env_type}_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}.zip"
    )

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Starting learning with {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == "curriculum":
    print(f"--- Starting CPP Curriculum Learning Training ({env_type.upper()}) ---")

    model_name = input(
        "Enter model filename (e.g., ppo_cpp_smart_5_3_200_0.05_20260324_100000): "
    )
    model_path = f"data/{model_name}.zip"

    env = gym.make(
        TARGET_ENV_ID,
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )

    # Carrega os pesos do modelo anterior e associa ao novo ambiente
    model = PPO.load(model_path, env=env, device="cpu")

    # Primeiro reset de timesteps não zera completamente os pesos, é uma prática do SB3 para transfer learning
    model.learn(total_timesteps=MAX_STEPS, reset_num_timesteps=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"log/ppo_cpp_{env_type}_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}_curriculum"
    model_path = f"data/ppo_cpp_{env_type}_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}_curriculum.zip"

    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Starting curriculum learning with {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == "run":
    model_name = input(
        "Enter model filename (e.g., ppo_cpp_smart_5_3_200_0.05_20260324_100000): "
    )
    model_path = f"data/{model_name}.zip"
    print(f"--- Loading {env_type.upper()} model from {model_path} for a run ---")

    model = PPO.load(model_path)
    env = gym.make(
        TARGET_ENV_ID,
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="human",
    )

    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0
    total_reward = 0
    
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action.item())
        total_reward += reward
        steps += 1
        print(
            f"Step: {steps}, Action: {print_action(action.item())}, "
            f"Reward: {reward:.2f}, Coverage: {info['coverage']:.1%}, "
            f"Done: {done}, Truncated: {truncated}"
        )
        
    print(
        f"--- Run Finished --- Total reward: {total_reward:.2f}, Coverage: {info['coverage']:.1%}"
    )

elif mode == "test":
    model_name = input(
        "Enter model filename (e.g., ppo_cpp_smart_5_3_200_0.05_20260324_100000): "
    )
    model_path = f"data/{model_name}.zip"
    print(f"--- Loading {env_type.upper()} model from {model_path} for testing ---")

    model = PPO.load(model_path)
    env = gym.make(
        TARGET_ENV_ID,
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array",
    )

    num_episodes = 100
    full_coverage_count = 0
    total_coverages = []
    total_steps_list = []

    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action.item())
            steps += 1

        total_coverages.append(info["coverage"])
        total_steps_list.append(steps)

        if done and not truncated:
            full_coverage_count += 1
            print(f"Episode {i+1}: Full coverage in {steps} steps.")
        else:
            print(f"Episode {i+1}: Coverage {info['coverage']:.1%} in {steps} steps.")

    import numpy as np

    full_coverage_rate = (full_coverage_count / num_episodes) * 100
    avg_coverage = np.mean(total_coverages) * 100
    standard_deviation = np.std(total_coverages) * 100
    avg_steps = np.mean(total_steps_list)
    standard_deviation_steps = np.std(total_steps_list)
    
    print(f"\n--- Test Finished ({env_type.upper()} Model) ---")
    print(
        f"Full Coverage Rate: {full_coverage_rate:.2f}% ({full_coverage_count}/{num_episodes})"
    )
    print(
        f"Average Coverage: {avg_coverage:.2f}% Standard Deviation: {standard_deviation:.2f}% Min Coverage: {np.min(total_coverages)*100:.2f}% Max Coverage: {np.max(total_coverages)*100:.2f}%"
    )
    print(
        f"Average Steps: {avg_steps:.1f} Standard Deviation: {standard_deviation_steps:.1f} Min Steps: {np.min(total_steps_list)} Max Steps: {np.max(total_steps_list)}"
    )