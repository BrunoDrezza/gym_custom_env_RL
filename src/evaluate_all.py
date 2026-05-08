# src/evaluate_all.py

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os

# Importa as duas versões usando aliases
from gymnasium_env.grid_world_cpp_dumb import GridWorldCPPEnv as DumbEnv
from gymnasium_env.grid_world_cpp_smart import GridWorldCPPEnv as SmartEnv

# Registra os dois ambientes separadamente
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

def evaluate_scenario(model_path, env_type, dim, obstacles, max_steps, num_episodes=100):
    if not os.path.exists(model_path):
        return f"Modelo não encontrado em: {model_path}", None, None

    print(f"Avaliando cenário {dim}x{dim} ({env_type.upper()}) com modelo {model_path}...")
    
    # Define qual ID usar com base no tipo
    env_id = "gymnasium_env/GridWorldCPP-Smart-v0" if env_type == "smart" else "gymnasium_env/GridWorldCPP-Dumb-v0"
    
    model = PPO.load(model_path)
    env = gym.make(
        env_id,
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        render_mode="rgb_array",
    )

    full_coverage_count = 0
    total_coverages = []
    total_steps_list = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action.item())
            steps += 1

        total_coverages.append(info["coverage"])
        total_steps_list.append(steps)

        if done and not truncated:
            full_coverage_count += 1

    full_coverage_rate = (full_coverage_count / num_episodes) * 100
    avg_coverage = np.mean(total_coverages) * 100
    avg_steps = np.mean(total_steps_list)

    return full_coverage_rate, avg_coverage, avg_steps

if __name__ == "__main__":
    print("Iniciando bateria de testes...")

    # Dicionário de configurações: Agora inclui qual ambiente o modelo usa!
    scenarios = [
        # Modelos Base (Míopes)
        {"type": "dumb", "dim": 5, "obs": 3, "steps": 200, "model": "data/ppo_cpp_5_3_200_0.05_20260507_142958.zip"},
        {"type": "dumb", "dim": 10, "obs": 12, "steps": 400, "model": "data/ppo_cpp_10_12_400_0.05_20260507_151104_curriculum.zip"},
        
        # Modelos com Bússola e Reward Shaping (Apenas ajuste o nome quando treiná-los!)
        {"type": "smart", "dim": 5, "obs": 3, "steps": 200, "model": "data/modelo_smart_5x5.zip"},
        {"type": "smart", "dim": 10, "obs": 12, "steps": 400, "model": "data/modelo_smart_10x10.zip"},
        {"type": "smart", "dim": 20, "obs": 48, "steps": 800, "model": "data/modelo_smart_20x20.zip"},
    ]

    print("\n| Cenário | Tipo  | Taxa de Sucesso | Cobertura Média | Passos Médios |")
    print("|---------|-------|-----------------|-----------------|---------------|")

    for config in scenarios:
        success_rate, avg_cov, avg_steps = evaluate_scenario(
            model_path=config["model"],
            env_type=config["type"],
            dim=config["dim"],
            obstacles=config["obs"],
            max_steps=config["steps"],
        )

        if avg_cov is not None:
            print(f"| {config['dim']}x{config['dim']}   | {config['type']:5s} | {success_rate:14.1f}% | {avg_cov:14.2f}% | {avg_steps:13.1f} |")
        else:
            print(f"| {config['dim']}x{config['dim']}   | {config['type']:5s} | ERRO: Modelo não encontrado                 |")

    print("\nAvaliação concluída.")