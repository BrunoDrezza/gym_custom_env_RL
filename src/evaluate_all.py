# src/evaluate_all.py

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os
from gymnasium_env.grid_world_cpp_dumb import GridWorldCPPEnv

# Registra o ambiente customizado (com correção de tipagem para o Pylance)
try:
    gym.register(
        id="gymnasium_env/GridWorldCPP-v0",
        entry_point=lambda **kwargs: GridWorldCPPEnv(**kwargs),
    )
except Exception:
    pass


def evaluate_scenario(model_path, dim, obstacles, max_steps, num_episodes=100):
    if not os.path.exists(model_path):
        return f"Modelo não encontrado em: {model_path}", None, None

    print(f"Avaliando cenário {dim}x{dim} com modelo {model_path}...")
    model = PPO.load(model_path)
    env = gym.make(
        "gymnasium_env/GridWorldCPP-v0",
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        render_mode="rgb_array",  # Modo rápido, sem renderização visual
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
            action, _ = model.predict(
                obs, deterministic=True
            )  # Usamos determinístico no teste final
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

    # Dicionário de configurações com os caminhos corretos (sem ../)
    scenarios = [
        {
            "dim": 5,
            "obs": 3,
            "steps": 200,
            "model": "data/ppo_cpp_5_3_200_0.05_20260507_142958.zip",
        },
        {
            "dim": 10,
            "obs": 12,
            "steps": 400,
            "model": "data/ppo_cpp_10_12_400_0.05_20260507_151104_curriculum.zip",
        },
        {"dim": 20, "obs": 48, "steps": 800, "model": "data/modelo_final_20x20.zip"},
    ]

    print("\n| Cenário | Taxa de Sucesso (100%) | Cobertura Média | Passos Médios |")
    print("|---------|------------------------|-----------------|---------------|")

    for config in scenarios:
        success_rate, avg_cov, avg_steps = evaluate_scenario(
            model_path=config["model"],
            dim=config["dim"],
            obstacles=config["obs"],
            max_steps=config["steps"],
        )

        if avg_cov is not None:
            print(
                f"| {config['dim']}x{config['dim']}   | {success_rate:5.1f}%                 | {avg_cov:6.2f}%         | {avg_steps:7.1f}       |"
            )
        else:
            print(
                f"| {config['dim']}x{config['dim']}   | ERRO: Modelo não encontrado |                 |               |"
            )

    print("\nAvaliação concluída.")
