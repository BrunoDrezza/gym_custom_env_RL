# src/plot_learning_curve.py

import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_curve(csv_path, save_name, title):
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo {csv_path} não encontrado.")
        return

    # Lê os dados do treinamento
    df = pd.read_csv(csv_path)

    # O Stable Baselines 3 salva os timesteps e a recompensa média
    x = df["time/total_timesteps"]
    y = df["rollout/ep_rew_mean"]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Recompensa Média", color="blue", linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Recompensa Média por Episódio", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Salva na pasta data
    save_path = f"data/{save_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Gráfico salvo com sucesso em: {save_path}")
    plt.show()


if __name__ == "__main__":
    # Importante: Substitua pelo nome exato da pasta de log gerada no seu treinamento do 5x5
    # Exemplo: "log/ppo_cpp_5_3_200_0.05_20260507_142958/progress.csv"

    caminho_do_csv_5x5_dumb = "log/ppo_cpp_5_3_200_0.05_20260507_142958/progress.csv"
    caminho_do_csv_10x10_dumb = (
        "log/ppo_cpp_10_12_400_0.05_20260507_151104_curriculum/progress.csv"
    )

    plot_curve(
        csv_path=caminho_do_csv_5x5_dumb,
        save_name="5x5_learning_curve_dumb",
        title="Curva de Aprendizado - Coverage Path Planning (5x5) dumb",
    )

    plot_curve(
        csv_path=caminho_do_csv_10x10_dumb,
        save_name="10x10_learning_curve_dumb",
        title="Curva de Aprendizado - Coverage Path Planning (10x10) dumb",
    )
