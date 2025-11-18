import pandas as pd
import os
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import csv
import matplotlib.pyplot as plt
import math
from collections import deque
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# DATA_DIR = "data"   # Folder containing all stock CSVs
LAMBDA = 0.5        # Trade-off between return and risk
                    # ↑ higher lambda = safer (lower risk, lower return)

def loadData(path=DATA_DIR):
    data = []

    for file in os.listdir(path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            stockName = file.replace(".csv", "")

            colDate = "Date" if "Date" in df.columns else df.columns[0]
            colPrice = next((c for c in df.columns if "Close" in c), None)
            if not colPrice:
                continue

            df = df[[colDate, colPrice]].rename(columns={colPrice: stockName})
            df[colDate] = pd.to_datetime(df[colDate])
            data.append(df)

    merged = data[0]
    for df in data[1:]:
        merged = pd.merge(merged, df, on="Date", how="inner")

    merged = merged.sort_values("Date")
    returns = merged.drop(columns=["Date"]).pct_change().dropna()
    return returns


stocks = loadData()
meanReturns = stocks.mean()
matrizCov = stocks.cov()
print(f"Loaded {len(meanReturns)} stocks | {stocks.shape[0]} daily records")


def evaluate(peso, meanReturns, matrizCov, lambda_=LAMBDA):
    peso = np.clip(peso, 0, None)
    peso = peso / np.sum(peso)

    retorno = np.dot(peso, meanReturns)
    risco = np.sqrt(np.dot(peso.T, np.dot(matrizCov, peso)))
    score = retorno - lambda_ * risco

    return score, retorno, risco


def printAll(
    hc_score_p,
    sa_score_p,
    ts_score_p,
    ga_score,
    timeHc,
    timeSa,
    timeTs,
    timeGa,
    mainStartTime,
    mainFinalTime,
):
    results = {
        "Hill Climbing":       {"score": hc_score_p, "time": timeHc},
        "Simulated Annealing": {"score": sa_score_p, "time": timeSa},
        "Tabu Search":         {"score": ts_score_p, "time": timeTs},
        "Genetic Algorithm":   {"score": ga_score,   "time": timeGa},
    }

    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("FINAL RESULTS SUMMARY")
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print(f"{'Algorithm':<25} {'Best Score':<15} {'Time (s)':<10}")
    print("-" * 55)

    for algo, res in results.items():
        print(f"{algo:<25} {res['score']:<15.6f} {res['time']:<10.2f}")

    best_algo = max(results, key=lambda k: results[k]['score'])
    total_time_sec = mainFinalTime - mainStartTime

    print("-" * 55)
    print(f"Best Algorithm: {best_algo} → Score = {results[best_algo]['score']:.6f}")
    print(f"Total Execution Time: {total_time_sec:.2f} seconds")
    print(f"Total Execution Time: {total_time_sec / 60:.2f} minutes")
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

    return results, best_algo


def _plot_score_bars(algo_names, algo_scores):
    plt.figure(figsize=(8, 5))
    plt.bar(algo_names, algo_scores)
    plt.title("Comparação da Qualidade das Soluções")
    plt.xlabel("Algoritmo")
    plt.ylabel("Score (Retorno - λ * Risco)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    summary_png = os.path.join(RESULTS_DIR, "results_summary.png")
    plt.savefig(summary_png, dpi=150)
    plt.close()
    return summary_png


def _plot_time_bars(algo_names, algo_times):
    plt.figure(figsize=(8, 5))
    plt.bar(algo_names, algo_times)
    plt.title("Tempo de Execução por Algoritmo")
    plt.xlabel("Algoritmo")
    plt.ylabel("Tempo (segundos)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    time_png = os.path.join(RESULTS_DIR, "results_time.png")
    plt.savefig(time_png, dpi=150)
    plt.close()
    return time_png


def _plot_evolution_curves(hc_history, sa_history, ts_history, ga_history):
    plt.figure(figsize=(10, 6))
    plt.plot(hc_history, label="Hill Climbing")
    plt.plot(sa_history, label="Simulated Annealing")
    plt.plot(ts_history, label="Tabu Search")
    plt.plot(ga_history, label="Genetic Algorithm")
    plt.title("Curva de Evolução – Melhor Score ao Longo das Iterações")
    plt.xlabel("Iterações / Gerações")
    plt.ylabel("Melhor Score Encontrado")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    evolution_png = os.path.join(RESULTS_DIR, "evolution_curve.png")
    plt.savefig(evolution_png, dpi=150)
    plt.close()
    return evolution_png


def _plot_scores_boxplot(algo_names, hc_history, sa_history, ts_history, ga_history):
    data = [hc_history, sa_history, ts_history, ga_history]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=algo_names)
    plt.title("Distribuição de Scores ao Longo da Execução")
    plt.xlabel("Algoritmo")
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    boxplot_png = os.path.join(RESULTS_DIR, "scores_boxplot.png")
    plt.savefig(boxplot_png, dpi=150)
    plt.close()
    return boxplot_png


def _plot_risk_return_scatter(algo_names, best_weights_dict):
    """
    best_weights_dict: { "Hill Climbing": pesos, ... }
    Usa evaluate() + meanReturns/matrizCov globais.
    """
    plt.figure(figsize=(8, 6))

    for name in algo_names:
        pesos = best_weights_dict[name]
        score, retorno, risco = evaluate(pesos, meanReturns, matrizCov)
        plt.scatter(risco, retorno)
        plt.annotate(name, (risco, retorno), textcoords="offset points", xytext=(5, 5))

    plt.title("Risco vs Retorno – Melhor Portfólio por Algoritmo")
    plt.xlabel("Risco (desvio padrão do portfólio)")
    plt.ylabel("Retorno Esperado")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    scatter_png = os.path.join(RESULTS_DIR, "risk_return_scatter.png")
    plt.savefig(scatter_png, dpi=150)
    plt.close()
    return scatter_png


def _plot_ga_top10_weights(ga_best_weights, asset_names):
    pesos = np.array(ga_best_weights)
    idx_sorted = np.argsort(pesos)[::-1]  # ordem decrescente
    top_k = idx_sorted[:10]

    labels = [asset_names[i] for i in top_k]
    values = pesos[top_k]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Top-10 Pesos do Melhor Portfólio (Genetic Algorithm)")
    plt.xlabel("Ativo")
    plt.ylabel("Peso no Portfólio")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    top10_png = os.path.join(RESULTS_DIR, "ga_top10_weights.png")
    plt.savefig(top10_png, dpi=150)
    plt.close()
    return top10_png


def plotAll(
    results,
    best_algo,
    hc_history,
    sa_history,
    ts_history,
    ga_history,
    hc_score_p,
    sa_score_p,
    ts_score_p,
    ga_score,
    mainStartTime,
    mainFinalTime,
    hc_best_weights,
    sa_best_weights,
    ts_best_weights,
    ga_best_weights,
    asset_names,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    algo_names = ["Hill Climbing", "Simulated Annealing", "Tabu Search", "Genetic Algorithm"]
    algo_scores = [hc_score_p, sa_score_p, ts_score_p, ga_score]
    algo_times = [results[a]["time"] for a in algo_names]

    # mapas com pesos dos melhores portfólios
    best_weights_dict = {
        "Hill Climbing": hc_best_weights,
        "Simulated Annealing": sa_best_weights,
        "Tabu Search": ts_best_weights,
        "Genetic Algorithm": ga_best_weights,
    }

    # 1) Barras de score
    summary_png = _plot_score_bars(algo_names, algo_scores)

    # 2) Barras de tempo
    time_png = _plot_time_bars(algo_names, algo_times)

    # 3) Curvas de evolução
    evolution_png = _plot_evolution_curves(hc_history, sa_history, ts_history, ga_history)

    # 4) Boxplot de scores
    boxplot_png = _plot_scores_boxplot(algo_names, hc_history, sa_history, ts_history, ga_history)

    # 5) Scatter Risco x Retorno (melhor portfólio de cada algoritmo)
    scatter_png = _plot_risk_return_scatter(algo_names, best_weights_dict)

    # 6) Top-10 pesos do melhor GA
    top10_png = _plot_ga_top10_weights(ga_best_weights, asset_names)

    # ---------- Arquivos TXT / CSV ----------
    txt_path = os.path.join(RESULTS_DIR, "results.txt")
    with open(txt_path, "w") as f:
        f.write("Resultados Finais – Metaheurísticas de Otimização\n")
        f.write("=====================================================\n\n")
        for algo, res in results.items():
            f.write(f"{algo}\n")
            f.write(f"  Melhor Score: {res['score']:.6f}\n")
            f.write(f"  Tempo: {res['time']:.2f} s\n")
            f.write("-----------------------------------------------------\n")

        f.write(f"Melhor algoritmo: {best_algo}\n")
        f.write(f"Score final: {results[best_algo]['score']:.6f}\n")
        f.write(f"Tempo total: {round(mainFinalTime - mainStartTime, 2)} s\n")

    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Algorithm", "Best Score", "Time (s)"])
        for algo, res in results.items():
            writer.writerow([algo, res["score"], res["time"]])

    print("\nGráficos e arquivos de resultados gerados com sucesso na pasta 'results':")
    print(f"   - {txt_path}")
    print(f"   - {csv_path}")
    print(f"   - {summary_png}")
    print(f"   - {time_png}")
    print(f"   - {evolution_png}")
    print(f"   - {boxplot_png}")
    print(f"   - {scatter_png}")
    print(f"   - {top10_png}")
