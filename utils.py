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

DATA_DIR = "data"   # Folder containing all stock CSVs
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

    # Merge by common dates
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



def printAll(hc_score_p, sa_score_p, ts_score_p, ga_score, timeHc, timeSa, timeTs, timeGa, mainStartTime, mainFinalTime):
    results = {
        "Hill Climbing": {"score": hc_score_p, "time": timeHc},
        "Simulated Annealing": {"score": sa_score_p, "time": timeSa},
        "Tabu Search": {"score": ts_score_p, "time": timeTs},
        "Genetic Algorithm": {"score": ga_score, "time": timeGa}
    }

    print("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("📊 FINAL RESULTS SUMMARY")
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print(f"{'Algorithm':<25} {'Best Score':<15} {'Time (s)':<10}")
    print("-" * 55)

    for algo, res in results.items():
        print(f"{algo:<25} {res['score']:<15.6f} {res['time']:<10}")

    best_algo = max(results, key=lambda k: results[k]['score'])
    print("-" * 55)
    print(f"🏆 Best Algorithm: {best_algo} → Score = {results[best_algo]['score']:.6f}")
    print(f"⏱️ Total Execution Time: {round(mainFinalTime - mainStartTime, 2)} seconds")
    print(f"⏱️ Total Execution Time: {round((mainFinalTime - mainStartTime) / 60, 2)} minutes")

    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")


def plotAll(results, best_algo, hc_history, sa_history, ts_history, ga_history, hc_score_p, sa_score_p, ts_score_p, ga_score, mainStartTime, mainFinalTime):
    algo_names = ["Hill Climbing", "Simulated Annealing", "Tabu Search", "Genetic Algorithm"]
    algo_scores = [hc_score_p, sa_score_p, ts_score_p, ga_score]

    plt.figure(figsize=(8, 5))
    plt.bar(algo_names, algo_scores, color="skyblue")
    plt.title("Comparação da Qualidade das Soluções")
    plt.xlabel("Algoritmo")
    plt.ylabel("Score (Retorno - λ*Risco)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results_summary.png", dpi=150)
    plt.show()


    algo_times = [results[a]["time"] for a in algo_names]
    plt.figure(figsize=(8, 5))
    plt.bar(algo_names, algo_times, color="lightcoral")
    plt.title("Tempo de Execução por Algoritmo")
    plt.xlabel("Algoritmo")
    plt.ylabel("Tempo (segundos)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results_time.png", dpi=150)
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(hc_history, label="Hill Climbing")
    plt.plot(sa_history, label="Simulated Annealing")
    plt.plot(ts_history, label="Tabu Search")
    plt.plot(ga_history, label="Genetic Algorithm")
    plt.title("Curva de Evolução – Score ao Longo das Iterações")
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Score Encontrado")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("evolution_curve.png", dpi=150)
    plt.show()

    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

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

    summary_png = os.path.join(RESULTS_DIR, "results_summary.png")
    time_png = os.path.join(RESULTS_DIR, "results_time.png")
    evolution_png = os.path.join(RESULTS_DIR, "evolution_curve.png")  # optional curve chart if generated

    if os.path.exists("results_summary.png"):
        os.replace("results_summary.png", summary_png)
    if os.path.exists("results_time.png"):
        os.replace("results_time.png", time_png)
    if os.path.exists("evolution_curve.png"):
        os.replace("evolution_curve.png", evolution_png)

    print("\n✅ Gráficos e arquivos de resultados gerados com sucesso na pasta 'results':")
    print(f"   - {txt_path}")
    print(f"   - {csv_path}")
    if os.path.exists(summary_png): print(f"   - {summary_png}")
    if os.path.exists(time_png): print(f"   - {time_png}")
    if os.path.exists(evolution_png): print(f"   - {evolution_png}")