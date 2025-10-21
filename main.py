import pandas as pd
import numpy as np
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import csv


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

DATA_DIR = "data"   # Folder containing all stock CSVs
LAMBDA = 0.5        # Trade-off between return and risk
                    # ↑ higher lambda = safer (lower risk, lower return)


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def evaluate(peso, meanReturns, matrizCov, lambda_=LAMBDA):
    peso = np.clip(peso, 0, None)
    peso = peso / np.sum(peso)

    retorno = np.dot(peso, meanReturns)
    risco = np.sqrt(np.dot(peso.T, np.dot(matrizCov, peso)))
    score = retorno - lambda_ * risco

    return score, retorno, risco


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def hillClimbingOnce(args):
    meanReturns, matrizCov, iterations, stepSize = args
    n = len(meanReturns)

    peso = np.random.random(n)
    bestScore, _, _ = evaluate(peso, meanReturns, matrizCov)
    history = [bestScore]  # histórico de evolução

    for i in range(iterations):
        novoPeso = peso + np.random.uniform(-stepSize, stepSize, n)
        novoPeso = np.clip(novoPeso, 0, None)
        novoPeso = novoPeso / np.sum(novoPeso)

        newScore, _, _ = evaluate(novoPeso, meanReturns, matrizCov)
        if newScore > bestScore:
            bestScore = newScore
            peso = novoPeso

        history.append(bestScore)

        print(f"Hill Climbing {100 * i / iterations:.2f}% done")

    return peso, bestScore, history


def hillClimbingParallel(meanReturns, matrizCov, runs=8, iterations=10000, stepSize=0.05):
    print(f"Running Hill Climbing ({runs} parallel runs)...")
    tasks = [(meanReturns, matrizCov, iterations, stepSize)] * runs
    results = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(hillClimbingOnce, t) for t in tasks]
        for idx, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            print(f"  Run {idx}/{runs} complete")

    best = max(results, key=lambda x: x[1])
    return best


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def simulatedAnnealing(meanReturns, matrizCov, iterations=2000, temp=1.0, cooling=0.995):
    n = len(meanReturns)
    peso = np.random.random(n)
    peso = np.clip(peso, 0, None)
    peso = peso / np.sum(peso)

    melhorPeso = peso.copy()
    bestScore, _, _ = evaluate(peso, meanReturns, matrizCov)
    scoreAtual = bestScore
    history = [bestScore]

    for i in range(iterations):
        novoPeso = peso + np.random.uniform(-0.1, 0.1, n)
        novoPeso = np.clip(novoPeso, 0, None)
        novoPeso = novoPeso / np.sum(novoPeso)
        newScore, _, _ = evaluate(novoPeso, meanReturns, matrizCov)

        if newScore > scoreAtual or np.exp((newScore - scoreAtual) / temp) > random.random():
            peso = novoPeso
            scoreAtual = newScore
            if scoreAtual > bestScore:
                bestScore = scoreAtual
                melhorPeso = peso.copy()

        temp = max(temp * cooling, 1e-6)
        history.append(bestScore)

        print(f"Simulated Annealing {100 * i / iterations:.2f}% done")

    return melhorPeso, bestScore, history

def simulatedAnnealingParallel(meanReturns, matrizCov, runs=8, iterations=2000, temp=1.0, cooling=0.995):
    print(f"Running Simulated Annealing ({runs} parallel runs)...")
    tasks = [(meanReturns, matrizCov, iterations, temp, cooling)] * runs
    results = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(simulatedAnnealing, *t) for t in tasks]
        for idx, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            print(f"  Run {idx}/{runs} complete")

    best = max(results, key=lambda x: x[1])
    return best


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def tabuSearch(meanReturns, matrizCov, iterations=1000, stepSize=0.05, tabuSize=30):
    n = len(meanReturns)
    current = np.random.random(n)
    current = np.clip(current, 0, None)
    current = current / np.sum(current)
    best = current.copy()

    bestScore, _, _ = evaluate(current, meanReturns, matrizCov)
    tabuList = [tuple(np.round(current, 3))]
    history = [bestScore]

    for i in range(iterations):
        neighborhood = []
        for _ in range(20):
            neighbor = current + np.random.uniform(-stepSize, stepSize, n)
            neighbor = np.clip(neighbor, 0, None)
            neighbor = neighbor / np.sum(neighbor)
            if tuple(np.round(neighbor, 3)) not in tabuList:
                neighborhood.append(neighbor)

        if not neighborhood:
            continue

        scores = [evaluate(w, meanReturns, matrizCov)[0] for w in neighborhood]
        bestNeighbor = neighborhood[np.argmax(scores)]
        bestNeighborScore = np.max(scores)

        current = bestNeighbor
        if bestNeighborScore > bestScore:
            best = bestNeighbor
            bestScore = bestNeighborScore

        tabuList.append(tuple(np.round(bestNeighbor, 3)))
        if len(tabuList) > tabuSize:
            tabuList.pop(0)

        history.append(bestScore)

        print(f"  Tabu Search {100 * i / iterations:.2f}% done")

    return best, bestScore, history

def tabuSearchParallel(meanReturns, matrizCov, runs=8, iterations=1000, stepSize=0.05, tabuSize=30):
    print(f"Running Tabu Search ({runs} parallel runs)...")
    tasks = [(meanReturns, matrizCov, iterations, stepSize, tabuSize)] * runs
    results = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(tabuSearch, *t) for t in tasks]
        for idx, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            print(f"  Run {idx}/{runs} complete")

    best = max(results, key=lambda x: x[1])
    return best


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def geneticAlgorithm(meanReturns, matrizCov, population_size=30, generations=200, mutation_rate=0.1):
    n = len(meanReturns)
    pop = [np.random.random(n) for _ in range(population_size)]
    history = []

    for g in range(generations):
        scores = [evaluate(ind, meanReturns, matrizCov)[0] for ind in pop]
        bestScore = max(scores)
        history.append(bestScore)

        ranked = [x for _, x in sorted(zip(scores, pop), key=lambda x: x[0], reverse=True)]
        new_pop = ranked[:int(0.2 * population_size)]  # elitism

        while len(new_pop) < population_size:
            parents = random.sample(ranked[:10], 2)
            point = random.randint(1, n - 1)
            child = np.concatenate((parents[0][:point], parents[1][point:]))
            if random.random() < mutation_rate:
                child += np.random.uniform(-0.1, 0.1, n)
            child = np.clip(child, 0, None)
            child = child / np.sum(child)
            new_pop.append(child)

        pop = new_pop
        print(f"  Genetic Algorithm {100 * g / generations:.2f}% done")

    best = max(pop, key=lambda w: evaluate(w, meanReturns, matrizCov)[0])
    bestScore = evaluate(best, meanReturns, matrizCov)[0]
    return best, bestScore, history

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
mainStartTime = time.time()

# Hill Climbing
iterations_hc = 10_000
stepSize_hc = 0.05
start = time.time()
hc_w_p, hc_score_p, hc_history = hillClimbingParallel(meanReturns, matrizCov, runs=8, iterations=iterations_hc, stepSize=stepSize_hc)
# print(f"Hill Climbing best score: {hc_score_p:.6f} | Time: {round(time.time() - start, 2)}s\n")
os.system('cls' if os.name == 'nt' else 'clear')
timeHc = round(time.time() - start, 2)

# Simulated Annealing
iterations_sa = 10_000
temp_sa = 1.0
cooling_sa = 0.995
start = time.time()
sa_w_p, sa_score_p, sa_history = simulatedAnnealingParallel(meanReturns, matrizCov, runs=8, iterations=iterations_sa, temp=temp_sa, cooling=cooling_sa)
# print(f"Simulated Annealing best score: {sa_score_p:.6f} | Time: {round(time.time() - start, 2)}s\n")
os.system('cls' if os.name == 'nt' else 'clear')
timeSa = round(time.time() - start, 2)

# Tabu Search
iterations_ts = 1_000
stepSize_ts = 0.05
tabuSize_ts = 30
start = time.time()
ts_w_p, ts_score_p, ts_history = tabuSearchParallel(meanReturns, matrizCov, runs=8, iterations=iterations_ts, stepSize=stepSize_ts, tabuSize=tabuSize_ts)
# print(f"Tabu Search best score: {ts_score_p:.6f} | Time: {round(time.time() - start, 2)}s\n")
os.system('cls' if os.name == 'nt' else 'clear')
timeTs = round(time.time() - start, 2)

# Genetic Algorithm
population_size_ga = 30
generations_ga = 200
mutation_rate_ga = 0.1
start = time.time()
ga_w, ga_score, ga_history = geneticAlgorithm(meanReturns, matrizCov, population_size=population_size_ga, generations=generations_ga, mutation_rate=mutation_rate_ga)
# print(f"Genetic Algorithm best score: {ga_score:.6f} | Time: {round(time.time() - start, 2)}s")
os.system('cls' if os.name == 'nt' else 'clear')
timeGa = round(time.time() - start, 2)
mainFinalTime = time.time()



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