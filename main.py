import pandas as pd
import os
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from algoritmos.hillClimbing import hillClimbingParallel
from algoritmos.simulatedAnnealing import simulatedAnnealingParallel
from algoritmos.tabuSearch import tabuSearchParallel
from algoritmos.geneticAlgorithm import geneticAlgorithm
from utils import *

stocks = loadData()
meanReturns = stocks.mean()
matrizCov = stocks.cov()
print(f"Loaded {len(meanReturns)} stocks | {stocks.shape[0]} daily records")


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
best_algo = max(results, key=lambda k: results[k]['score'])


print(printAll(hc_score_p, sa_score_p, ts_score_p, ga_score, timeHc, timeSa, timeTs, timeGa, mainStartTime, mainFinalTime))
print(plotAll(results, best_algo, hc_history, sa_history, ts_history, ga_history, hc_score_p, sa_score_p, ts_score_p, ga_score, mainStartTime, mainFinalTime))