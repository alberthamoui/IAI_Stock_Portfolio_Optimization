from utils import loadData, printAll, plotAll, LAMBDA
from problemas.portfolio import ProblemaPortfolio
from agent import AgenteOtimizacao
from algoritmos.hillClimbing import hillClimbingParallel
from algoritmos.simulatedAnnealing import simulatedAnnealingParallel
from algoritmos.tabuSearch import tabuSearchParallel
from algoritmos.geneticAlgorithm import geneticAlgorithmParallel
import time
from itertools import product
import pandas as pd  # para organizar os resultados

# 0. Tempo total
mainStartTime = time.time()

# 1. Dados
stocks = loadData()
meanReturns = stocks.mean()
matrizCov = stocks.cov()
problema = ProblemaPortfolio(meanReturns, matrizCov, lambda_=LAMBDA)


# ===========================
# GRID DE PARÂMETROS
# ===========================

# Hill Climbing
param_grid_hc = {
    # "runs":       [4, 8],
    "runs":       [1],
    "iterations": [3000, 5000],
    "stepSize": [0.01, 0.03, 0.05, 0.08],     # "stepSize":   [0.01, 0.05],
}

# Simulated Annealing
param_grid_sa = {
    # "runs":        [4, 8],
    "runs":        [1],
    "iterations":  [3000, 5000],    
    "stepSize": [0.01, 0.03, 0.05, 0.08],     # "stepSize":   [0.01, 0.05],
    "initialTemp": [1.0, 3.0, 5.0], #     "initialTemp": [1.0, 5.0],
    "coolingRate": [0.99, 0.995]
}

# Tabu Search
param_grid_ts = {
    # "runs":       [4, 8],
    "runs":       [1],
    "iterations": [3000, 5000],
    "stepSize":    [0.01, 0.03, 0.05, 0.08],    # "stepSize":   [0.01, 0.05],
    "tabuSize": [10, 30, 50, 80], # "tabuSize":   [20, 50, 100], 
}

# Genetic Algorithm
# Aqui, note que 'iterations' não existe. Em vez disso temos pop_size e generations.
param_grid_ga = {
    # "runs":          [4, 8],
    "runs":          [1],
    "pop_size":      [20, 30],
    "generations":   [3000, 5000],
    "crossover_rate":[0.6, 0.7, 0.8, 0.9],
    "mutation_rate": [0.01, 0.05, 0.1],
    "stepSize":      [0.01, 0.03, 0.05, 0.08],     # "stepSize":   [0.01, 0.05],
}



def grid_search_algo(nome_algo, func_algo, problema, param_grid):
    """
    Faz um grid search manual para um algoritmo.
    param_grid: dict com nome_param -> lista de valores
    """
    keys = list(param_grid.keys())
    values_product = list(product(*param_grid.values()))

    results = []

    print(f"\n===== Grid search para {nome_algo} ({len(values_product)} combinações) =====")

    for combo_idx, combo in enumerate(values_product, 1):
        params = dict(zip(keys, combo))
        print(f"[{nome_algo}] Combinação {combo_idx}/{len(values_product)}: {params}")

        # Criar agente e executar com esses parâmetros
        agente = AgenteOtimizacao(problema, func_algo)
        res = agente.executar(**params)

        results.append({
            "algoritmo":   nome_algo,
            **params,
            "melhor_valor": res["melhor_valor"],
            "tempo":        res["tempo"],
        })

    return results


# ===========================
# RODAR GRID SEARCH
# ===========================

results_hc = grid_search_algo("HillClimbing", hillClimbingParallel, problema, param_grid_hc)
results_sa = grid_search_algo("SimulatedAnnealing", simulatedAnnealingParallel, problema, param_grid_sa)
results_ts = grid_search_algo("TabuSearch", tabuSearchParallel, problema, param_grid_ts)
results_ga = grid_search_algo("GeneticAlgorithm", geneticAlgorithmParallel, problema, param_grid_ga)

# Juntar tudo
all_results = results_hc + results_sa + results_ts + results_ga

df_results = pd.DataFrame(all_results)

# Ordenar por melhor_valor (maior é melhor) e depois por tempo (menor é melhor)
df_results = df_results.sort_values(by=["melhor_valor", "tempo"], ascending=[False, True])

print("\n===== TOP 10 CONFIGURAÇÕES =====")
print(df_results.head(10))

# Se quiser salvar para analisar com calma:
df_results.to_csv("grid_search_results.csv", index=False)

mainFinalTime = time.time()
print(f"\nTempo total do grid search: {mainFinalTime - mainStartTime:.2f} segundos")


best_ga_row = df_results[df_results["algoritmo"] == "GeneticAlgorithm"].iloc[0]
print("\nMelhor GA encontrado:")
print(best_ga_row.to_dict())
