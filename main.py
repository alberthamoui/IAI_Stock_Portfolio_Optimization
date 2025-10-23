from utils import loadData
from problemas.portfolio import ProblemaPortfolio
from agent import AgenteOtimizacao
from algoritmos.hillClimbing import hillClimbingParallel
from algoritmos.simulatedAnnealing import simulatedAnnealingParallel
from algoritmos.tabuSearch import tabuSearchParallel
from algoritmos.geneticAlgorithm import geneticAlgorithm
import time

# 1. Dados
stocks = loadData()
meanReturns = stocks.mean()
matrizCov = stocks.cov()
problema = ProblemaPortfolio(meanReturns, matrizCov, lambda_=0.5)

# 2. Hill Climbing
agente_hc = AgenteOtimizacao(problema, hillClimbingParallel)
res_hc = agente_hc.executar(runs=8, iterations=5000, stepSize=0.05)

# 3. Simulated Annealing
agente_sa = AgenteOtimizacao(problema, simulatedAnnealingParallel)
res_sa = agente_sa.executar(runs=8, iterations=5000, stepSize=0.05, initialTemp=1.0, coolingRate=0.995)

# 4. Tabu Search
agente_tb = AgenteOtimizacao(problema, tabuSearchParallel)
res_tb = agente_tb.executar(runs=8, iterations=1000, stepSize=0.05, tabuSize=50)

# 5. Algoritmo Genético
agente_ga = AgenteOtimizacao(problema, geneticAlgorithm)
res_ga = agente_ga.executar(runs=8, pop_size=30, generations=200, crossover_rate=0.8, mutation_rate=0.1, stepSize=0.05)

# 5. Comparação final
print("\n==== RESULTADOS FINAIS ====")
print(f"Hill Climbing     → Score: {res_hc['melhor_valor']:.6f} | Tempo: {res_hc['tempo']:.2f}s")
print(f"Simul. Annealing  → Score: {res_sa['melhor_valor']:.6f} | Tempo: {res_sa['tempo']:.2f}s")
print(f"Tabu Search       → Score: {res_tb['melhor_valor']:.6f} | Tempo: {res_tb['tempo']:.2f}s")
print(f"Genetic Algorithm → Score: {res_ga['melhor_valor']:.6f} | Tempo: {res_ga['tempo']:.2f}s")

melhor = max(
    [('Hill Climbing', res_hc), ('Simulated Annealing', res_sa), ('Tabu Search', res_tb), ('Genetic Algorithm', res_ga)],
    key=lambda x: x[1]['melhor_valor']
)
print(f"\n🏆 Melhor algoritmo: {melhor[0]} (Score: {melhor[1]['melhor_valor']:.6f})")
