from utils import loadData, printAll, plotAll, LAMBDA
from problemas.portfolio import ProblemaPortfolio
from agent import AgenteOtimizacao
from algoritmos.hillClimbing import hillClimbingParallel
from algoritmos.simulatedAnnealing import simulatedAnnealingParallel
from algoritmos.tabuSearch import tabuSearchParallel
from algoritmos.geneticAlgorithm import geneticAlgorithmParallel
import time

# Definir parâmetros 
runsHC = 8
iterationsHC = 5000
stepSizeHC = 0.05

runSA = 8
iterationsSA = 5000
stepSizeSA = 0.05
initialTempSA = 1.0
coolingRateSA = 0.995

runsTS = 8
iterationsTS = 1000
stepSizeTS = 0.05
tabuSizeTS = 50

runsGA = 8
pop_sizeGA = 30
generationsGA = 200
crossover_rateGA = 0.8
mutation_rateGA = 0.1
stepSizeGA = 0.05

# 0. Tempo total
mainStartTime = time.time()

# 1. Dados
stocks = loadData()
meanReturns = stocks.mean()
matrizCov = stocks.cov()
problema = ProblemaPortfolio(meanReturns, matrizCov, lambda_=LAMBDA)

# 2. Hill Climbing
agente_hc = AgenteOtimizacao(problema, hillClimbingParallel)
res_hc = agente_hc.executar(runs=runsHC, iterations=iterationsHC, stepSize=stepSizeHC)

# 3. Simulated Annealing
agente_sa = AgenteOtimizacao(problema, simulatedAnnealingParallel)
res_sa = agente_sa.executar(runs=runSA, iterations=iterationsSA, stepSize=stepSizeSA,
                            initialTemp=initialTempSA, coolingRate=coolingRateSA)

# 4. Tabu Search
agente_tb = AgenteOtimizacao(problema, tabuSearchParallel)
res_tb = agente_tb.executar(runs=runsTS, iterations=iterationsTS, stepSize=stepSizeTS, tabuSize=tabuSizeTS)
# 5. Genetic Algorithm
agente_ga = AgenteOtimizacao(problema, geneticAlgorithmParallel)
res_ga = agente_ga.executar(runs=runsGA, pop_size=pop_sizeGA, generations=generationsGA,
                            crossover_rate=crossover_rateGA, mutation_rate=mutation_rateGA, stepSize=stepSizeGA)
# 6. Tempo final
mainFinalTime = time.time()

# 7. Scores, tempos, históricos
hc_score_p = res_hc["melhor_valor"]
sa_score_p = res_sa["melhor_valor"]
ts_score_p = res_tb["melhor_valor"]
ga_score   = res_ga["melhor_valor"]

timeHc = res_hc["tempo"]
timeSa = res_sa["tempo"]
timeTs = res_tb["tempo"]
timeGa = res_ga["tempo"]

hc_history = res_hc["historico"]
sa_history = res_sa["historico"]
ts_history = res_tb["historico"]
ga_history = res_ga["historico"]

# 8. Pesos dos melhores portfólios
hc_best_weights = res_hc["melhor_estado"]
sa_best_weights = res_sa["melhor_estado"]
ts_best_weights = res_tb["melhor_estado"]
ga_best_weights = res_ga["melhor_estado"]

asset_names = stocks.columns.tolist()

# 9. Imprimir resumo e gerar arquivos/gráficos
results, best_algo = printAll(
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
)

plotAll(
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
)
