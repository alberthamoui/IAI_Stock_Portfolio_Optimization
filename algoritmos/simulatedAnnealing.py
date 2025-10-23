from utils import *


def simulatedAnnealingOnce(problema, iterations=10000, initialTemp=1.0, coolingRate=0.995, stepSize=0.05):
    """
    Executa uma única rodada do Simulated Annealing.
    """
    peso = problema.gerar_estado_inicial()
    bestPeso = peso.copy()
    bestScore, _, _ = problema.avaliar(peso)
    currentScore = bestScore
    temperature = initialTemp
    history = [bestScore]

    for i in range(iterations):
        novoPeso = problema.gerar_vizinho(peso, stepSize)
        newScore, _, _ = problema.avaliar(novoPeso)
        delta = newScore - currentScore

        # Critério de aceitação de Metropolis
        if delta > 0 or random.random() < math.exp(delta / (temperature + 1e-9)):
            peso = novoPeso
            currentScore = newScore
            if newScore > bestScore:
                bestPeso = novoPeso
                bestScore = newScore

        history.append(bestScore)
        temperature *= coolingRate  # Reduz a temperatura gradualmente

    return bestPeso, bestScore, history


def simulatedAnnealingParallel(problema, runs=8, iterations=10000, initialTemp=1.0, coolingRate=0.995, stepSize=0.05):
    """
    Executa várias rodadas de SA em paralelo e retorna o melhor resultado.
    """
    print(f"Running Simulated Annealing ({runs} parallel runs)...")
    results = []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(simulatedAnnealingOnce, problema, iterations, initialTemp, coolingRate, stepSize)
            for _ in range(runs)
        ]
        for idx, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            print(f"  Run {idx}/{runs} complete")

    best = max(results, key=lambda x: x[1])
    return best
