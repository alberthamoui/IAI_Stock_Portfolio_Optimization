from utils import *

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

