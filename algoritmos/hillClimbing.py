from utils import *

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
