from utils import *

def hillClimbingOnce(problema, iterations=10000, stepSize=0.05):
    peso = problema.gerar_estado_inicial()
    bestScore, _, _ = problema.avaliar(peso)
    history = [bestScore]

    for i in range(iterations):
        novoPeso = problema.gerar_vizinho(peso, stepSize)
        newScore, _, _ = problema.avaliar(novoPeso)

        if newScore > bestScore:
            bestScore = newScore
            peso = novoPeso
        history.append(bestScore)

    return peso, bestScore, history


def hillClimbingParallel(problema, runs=8, iterations=10000, stepSize=0.05):
    print(f"Running Hill Climbing ({runs} parallel runs)...")
    results = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(hillClimbingOnce, problema, iterations, stepSize) for _ in range(runs)]
        for idx, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            print(f"  Run {idx}/{runs} complete")

    best = max(results, key=lambda x: x[1])
    return best
