from utils import *


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

