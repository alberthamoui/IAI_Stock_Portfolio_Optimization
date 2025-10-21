from utils import *

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

