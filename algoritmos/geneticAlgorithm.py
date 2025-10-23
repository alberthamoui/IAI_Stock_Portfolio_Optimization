from utils import *

def criar_individuo(problema):
    """Gera um vetor de pesos válido (um indivíduo da população)."""
    individuo = problema.gerar_estado_inicial()
    return individuo


def crossover(pai1, pai2):
    """Cruzamento de um ponto entre dois indivíduos."""
    n = len(pai1)
    ponto = random.randint(1, n - 2)
    filho = np.concatenate((pai1[:ponto], pai2[ponto:]))
    filho = np.clip(filho, 0, None)
    filho /= np.sum(filho)
    return filho


def mutacao(individuo, taxa_mutacao=0.1, stepSize=0.05):
    """Aplica uma pequena perturbação em parte dos genes."""
    novo = individuo.copy()
    for i in range(len(novo)):
        if random.random() < taxa_mutacao:
            novo[i] += random.uniform(-stepSize, stepSize)
    novo = np.clip(novo, 0, None)
    novo /= np.sum(novo)
    return novo


def geneticAlgorithmOnce(problema, pop_size=30, generations=200, crossover_rate=0.8, mutation_rate=0.1, stepSize=0.05):
    """
    Executa uma única instância do Algoritmo Genético.
    """
    # População inicial
    populacao = [criar_individuo(problema) for _ in range(pop_size)]
    scores = [problema.avaliar(ind)[0] for ind in populacao]
    best_idx = np.argmax(scores)
    best = populacao[best_idx]
    bestScore = scores[best_idx]
    history = [bestScore]

    for gen in range(generations):
        nova_pop = []

        # Seleção por torneio
        for _ in range(pop_size // 2):
            i, j = random.sample(range(pop_size), 2)
            pai1 = populacao[i] if scores[i] > scores[j] else populacao[j]

            i, j = random.sample(range(pop_size), 2)
            pai2 = populacao[i] if scores[i] > scores[j] else populacao[j]

            # Cruzamento
            if random.random() < crossover_rate:
                filho1 = crossover(pai1, pai2)
                filho2 = crossover(pai2, pai1)
            else:
                filho1, filho2 = pai1.copy(), pai2.copy()

            # Mutação
            filho1 = mutacao(filho1, mutation_rate, stepSize)
            filho2 = mutacao(filho2, mutation_rate, stepSize)
            nova_pop += [filho1, filho2]

        # Avaliar nova geração
        populacao = nova_pop
        scores = [problema.avaliar(ind)[0] for ind in populacao]
        gen_best_idx = np.argmax(scores)
        gen_best = populacao[gen_best_idx]
        gen_bestScore = scores[gen_best_idx]

        if gen_bestScore > bestScore:
            bestScore = gen_bestScore
            best = gen_best

        history.append(bestScore)

    return best, bestScore, history


def geneticAlgorithm(problema, runs=8, pop_size=30, generations=200, crossover_rate=0.8, mutation_rate=0.1, stepSize=0.05):
    """
    Executa várias instâncias do Algoritmo Genético em paralelo.
    """
    print(f"Running Genetic Algorithm ({runs} parallel runs)...")
    results = []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(geneticAlgorithmOnce, problema, pop_size, generations, crossover_rate, mutation_rate, stepSize)
            for _ in range(runs)
        ]
        for idx, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            print(f"  Run {idx}/{runs} complete")

    best = max(results, key=lambda x: x[1])
    return best
