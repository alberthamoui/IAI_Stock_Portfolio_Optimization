from utils import *

def tabuSearchOnce(problema, iterations=10000, stepSize=0.05, tabuSize=50):
    """
    Executa uma única busca Tabu (memória de curto prazo) com barra de progresso (%).
    """
    peso = problema.gerar_estado_inicial()
    bestPeso = peso.copy()
    bestScore, _, _ = problema.avaliar(peso)
    currentScore = bestScore
    tabuList = deque(maxlen=tabuSize)
    history = [bestScore]

    print("  Progress: ", end="")
    last_percent = -1

    for i in range(iterations):
        # ==== algoritmo principal ====
        vizinhos = [problema.gerar_vizinho(peso, stepSize) for _ in range(10)]
        avaliacoes = [(p, problema.avaliar(p)[0]) for p in vizinhos if not any(np.allclose(p, t) for t in tabuList)]

        if not avaliacoes:
            continue

        melhorVizinho, melhorScore = max(avaliacoes, key=lambda x: x[1])
        tabuList.append(melhorVizinho)

        if melhorScore > bestScore:
            bestPeso = melhorVizinho
            bestScore = melhorScore

        peso = melhorVizinho
        currentScore = melhorScore
        history.append(bestScore)

        # ==== progress bar update ====
        percent = int(100 * (i + 1) / iterations)
        if percent != last_percent:  # print only on change
            bar = "=" * (percent // 2) + "-" * ((100 - percent) // 2)
            sys.stdout.write(f"\r  [{bar}] {percent:3d}%")
            sys.stdout.flush()
            last_percent = percent

    print("\r  [==================================================] 100% ✅")
    return bestPeso, bestScore, history


def tabuSearchParallel(problema, runs=8, iterations=10000, stepSize=0.05, tabuSize=50):
    """
    Executa várias buscas Tabu em paralelo e retorna o melhor resultado.
    """
    print(f"Running Tabu Search ({runs} parallel runs)...")
    results = []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(tabuSearchOnce, problema, iterations, stepSize, tabuSize)
            for _ in range(runs)
        ]
        for idx, f in enumerate(as_completed(futures), 1):
            results.append(f.result())
            print(f"  Run {idx}/{runs} complete")

    best = max(results, key=lambda x: x[1])
    return best
