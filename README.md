# Stock Portfolio Optimization with Metaheuristics

## 1. Problem Definition

The goal of this project is to optimize a stock portfolio by choosing weights for a set of assets, such that:

- the **expected return** of the portfolio is maximized, and
- the **risk** (volatility) is minimized.

We follow the classic mean–variance formulation:

- Let $\mu$ be the vector of mean daily returns of each stock.
- Let $\Sigma$ be the covariance matrix of returns.
- Let $w$ be the vector of portfolio weights ($w_i \geq 0$, $\sum w_i = 1$).

Our **score function** is:

$ \text{score}(w) = \mu^T w - \lambda \cdot \sqrt{w^T \Sigma w} $

with $\lambda = 0.5$, meaning we penalize portfolios with higher volatility.

The optimization problem is:

- maximize score(w)
- subject to:
  - $w_i \geq 0$ for all $i$
  - $\sum w_i = 1$

---

## 2. Dataset

- Dataset: *Australian Historical Stock Prices* (Kaggle).
- We use **100 stocks**, each with **343 trading days** of historical prices.
- For each stock, we:
  - read the daily closing prices,
  - align all stocks by date,
  - compute **daily returns** using percentage change,
  - drop the initial NaN row.

The final data matrix has shape (343, 100), where each column is a stock and each row is a trading day.

From this matrix of returns:

- `meanReturns`: vector of mean daily returns (length 100).
- `matrizCov`: 100×100 covariance matrix of returns.

---

## 3. Problem Modelling

We encapsulated the portfolio optimization problem in the class `ProblemaPortfolio`:

- `gerar_estado_inicial()`: generates a random feasible portfolio:
  - sample random non-negative weights,
  - normalize them so that $\sum w_i = 1$.
- `avaliar(pesos)`: computes (score, return, risk):
  - clip weights to be non-negative,
  - renormalize them to sum to 1,
  - compute expected return $\mu^T w$,
  - compute risk as $\sqrt{w^T \Sigma w}$,
  - score = return – $\lambda$ × risk.
- `gerar_vizinho(pesos, stepSize)`: generates a neighbor solution by:
  - applying a small random perturbation to each weight in [−stepSize, +stepSize],
  - clipping negative values to 0,
  - renormalizing to keep $\sum w_i = 1$.

This representation is shared by all metaheuristics: **every algorithm operates on the same search space** (valid portfolios) and uses the same score function.

---

## 4. Metaheuristics Implemented

We implemented four metaheuristics, each in its own module under `algoritmos/`, and a generic wrapper `AgenteOtimizacao` to run them.

### 4.1 Hill Climbing

File: `algoritmos/hillClimbing.py`

- Start from a random feasible portfolio w.
- At each iteration:
  - generate a neighbor w' using `gerar_vizinho`,
  - evaluate its score,
  - if score(w') > score(w), accept w' as the new current solution.
- Keep track of the best score found and save the history.

We also implemented a **parallel** version (`hillClimbingParallel`) that runs multiple independent hill-climbing searches and returns the best result among them.

Main hyperparameters:

- iterations: 10 000
- stepSize: 0.05
- runs (parallel): 8

---

### 4.2 Simulated Annealing

File: `algoritmos/simulatedAnnealing.py`

- Also starts from a random portfolio $w$.
- At each iteration:
  - generate a neighbor $w'$ and compute $\Delta = \text{score}(w') - \text{score}(w)$.
  - If $\Delta > 0$, accept $w'$ (improvement).
  - If $\Delta \leq 0$, accept $w'$ with probability:
    - $\exp(\Delta / T)$, where $T$ is the current temperature.
- The temperature is decreased over time using a geometric schedule:
  - $T \leftarrow T \times \text{coolingRate}$

This allows the algorithm to accept worse solutions at the beginning (exploration) and gradually become more greedy (exploitation).

Main hyperparameters:

- iterations: 10 000
- initialTemp: 1.0
- coolingRate: 0.995
- stepSize: 0.05
- runs (parallel): 8

---

### 4.3 Tabu Search

File: `algoritmos/tabuSearch.py`

- Maintains:
  - a current solution w,
  - a **tabu list** (short-term memory) of recently visited solutions,
  - the global best solution found so far.
- At each iteration:
  - generate a small set of neighbors (e.g. 10),
  - discard neighbors that are too close to solutions in the tabu list,
  - select the neighbor with the best score among the remaining ones,
  - update the current solution and add it to the tabu list,
  - update the global best if needed.

The tabu list blocks immediate revisits to recent points in the search space, helping to avoid cycles and local traps.

Main hyperparameters:

- iterations: 1 000
- stepSize: 0.05
- tabuSize: 30 (length of the short-term memory)
- runs (parallel): 8

---

### 4.4 Genetic Algorithm

File: `algoritmos/geneticAlgorithm.py`

- Representation: each individual is a feasible portfolio (vector of weights).
- Initial population:
  - `pop_size` randomly generated portfolios.
- Evaluation:
  - score of each individual computed with `problema.avaliar`.
- Selection:
  - **tournament selection**:
    - randomly pick two individuals,
    - select the one with higher score as parent.
- Crossover:
  - one-point crossover:
    - choose a random cut point,
    - concatenate prefix of parent 1 with suffix of parent 2,
    - clip to non-negative and renormalize.
- Mutation:
  - for each gene, with probability `mutation_rate`,
    - add a small perturbation in [−stepSize, +stepSize],
    - clip and renormalize.
- Replacement:
  - the new population is composed by the children generated each generation.
- We track the best individual and the evolution of the best score over generations.

Main hyperparameters:

- pop_size: 30
- generations: 200
- crossover_rate: 0.8
- mutation_rate: 0.1
- stepSize: 0.05
- runs (parallel): 8

---

## 5. Experimental Setup

We use the same dataset, score function and constraints for all algorithms. For the main experiment:

- runs per algorithm: 8
- Hyperparameters (typical configuration):

| Algorithm            | Runs | Iterations / Generations | Key Parameters                         |
|----------------------|------|--------------------------|----------------------------------------|
| Hill Climbing        | 8    | 10 000                   | stepSize = 0.05                        |
| Simulated Annealing  | 8    | 10 000                   | T₀ = 1.0, coolingRate = 0.995          |
| Tabu Search          | 8    | 1 000                    | stepSize = 0.05, tabuSize = 30         |
| Genetic Algorithm    | 8    | 200 generations          | pop_size = 30, mutation = 0.1, step=0.05 |

We record, for each algorithm:

- the **best score** obtained across all runs;
- the **execution time** in seconds;
- the **evolution curve** (best score vs iteration/generation) for visualization.

---

## 6. Results

The aggregated results are stored in `results/results.csv` and `results/results.txt`.  
The final scores and times are:

| Algorithm            | Best Score     | Time (s) |
|----------------------|---------------:|--------:|
| Hill Climbing        | -0.0056747     | 14.88   |
| Simulated Annealing  | -0.0058570     | 14.04   |
| Tabu Search          | -0.0057232     | 24.46   |
| Genetic Algorithm    | -0.0056340     | 1.41    |

Note: all scores are negative because the risk term (volatility) is heavily penalizing, but what matters is the **relative comparison** between algorithms.

We also produced:

- `results_summary.png`: bar plots comparing best score per algorithm;
- `results_time.png`: bar plots comparing execution time per algorithm;
- `evolution_curve.png`: line plot with the evolution of the best score along iterations/generations.

---

## 7. Discussion

From the results:

- **Genetic Algorithm** obtained the **best score** (highest, i.e., least negative):
  - it consistently finds portfolios with a better risk–return balance than HC, SA and TS.
  - it is also the **fastest** method in our setup (≈ 1.4 seconds).
- **Hill Climbing** and **Tabu Search** reach slightly worse scores and take longer:
  - Hill Climbing can easily get stuck in local optima and explores less.
  - Tabu Search maintains some memory and can escape small basins of attraction, but each iteration is more expensive (due to neighbor generation and tabu checks).
- **Simulated Annealing** shows a competitive exploration behavior, but in our parameter setting it did not outperform Genetic Algorithm:
  - with a different cooling schedule or number of iterations, it might improve further.

The **evolution curves** show that:

- Hill Climbing typically improves quickly at the beginning and then stagnates.
- Simulated Annealing and Tabu Search keep improving for longer, but with smaller increments.
- Genetic Algorithm tends to improve across generations as selection and crossover propagate good building blocks across the population.

---

## 8. Conclusion and Future Work

In this project we modeled a stock portfolio optimization problem using real market data and applied four metaheuristic algorithms: Hill Climbing, Simulated Annealing, Tabu Search, and Genetic Algorithms.

Under a common experimental setup:

- Genetic Algorithm achieved the best trade-off between return and risk, and did so with the lowest execution time.
- Local-search-based methods (HC, SA, TS) were competitive but generally slower and more sensitive to initialization and parameter tuning.

Possible extensions:

- explore different values of λ (risk aversion), comparing conservative vs aggressive portfolios;
- include constraints such as maximum exposure per sector or per asset;
- experiment with hybrid methods (e.g., GA + local search refinement);
- perform statistical tests over multiple random seeds to assess robustness.

Overall, this work demonstrates how metaheuristics can be effectively used to tackle real-world portfolio optimization problems when the search space is large and gradient-based methods are not straightforward to apply.
