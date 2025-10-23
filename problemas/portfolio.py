import numpy as np

class ProblemaPortfolio:
    """
    Classe que representa o problema de otimização de portfólio.
    O agente tenta encontrar a melhor distribuição de pesos nas ações,
    maximizando o retorno esperado e minimizando o risco.
    """

    def __init__(self, meanReturns, matrizCov, lambda_=0.5):
        self.meanReturns = meanReturns
        self.matrizCov = matrizCov
        self.lambda_ = lambda_
        self.n = len(meanReturns)

    def gerar_estado_inicial(self):
        """Gera um vetor de pesos aleatórios que somam 1."""
        pesos = np.random.random(self.n)
        pesos /= np.sum(pesos)
        return pesos

    def avaliar(self, pesos):
        """Calcula score, retorno e risco para um conjunto de pesos."""
        pesos = np.clip(pesos, 0, None)
        pesos /= np.sum(pesos)
        retorno = np.dot(pesos, self.meanReturns)
        risco = np.sqrt(np.dot(pesos.T, np.dot(self.matrizCov, pesos)))
        score = retorno - self.lambda_ * risco
        return score, retorno, risco

    def gerar_vizinho(self, pesos, stepSize=0.05):
        """Gera uma nova solução a partir da atual (pequena perturbação)."""
        novo = pesos + np.random.uniform(-stepSize, stepSize, self.n)
        novo = np.clip(novo, 0, None)
        novo /= np.sum(novo)
        return novo
