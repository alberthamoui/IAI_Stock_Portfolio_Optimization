import time

class AgenteOtimizacao:
    def __init__(self, problema, metodo):
        self.problema = problema
        self.metodo = metodo  # função ou objeto que implementa o algoritmo

    def executar(self, **kwargs):
        inicio = time.time()
        melhor_estado, melhor_valor = self.metodo(self.problema, **kwargs)
        duracao = time.time() - inicio
        return {
            "melhor_estado": melhor_estado,
            "melhor_valor": melhor_valor,
            "tempo": duracao
        }
