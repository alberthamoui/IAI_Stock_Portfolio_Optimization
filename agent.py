import time

class AgenteOtimizacao:
    def __init__(self, problema, metodo):
        self.problema = problema
        self.metodo = metodo  # função (ex: hillClimbingParallel, etc.)

    def executar(self, **kwargs):
        inicio = time.time()
        melhor_estado, melhor_valor, _ = self.metodo(self.problema, **kwargs)
        duracao = time.time() - inicio
        return {
            "melhor_estado": melhor_estado,
            "melhor_valor": melhor_valor,
            "tempo": duracao
        }