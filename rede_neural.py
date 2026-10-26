import numpy as np


class RedeNeural:
    def __init__(self, entradas=3, ocultos=4, taxa_aprendizado=0.01):
        np.random.seed(42)

        self.pesos1 = np.random.randn(entradas, ocultos) * 0.5
        self.bias1 = np.zeros((1, ocultos))

        self.pesos2 = np.random.randn(ocultos, 1) * 0.5
        self.bias2 = np.zeros((1, 1))

        self.taxa = taxa_aprendizado

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def treinar(self, X, y, epocas=2000):
        quantidade = len(X)

        for epoca in range(epocas):
            # Propagação para frente
            z1 = X @ self.pesos1 + self.bias1
            a1 = np.tanh(z1)

            z2 = a1 @ self.pesos2 + self.bias2
            previsao = self.sigmoid(z2)

            # Erro entre previsão e resultado real
            erro = previsao - y

            # Retropropagação: ajuste dos parâmetros
            grad_pesos2 = a1.T @ erro / quantidade
            grad_bias2 = np.mean(erro, axis=0, keepdims=True)

            erro_oculto = (erro @ self.pesos2.T) * (1 - a1**2)
            grad_pesos1 = X.T @ erro_oculto / quantidade
            grad_bias1 = np.mean(erro_oculto, axis=0, keepdims=True)

            self.pesos2 -= self.taxa * grad_pesos2
            self.bias2 -= self.taxa * grad_bias2
            self.pesos1 -= self.taxa * grad_pesos1
            self.bias1 -= self.taxa * grad_bias1

            if (epoca + 1) % 500 == 0:
                perda = -np.mean(
                    y * np.log(previsao + 1e-8)
                    + (1 - y) * np.log(1 - previsao + 1e-8)
                )
                print(f"Época {epoca + 1}: erro = {perda:.4f}")

    def prever(self, X):
        camada_oculta = np.tanh(X @ self.pesos1 + self.bias1)
        saida = self.sigmoid(camada_oculta @ self.pesos2 + self.bias2)
        return saida
