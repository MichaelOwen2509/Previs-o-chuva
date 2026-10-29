import numpy as np


class RedeNeural:
    def __init__(self, entradas=None, ocultos=None, dims=None,
                 taxa_aprendizado=0.01, l2_lambda=0.001, seed=42):
        np.random.seed(seed)

        if dims is not None:
            self.dims = list(dims)
        elif entradas is not None and ocultos is not None:
            if isinstance(ocultos, (list, tuple)):
                self.dims = [int(entradas)] + [int(o) for o in ocultos] + [1]
            else:
                self.dims = [int(entradas), int(ocultos), 1]
        else:
            self.dims = [3, 32, 16, 1]

        self.taxa_inicial = float(taxa_aprendizado)
        self.taxa = float(taxa_aprendizado)
        self.l2_lambda = float(l2_lambda)

        self.pesos = []
        self.biases = []
        for i in range(len(self.dims) - 1):
            w = np.random.randn(self.dims[i], self.dims[i + 1]) * np.sqrt(2.0 / self.dims[i])
            b = np.zeros((1, self.dims[i + 1]))
            self.pesos.append(w)
            self.biases.append(b)

        self.historico_treino = []
        self.historico_val = []
        self.media_treino = None
        self.desvio_treino = None
        self.threshold_otimo = 0.5
        self.colunas_feature = None
        self.colunas_extra = None
        self.janela_dias = None

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    @staticmethod
    def _calcular_auc(probabilidades, y_true):
        """AUC via estatística de Mann-Whitney U."""
        probabilidades = np.asarray(probabilidades).reshape(-1)
        y_true = np.asarray(y_true).reshape(-1)
        pos = probabilidades[y_true == 1]
        neg = probabilidades[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        maior = 0.0
        for p in pos:
            maior += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        return float(maior / (len(pos) * len(neg)))

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        activations = [X]
        zs = []
        a = X
        for i in range(len(self.pesos)):
            z = a @ self.pesos[i] + self.biases[i]
            zs.append(z)
            if i == len(self.pesos) - 1:
                a = self.sigmoid(z)
            else:
                a = self.relu(z)
            activations.append(a)
        return activations, zs

    def treinar(self, X, y, epocas=5000, batch_size=64, X_val=None, y_val=None,
                paciencia=3000, decaimento_taxa=0.8, class_weight=None, verbose=True,
                metrica_espera='val_loss', epocas_minimas=2000):
        quantidade = len(X)
        n_batches = max(1, quantidade // batch_size)

        pesos_amostra = None
        if class_weight is not None:
            w0 = float(class_weight.get(0, 1.0))
            w1 = float(class_weight.get(1, 1.0))
            pesos_amostra = np.where(y.reshape(-1) == 1, w1, w0).reshape(-1, 1)

        maximizar = metrica_espera in ('f1', 'auc')
        melhor_val_perda = float('-inf') if maximizar else float('inf')
        epocas_sem_melhora = 0
        melhores_pesos = [w.copy() for w in self.pesos]
        melhores_biases = [b.copy() for b in self.biases]

        for epoca in range(epocas):
            indices = np.random.permutation(quantidade)
            X_emb = X[indices]
            y_emb = y[indices]
            ps_emb = pesos_amostra[indices] if pesos_amostra is not None else None

            perda_epoch = 0.0

            for b in range(n_batches):
                ini = b * batch_size
                fim = min((b + 1) * batch_size, quantidade)
                Xb = X_emb[ini:fim]
                yb = y_emb[ini:fim]
                psb = ps_emb[ini:fim] if ps_emb is not None else None

                activations, zs = self.forward(Xb)
                previsao = activations[-1]

                erro = previsao - yb
                if psb is not None:
                    erro = erro * psb

                perda_batch = -np.mean(
                    yb * np.log(previsao + 1e-8)
                    + (1 - yb) * np.log(1 - previsao + 1e-8)
                )
                perda_epoch += perda_batch * (len(Xb) / quantidade)

                m = len(Xb)
                grads_pesos = [None] * len(self.pesos)
                grads_biases = [None] * len(self.biases)

                # BCE + sigmoide: gradiente da saida eh apenas (p - y)
                delta = erro
                grads_pesos[-1] = (activations[-2].T @ delta) / m + self.l2_lambda * self.pesos[-1]
                grads_biases[-1] = np.mean(delta, axis=0, keepdims=True)

                for l in range(len(self.pesos) - 2, -1, -1):
                    delta = (delta @ self.pesos[l + 1].T) * self.relu_deriv(zs[l])
                    grads_pesos[l] = (activations[l].T @ delta) / m + self.l2_lambda * self.pesos[l]
                    grads_biases[l] = np.mean(delta, axis=0, keepdims=True)

                for l in range(len(self.pesos)):
                    self.pesos[l] -= self.taxa * grads_pesos[l]
                    self.biases[l] -= self.taxa * grads_biases[l]

            if (epoca + 1) % 1000 == 0:
                self.taxa *= decaimento_taxa

            if X_val is not None and y_val is not None:
                val_previsao = self.prever(X_val)
                y_v = y_val.reshape(-1)
                p_v = val_previsao.reshape(-1)

                val_perda = -np.mean(
                    y_v * np.log(p_v + 1e-8)
                    + (1 - y_v) * np.log(1 - p_v + 1e-8)
                )

                # Determina se houve melhora conforme a metrica escolhida
                if metrica_espera == 'auc':
                    metrica_val = self._calcular_auc(p_v, y_v)
                    melhor = metrica_val > melhor_val_perda
                elif metrica_espera == 'f1':
                    previsoes_v = (p_v >= 0.5).astype(int)
                    vp = int(np.sum((y_v == 1) & (previsoes_v == 1)))
                    fp = int(np.sum((y_v == 0) & (previsoes_v == 1)))
                    fn = int(np.sum((y_v == 1) & (previsoes_v == 0)))
                    precisao_v = vp / (vp + fp) if vp + fp > 0 else 0.0
                    recall_v = vp / (vp + fn) if vp + fn > 0 else 0.0
                    metrica_val = (
                        2 * precisao_v * recall_v / (precisao_v + recall_v)
                        if precisao_v + recall_v > 0 else 0.0
                    )
                    melhor = metrica_val > melhor_val_perda
                else:  # val_loss (minimizar)
                    metrica_val = val_perda
                    melhor = val_perda < melhor_val_perda

                self.historico_treino.append(perda_epoch)
                self.historico_val.append(val_perda)

                if verbose and (epoca + 1) % 500 == 0:
                    print(f"Época {epoca + 1}: treino = {perda_epoch:.4f}, val = {val_perda:.4f}, "
                          f"{metrica_espera} = {metrica_val:.4f}, taxa = {self.taxa:.6f}")

                # Só considera early stopping depois de epocas_minimas
                ja_passou_minimas = (epoca + 1) >= epocas_minimas

                if melhor:
                    melhor_val_perda = metrica_val
                    epocas_sem_melhora = 0
                    melhores_pesos = [w.copy() for w in self.pesos]
                    melhores_biases = [b.copy() for b in self.biases]
                elif ja_passou_minimas:
                    epocas_sem_melhora += 1
                    if epocas_sem_melhora >= paciencia:
                        if verbose:
                            print(f"Early stopping na época {epoca + 1} ({metrica_espera} não melhora há {paciencia} épocas)")
                        self.pesos = melhores_pesos
                        self.biases = melhores_biases
                        break
            else:
                self.historico_treino.append(perda_epoch)
                if verbose and (epoca + 1) % 500 == 0:
                    print(f"Época {epoca + 1}: treino = {perda_epoch:.4f}, taxa = {self.taxa:.6f}")

        # Restaura os melhores pesos se houver validacao (caso nao tenha disparado early stopping)
        if X_val is not None and y_val is not None:
            self.pesos = melhores_pesos
            self.biases = melhores_biases

        return self

    def prever(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def salvar(self, caminho):
        dados = {}
        for i, w in enumerate(self.pesos):
            dados[f'pesos_{i}'] = w
        for i, b in enumerate(self.biases):
            dados[f'biases_{i}'] = b
        if self.media_treino is not None:
            dados['media_treino'] = self.media_treino
        if self.desvio_treino is not None:
            dados['desvio_treino'] = self.desvio_treino
        dados['dims'] = np.array(self.dims, dtype=int)
        dados['threshold_otimo'] = np.array([self.threshold_otimo], dtype=float)
        if self.colunas_feature is not None:
            dados['colunas_feature'] = np.array(self.colunas_feature, dtype=object)
        if self.colunas_extra is not None:
            dados['colunas_extra'] = np.array(self.colunas_extra, dtype=object)
        if self.janela_dias is not None:
            dados['janela_dias'] = np.array([self.janela_dias], dtype=int)
        np.savez(caminho, **dados)

    @classmethod
    def carregar(cls, caminho):
        dados = np.load(caminho, allow_pickle=True)
        dims = [int(x) for x in dados['dims']]
        rede = cls(dims=dims)
        n_camadas = len(dims) - 1
        for i in range(n_camadas):
            rede.pesos[i] = dados[f'pesos_{i}']
            rede.biases[i] = dados[f'biases_{i}']
        if 'media_treino' in dados:
            rede.media_treino = dados['media_treino']
        if 'desvio_treino' in dados:
            rede.desvio_treino = dados['desvio_treino']
        if 'threshold_otimo' in dados:
            rede.threshold_otimo = float(dados['threshold_otimo'][0])
        if 'colunas_feature' in dados:
            rede.colunas_feature = [str(x) for x in dados['colunas_feature']]
        if 'colunas_extra' in dados:
            rede.colunas_extra = [str(x) for x in dados['colunas_extra']]
        if 'janela_dias' in dados:
            rede.janela_dias = int(dados['janela_dias'][0])
        return rede
