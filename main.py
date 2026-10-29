import itertools
import numpy as np
import pandas as pd
import sys
import unicodedata
from pathlib import Path
from rede_neural import RedeNeural


def normalizar_coluna(nome):
    """Normaliza um nome de coluna: remove acentos, maiúsculo, strip."""
    nome = str(nome).strip().upper()
    nome = ''.join(
        c for c in unicodedata.normalize('NFD', nome)
        if unicodedata.category(c) != 'Mn'
    )
    return nome


def detectar_encoding(caminho):
    """Tenta várias codificações e retorna a primeira que funcionar."""
    for enc in ('utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
        try:
            with open(caminho, 'r', encoding=enc) as f:
                f.read(2000)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 'latin-1'


def _mapear_coluna(df, palavras):
    """Encontra a primeira coluna que contenha alguma das palavras-chave."""
    for col in df.columns:
        for p in palavras:
            if p in col:
                return col
    return None


def _agregar_diario(df, coluna_data, coluna_chuva, colunas_extra=None):
    """Agrega registros horários por dia, calculando media/soma das features."""
    df = df.copy()
    df[coluna_data] = pd.to_datetime(df[coluna_data], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=[coluna_data])

    df["chuva"] = pd.to_numeric(
        df[coluna_chuva].astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df.loc[df["chuva"] < 0, "chuva"] = np.nan

    agg = {"chuva": ("chuva", "sum"), "horas_validas": ("chuva", "count")}

    if colunas_extra:
        for nome_col in colunas_extra:
            df[nome_col] = pd.to_numeric(
                df[nome_col].astype(str).str.strip().str.replace(",", ".", regex=False),
                errors="coerce"
            )
            agg[f"{nome_col}_media"] = (nome_col, "mean")
            agg[f"{nome_col}_max"] = (nome_col, "max")
            agg[f"{nome_col}_min"] = (nome_col, "min")

    diario = df.groupby(df[coluna_data].dt.normalize()).agg(**agg)
    diario = diario[diario["horas_validas"] == 24]
    diario = diario.sort_index()
    return diario


def carregar_dados(caminho_arquivo, colunas_extra=None):
    """Carrega e processa dados meteorológicos de um CSV do INMET.
    Retorna um DataFrame indexado por data com chuva_total e features opcionais.
    Mantido para compatibilidade com app.py.
    """
    caminho = Path(caminho_arquivo)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    encoding = detectar_encoding(caminho)

    with open(caminho, encoding=encoding, errors="replace") as f:
        cabecalho = next(itertools.islice(f, 8, None), "")
    separador = ";" if ";" in cabecalho else ","

    df = pd.read_csv(
        caminho,
        sep=separador,
        skiprows=8,
        encoding=encoding,
        low_memory=False
    )
    df.columns = [normalizar_coluna(c) for c in df.columns]

    coluna_data = _mapear_coluna(df, ("DATA",))
    coluna_chuva = _mapear_coluna(df, ("PRECIP", "CHUVA"))

    if coluna_data is None or coluna_chuva is None:
        raise ValueError(
            "Não encontrei as colunas esperadas no CSV. "
            f"Colunas encontradas (após normalização): {df.columns.tolist()}\n"
            "Procuro por colunas que contenham 'DATA' (data) e 'PRECIP' ou 'CHUVA' (precipitação)."
        )

    if colunas_extra is None:
        colunas_extra = []
    colunas_extra_encontradas = []
    for palavra in colunas_extra:
        col = _mapear_coluna(df, (palavra,))
        if col is not None and col not in colunas_extra_encontradas:
            colunas_extra_encontradas.append(col)

    diario = _agregar_diario(df, coluna_data, coluna_chuva, colunas_extra_encontradas)

    if len(diario) == 0:
        raise ValueError("Nenhum dia com 24 registros válidos foi encontrado.")

    # Identifica sequências de dias consecutivos.
    grupos = diario.index.to_series().diff().dt.days.ne(1).cumsum()
    tamanhos = diario.groupby(grupos).size()
    maior_grupo = tamanhos.idxmax()
    diario = diario[grupos == maior_grupo]

    print("\nDados meteorológicos preparados:")
    print(f"Primeiro dia: {diario.index.min().date()}")
    print(f"Último dia:   {diario.index.max().date()}")
    print(f"Dias consecutivos completos: {len(diario)}")

    return diario


def carregar_varias_cidades(caminhos, colunas_extra=None):
    """Carrega e agrega dados de múltiplos CSVs (várias cidades).

    Cada arquivo vira uma 'cidade' identificada pelo nome do arquivo.
    Os DataFrames são concatenados e ordenados por data.
    """
    frames = []
    for caminho in caminhos:
        diario = carregar_dados(caminho, colunas_extra=colunas_extra)
        nome_cidade = Path(caminho).stem
        diario = diario.copy()
        diario["cidade"] = nome_cidade
        frames.append(diario)

    if not frames:
        raise ValueError("Nenhum arquivo foi carregado.")

    df = pd.concat(frames, axis=0)
    df = df.sort_index()
    return df


def calcular_class_weight(y):
    """Calcula pesos inversamente proporcionais à frequência de cada classe."""
    y = np.asarray(y).reshape(-1)
    n_total = len(y)
    n_chuva = int(np.sum(y == 1))
    n_sem_chuva = int(np.sum(y == 0))

    w1 = n_total / (2 * n_chuva) if n_chuva > 0 else 1.0
    w0 = n_total / (2 * n_sem_chuva) if n_sem_chuva > 0 else 1.0

    return {0: w0, 1: w1}


def encontrar_threshold_otimo(probabilidades, y_true, passo=0.01, t_min=0.15, t_max=0.65):
    """Escolhe o threshold que maximiza o F1-score para a classe chuva.

    A busca fica restrita a [t_min, t_max] para evitar thresholds
    excessivamente conservadores (que nunca preveem chuva).
    """
    y_true = np.asarray(y_true).reshape(-1)
    probabilidades = np.asarray(probabilidades).reshape(-1)

    melhor_threshold = (t_min + t_max) / 2
    melhor_f1 = -1.0

    for t in np.arange(t_min, t_max + passo, passo):
        previsoes_t = (probabilidades >= t).astype(int)

        vp = int(np.sum((y_true == 1) & (previsoes_t == 1)))
        fp = int(np.sum((y_true == 0) & (previsoes_t == 1)))
        fn = int(np.sum((y_true == 1) & (previsoes_t == 0)))

        precisao_t = vp / (vp + fp) if vp + fp > 0 else 0.0
        recall_t = vp / (vp + fn) if vp + fn > 0 else 0.0

        f1 = (
            2 * precisao_t * recall_t / (precisao_t + recall_t)
            if precisao_t + recall_t > 0 else 0.0
        )

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_threshold = float(t)

    return melhor_threshold


def dividir_treino_teste(X, y, proporcao_treino=0.8):
    """Divide X e y garantindo que ambas as classes apareçam em treino e teste.

    Faz split estratificado simples: separa índices de cada classe e divide
    cada um na proporção informada, evitando que o teste fique sem exemplos
    da classe minoritária (chuva).
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    idx_chuva = np.where(y == 1)[0]
    idx_sem_chuva = np.where(y == 0)[0]

    n_chuva_treino = max(1, int(len(idx_chuva) * proporcao_treino))
    n_sem_treino = max(1, int(len(idx_sem_chuva) * proporcao_treino))

    # Garante que reste pelo menos 1 exemplo de cada classe para o teste.
    n_chuva_treino = min(n_chuva_treino, len(idx_chuva) - 1) if len(idx_chuva) > 1 else len(idx_chuva)
    n_sem_treino = min(n_sem_treino, len(idx_sem_chuva) - 1) if len(idx_sem_chuva) > 1 else len(idx_sem_chuva)

    idx_treino = np.concatenate([idx_chuva[:n_chuva_treino], idx_sem_chuva[:n_sem_treino]])
    idx_teste = np.concatenate([idx_chuva[n_chuva_treino:], idx_sem_chuva[n_sem_treino:]])

    # Embaralha dentro de cada conjunto para não deixar todas as chamas juntas.
    np.random.seed(42)
    np.random.shuffle(idx_treino)
    np.random.shuffle(idx_teste)

    return X[idx_treino], y[idx_treino].reshape(-1, 1), X[idx_teste], y[idx_teste].reshape(-1, 1)


def preparar_dados(df, coluna_chuva=None, janela_dias=3):
    """Converte um DataFrame diário em X (features dos N dias anteriores) e y (chuva hoje).

    Se o DataFrame tiver coluna 'cidade', cada cidade é tratada como sequência
    independente (não mistura o último dia de uma cidade com o primeiro da outra).

    X tem formato (amostras, janela_dias * n_features).
    """
    df = df.sort_index()

    if coluna_chuva is None:
        # Detecta automaticamente a coluna de chuva agregada (case-insensitive).
        for c in df.columns:
            if "CHUVA" in c.upper():
                coluna_chuva = c
                break
    if coluna_chuva is None:
        raise ValueError("Nenhuma coluna de chuva encontrada no DataFrame.")

    valores_chuva = df[coluna_chuva].to_numpy(dtype=float)

    # Colunas de features (tudo que não seja a coluna de chuva, horas_validas ou cidade)
    colunas_feature = [
        c for c in df.columns
        if c not in (coluna_chuva, "horas_validas", "cidade")
    ]

    n_features = len(colunas_feature)
    if n_features == 0:
        # Fallback: apenas chuva dos dias anteriores (comportamento original).
        matriz = valores_chuva.reshape(-1, 1)
        n_features = 1
    else:
        matriz = df[colunas_feature].to_numpy(dtype=float)

    X = []
    y = []
    cidades = df["cidade"].to_numpy() if "cidade" in df.columns else None

    n = len(valores_chuva)
    for i in range(janela_dias, n):
        # Não ultrapassa fronteira de cidade (se houver).
        if cidades is not None and cidades[i] != cidades[i - janela_dias]:
            continue

        janela = matriz[i - janela_dias:i].reshape(-1)  # achata janela_dias x n_features
        choveu = int(valores_chuva[i] > 0)

        X.append(janela)
        y.append(choveu)

    return (
        np.array(X, dtype=float),
        np.array(y, dtype=float).reshape(-1, 1),
        colunas_feature,
    )


def main(verbose=True):
    if len(sys.argv) < 2:
        print("Uso: python main.py <caminho_csv_ou_pasta> [colunas_extra...]")
        print("Exemplo: python main.py data/")
        print("Exemplo com features: python main.py data/ TEMPERATURA UMIDADE PRESSAO")
        sys.exit(1)

    caminho = sys.argv[1]
    colunas_extra = sys.argv[2:] if len(sys.argv) > 2 else []

    caminho = Path(caminho)

    if caminho.is_dir():
        csvs = sorted(caminho.glob("*.csv"))
        if not csvs:
            raise ValueError(f"Nenhum CSV encontrado na pasta: {caminho}")
        caminhos = [str(c) for c in csvs]
        print(f"\nCarregando {len(caminhos)} arquivo(s) CSV de: {caminho}")
        df = carregar_varias_cidades(caminhos, colunas_extra=colunas_extra)
    elif caminho.is_file():
        print(f"\nCarregando 1 arquivo CSV: {caminho}")
        df = carregar_dados(caminho, colunas_extra=colunas_extra)
    else:
        raise FileNotFoundError(f"Caminho não encontrado: {caminho}")

    X, y, colunas_feature = preparar_dados(df)

    n_entradas = X.shape[1]
    print(f"Features utilizadas ({len(colunas_feature)}): {colunas_feature}")
    print(f"Entradas da rede (janela x features): {n_entradas}")

    if len(X) < 30:
        raise ValueError(
            "Poucos exemplos disponíveis para treinar a rede."
        )

    # Separa treino/teste garantindo representação de ambas as classes.
    X_treino, y_treino, X_teste, y_teste = dividir_treino_teste(X, y, proporcao_treino=0.8)

    # Normaliza por coluna usando apenas os dados de treinamento.
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0)
    desvio = np.where(desvio == 0, 1, desvio)

    X_treino = (X_treino - media) / desvio
    X_teste = (X_teste - media) / desvio

    # Cria nossa rede neural com o número correto de entradas.
    rede = RedeNeural(
        entradas=n_entradas,
        ocultos=8,
        taxa_aprendizado=0.01
    )

    print(f"\nTotal de exemplos: {len(X)}")
    print(f"Exemplos para treinamento: {len(X_treino)}")
    print(f"Exemplos para teste: {len(X_teste)}")

    # Peso das classes para combater desbalanceamento (chuva é raridade).
    class_weight = calcular_class_weight(y_treino)
    print(f"Pesos das classes (treino): sem chuva={class_weight[0]:.3f}, chuva={class_weight[1]:.3f}")

    print("\nIniciando treinamento...")
    rede.treinar(X_treino, y_treino, epocas=5000, class_weight=class_weight)

    # Faz previsões com os dados de teste.
    probabilidades = rede.prever(X_teste)

    # Calibra o threshold para maximizar F1 na classe chuva (treino).
    probabilidades_treino = rede.prever(X_treino)
    threshold = encontrar_threshold_otimo(probabilidades_treino, y_treino)
    print(f"Threshold calibrado (F1 no treino): {threshold:.3f}")

    previsoes = (probabilidades >= threshold).astype(int)

    acuracia = np.mean(previsoes == y_teste)

    # Compara com uma estratégia simples de referência.
    classe_comum = int(np.mean(y_treino) >= 0.5)
    previsao_base = np.full_like(y_teste, classe_comum)
    acuracia_base = np.mean(previsao_base == y_teste)

    # Métricas detalhadas
    verdadeiros_positivos = int(np.sum((y_teste == 1) & (previsoes == 1)))
    falsos_positivos = int(np.sum((y_teste == 0) & (previsoes == 1)))
    falsos_negativos = int(np.sum((y_teste == 1) & (previsoes == 0)))
    verdadeiros_negativos = int(np.sum((y_teste == 0) & (previsoes == 0)))

    precisao = (
        verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
        if verdadeiros_positivos + falsos_positivos > 0 else 0
    )

    recall = (
        verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)
        if verdadeiros_positivos + falsos_negativos > 0 else 0
    )

    # --- RESUMO EXECUTIVO ---
    print("\n" + "="*50)
    print("       RESUMO EXECUTIVO")
    print("="*50)
    print(f"Acurácia da rede neural: {acuracia:.1%}")
    print(f"Acurácia da referência (classe mais comum): {acuracia_base:.1%}")
    print(f"Melhoria sobre referência: {(acuracia - acuracia_base):.1%}")
    print()
    print("Métricas de classificação:")
    print(f"  Precisão (quando diz chuva, acerta?): {precisao:.1%}")
    print(f"   Recall (dias chuvosos identificados): {recall:.1%}")
    print()
    print("Distribuição de classes no teste:")
    print(f"  Dias com chuva no teste:      {int(y_teste.sum())}")
    print(f"  Dias sem chuva no teste:      {int(len(y_teste) - y_teste.sum())}")
    print(f"  Previsões de chuva:           {int(previsoes.sum())}")
    print(f"  Previsões sem chuva:          {int(len(previsoes) - previsoes.sum())}")
    print("="*50 + "\n")

    # --- MÉTRICAS DETALHADAS DE CHUVA ---
    print("===== METRICAS DE CHUVA =====")
    print(f"Acertos de chuva:                      {verdadeiros_positivos}")
    print(f"Alarmes falsos de chuva:               {falsos_positivos}")
    print(f"Dias chuvosos nao identificados:       {falsos_negativos}")
    print(f"Acertos de dias sem chuva:             {verdadeiros_negativos}")
    print(f"Precisao:                              {precisao:.1%}")
    print(f"Recall:                                {recall:.1%}")

    # --- ÚLTIMOS EXEMPLOS DO TESTE (apenas se verbose=True) ---
    if verbose:
        print("\nÚltimos exemplos do teste:")

        inicio = max(0, len(y_teste) - 10)

        for i in range(inicio, len(y_teste)):
            real = "Chuva" if y_teste[i, 0] == 1 else "Sem chuva"
            previsto = "Chuva" if previsoes[i, 0] == 1 else "Sem chuva"
            probabilidade = probabilidades[i, 0] * 100

            print(
                f"Real: {real:10} | "
                f"Previsto: {previsto:10} | "
                f"Probabilidade estimada de chuva: {probabilidade:.1f}%"
            )


if __name__ == "__main__":
    # Permite passar --quiet para suprimir a tabela de exemplos
    modo_verbose = True
    if len(sys.argv) > 2 and sys.argv[2] == "--quiet":
        modo_verbose = False
    
    main(verbose=modo_verbose)