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


def carregar_dados(caminho_arquivo):
    """Carrega e processa dados meteorológicos de um CSV do INMET.
    Detecta encoding automaticamente e mapeia colunas por palavras-chave,
    funcionando com diferentes variações de nomes de coluna.
    """
    caminho = Path(caminho_arquivo)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_arquivo}"
        )

    encoding = detectar_encoding(caminho)

    # O INMET usa ";", mas arquivos salvos pelo Excel podem vir com ","
    with open(caminho, encoding=encoding, errors="replace") as f:
        cabecalho = next(itertools.islice(f, 8, None), "")
    separador = ";" if ";" in cabecalho else ","

    # Padrão INMET: 8 linhas de metadados antes do cabeçalho
    df = pd.read_csv(
        caminho,
        sep=separador,
        skiprows=8,
        encoding=encoding,
        low_memory=False
    )

    # Normalizar nomes das colunas
    df.columns = [normalizar_coluna(c) for c in df.columns]

    # Identificar coluna de data (contém 'DATA')
    coluna_data = None
    coluna_chuva = None
    for col in df.columns:
        if 'DATA' in col:
            coluna_data = col
        if 'PRECIP' in col or 'CHUVA' in col:
            coluna_chuva = col

    if coluna_data is None or coluna_chuva is None:
        raise ValueError(
            "Não encontrei as colunas esperadas no CSV. "
            f"Colunas encontradas (após normalização): {df.columns.tolist()}\n"
            "Procuro por colunas que contenham 'DATA' (data) e 'PRECIP' ou 'CHUVA' (precipitação)."
        )

    # Converte as datas para o formato que o Python entende.
    df[coluna_data] = pd.to_datetime(
        df[coluna_data],
        format="%Y/%m/%d",
        errors="coerce"
    )

    # Converte a precipitação para números.
    # O INMET pode utilizar vírgula como separador decimal.
    df["chuva"] = pd.to_numeric(
        df[coluna_chuva]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

    # Valores negativos ou ausentes não serão considerados válidos.
    df.loc[df["chuva"] < 0, "chuva"] = np.nan
    df = df.dropna(subset=[coluna_data])

    # Agrupa os registros horários por dia.
    diario = df.groupby(
        df[coluna_data].dt.normalize()
    ).agg(
        chuva_total=("chuva", "sum"),
        horas_validas=("chuva", "count")
    )

    # Só utiliza dias com as 24 horas preenchidas.
    diario = diario[diario["horas_validas"] == 24]

    chuva = diario["chuva_total"].sort_index()

    if len(chuva) == 0:
        raise ValueError("Nenhum dia com 24 registros válidos foi encontrado.")

    # Identifica sequências de dias consecutivos.
    grupos = chuva.index.to_series().diff().dt.days.ne(1).cumsum()

    # Escolhe a maior sequência sem interrupções.
    tamanhos = chuva.groupby(grupos).size()
    maior_grupo = tamanhos.idxmax()
    chuva = chuva[grupos == maior_grupo]

    print("\nDados meteorológicos preparados:")
    print(f"Primeiro dia: {chuva.index.min().date()}")
    print(f"Último dia:   {chuva.index.max().date()}")
    print(f"Dias consecutivos completos: {len(chuva)}")

    return chuva


def preparar_dados(chuva):
    valores = chuva.to_numpy(dtype=float)

    X = []
    y = []

    # Utiliza os 3 dias anteriores para prever o dia seguinte.
    for i in range(3, len(valores)):
        dias_anteriores = valores[i - 3:i]

        # 1 = choveu; 0 = não choveu.
        choveu = int(valores[i] > 0)

        X.append(dias_anteriores)
        y.append(choveu)

    return (
        np.array(X, dtype=float),
        np.array(y, dtype=float).reshape(-1, 1)
    )


def main(verbose=True):
    if len(sys.argv) < 2:
        print("Uso: python main.py <caminho_do_arquivo_csv>")
        print("Exemplo: python main.py data/inmet.csv")
        sys.exit(1)

    caminho_arquivo = sys.argv[1]

    chuva = carregar_dados(caminho_arquivo)
    X, y = preparar_dados(chuva)

    if len(X) < 30:
        raise ValueError(
            "Poucos exemplos disponíveis para treinar a rede."
        )

    # Separa os dados em ordem cronológica.
    limite = int(len(X) * 0.8)

    X_treino = X[:limite]
    y_treino = y[:limite]

    X_teste = X[limite:]
    y_teste = y[limite:]

    # Normaliza os valores usando apenas os dados de treinamento.
    media = X_treino.mean()
    desvio = X_treino.std()

    if desvio == 0:
        desvio = 1

    X_treino = (X_treino - media) / desvio
    X_teste = (X_teste - media) / desvio

    # Cria nossa rede neural.
    rede = RedeNeural(
        entradas=3,
        ocultos=4,
        taxa_aprendizado=0.01
    )

    print(f"\nTotal de exemplos: {len(X)}")
    print(f"Exemplos para treinamento: {len(X_treino)}")
    print(f"Exemplos para teste: {len(X_teste)}")

    print("\nIniciando treinamento...")
    rede.treinar(X_treino, y_treino, epocas=2000)

    # Faz previsões com os dados de teste.
    probabilidades = rede.prever(X_teste)

    previsoes = (probabilidades >= 0.5).astype(int)

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