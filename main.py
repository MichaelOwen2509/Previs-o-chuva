import itertools
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from rede_neural import RedeNeural


def normalizar_coluna(nome):
    nome = str(nome).strip().upper()
    nome = ''.join(
        c for c in unicodedata.normalize('NFD', nome)
        if unicodedata.category(c) != 'Mn'
    )
    return nome


def detectar_encoding(caminho):
    for enc in ('utf-8', 'latin-1', 'cp1252', 'iso-8859-1'):
        try:
            with open(caminho, 'r', encoding=enc) as f:
                f.read(2000)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 'latin-1'


def _mapear_coluna(df, palavras):
    for col in df.columns:
        for p in palavras:
            if p in col:
                return col
    return None


def _detectar_separador(caminho, encoding):
    with open(caminho, encoding=encoding, errors='replace') as f:
        cabecalho = next(itertools.islice(f, 8, None), "")
    return ";" if ";" in cabecalho else ","


def _limpar_valor(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip()
    if s == '' or s == '-9999' or s == '-9999.0':
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def carregar_dados(caminho_arquivo, colunas_extra=None):
    caminho = Path(caminho_arquivo)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    encoding = detectar_encoding(caminho)
    separador = _detectar_separador(caminho, encoding)

    df = pd.read_csv(
        caminho,
        sep=separador,
        skiprows=8,
        encoding=encoding,
        low_memory=False
    )
    df.columns = [normalizar_coluna(c) for c in df.columns]

    coluna_data = _mapear_coluna(df, ("DATA",))
    coluna_hora = _mapear_coluna(df, ("HORA", "HORARIO"))
    coluna_chuva = _mapear_coluna(df, ("PRECIP", "CHUVA"))

    if coluna_data is None or coluna_chuva is None:
        raise ValueError(
            "Não encontrei as colunas esperadas no CSV. "
            f"Colunas encontradas: {df.columns.tolist()}\n"
            "Procuro por 'DATA' e 'PRECIP' ou 'CHUVA'."
        )

    df[coluna_chuva] = df[coluna_chuva].apply(_limpar_valor)
    df.loc[df[coluna_chuva] < 0, coluna_chuva] = np.nan

    if coluna_hora is not None:
        df['_datetime'] = pd.to_datetime(
            df[coluna_data].astype(str) + ' ' + df[coluna_hora].astype(str),
            errors='coerce', utc=False
        )
    else:
        df['_datetime'] = pd.to_datetime(df[coluna_data], errors='coerce', utc=False)

    # Garante que todos fiquem sem fuso horário (tz-naive)
    if hasattr(df['_datetime'].dt, 'tz') and df['_datetime'].dt.tz is not None:
        df['_datetime'] = df['_datetime'].dt.tz_localize(None)

    df = df.dropna(subset=['_datetime', coluna_chuva])
    df = df.set_index('_datetime').sort_index()

    colunas_extra_encontradas = []
    if colunas_extra:
        for palavra in colunas_extra:
            col = _mapear_coluna(df, (palavra,))
            if col is not None and col not in colunas_extra_encontradas:
                df[col] = df[col].apply(_limpar_valor)
                colunas_extra_encontradas.append(col)

    # Agrega por dia
    agg = {coluna_chuva: (coluna_chuva, 'sum'), 'horas_validas': (coluna_chuva, 'count')}
    for nome_col in colunas_extra_encontradas:
        agg[f'{nome_col}_media'] = (nome_col, 'mean')
        agg[f'{nome_col}_max'] = (nome_col, 'max')
        agg[f'{nome_col}_min'] = (nome_col, 'min')

    diario = df.groupby(df.index.normalize()).agg(**agg)
    diario = diario[diario['horas_validas'] == 24]
    diario = diario.sort_index()

    # Identifica sequências de dias consecutivos
    if len(diario) > 0:
        grupos = diario.index.to_series().diff().dt.days.ne(1).cumsum()
        tamanhos = diario.groupby(grupos).size()
        maior_grupo = tamanhos.idxmax()
        diario = diario[grupos == maior_grupo]

    return diario


def carregar_varias_cidades(caminhos, colunas_extra=None):
    frames = []
    for caminho in caminhos:
        diario = carregar_dados(caminho, colunas_extra=colunas_extra)
        nome_cidade = Path(caminho).stem
        diario = diario.copy()
        diario['cidade'] = nome_cidade
        frames.append(diario)

    if not frames:
        raise ValueError("Nenhum arquivo foi carregado.")

    df = pd.concat(frames, axis=0)
    df = df.sort_index()
    return df


def carregar_todas_cidades_da_pasta(pasta, colunas_extra=None, padrao="*.csv"):
    pasta = Path(pasta)
    if not pasta.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta}")

    # Lista todos os .csv independente de maiúsculas/minúsculas da extensão
    arquivos = sorted(
        arq for arq in pasta.iterdir()
        if arq.is_file() and arq.suffix.lower() == '.csv'
    )

    # Remove duplicatas com o mesmo nome de arquivo (ignorando caixa da extensão)
    arquivos_unicos = []
    nomes_vistos = set()
    for arq in arquivos:
        chave = arq.stem.lower()
        if chave not in nomes_vistos:
            nomes_vistos.add(chave)
            arquivos_unicos.append(str(arq))

    if not arquivos_unicos:
        raise ValueError(f"Nenhum CSV encontrado em: {pasta}")

    return carregar_varias_cidades(arquivos_unicos, colunas_extra=colunas_extra)


def calcular_class_weight(y):
    y = np.asarray(y).reshape(-1)
    n_total = len(y)
    n_chuva = int(np.sum(y == 1))
    n_sem_chuva = int(np.sum(y == 0))

    w1 = n_total / (2 * n_chuva) if n_chuva > 0 else 1.0
    w0 = n_total / (2 * n_sem_chuva) if n_sem_chuva > 0 else 1.0

    return {0: w0, 1: w1}


def encontrar_threshold_otimo(probabilidades, y_true, passo=0.01, t_min=0.50, t_max=0.95):
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


def calcular_auc(probabilidades, y_true):
    """AUC via estatística de Mann-Whitney U (sem depender de sklearn)."""
    probabilidades = np.asarray(probabilidades).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    pos = probabilidades[y_true == 1]
    neg = probabilidades[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    # Conta quantos pares (pos, neg) tem pos > neg
    maior = 0.0
    for p in pos:
        maior += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(maior / (len(pos) * len(neg)))


def calcular_brier(probabilidades, y_true):
    """Brier score: erro quadrático médio das probabilidades."""
    probabilidades = np.asarray(probabilidades).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    return float(np.mean((probabilidades - y_true) ** 2))


def engenharia_features_diaria(df, coluna_chuva=None):
    df = df.copy().sort_index()

    # Detecta coluna de chuva se não informada
    if coluna_chuva is None or coluna_chuva not in df.columns:
        for c in df.columns:
            if 'CHUVA' in c.upper() or 'PRECIP' in c.upper():
                coluna_chuva = c
                break
    if coluna_chuva is None:
        raise ValueError("Nenhuma coluna de chuva/precipitação encontrada no DataFrame.")

    # Segmentos contíguos: quebra quando falta dia (gap entre anos) ou muda a cidade.
    # Rolling/diff ffill são feitos por segmento para não misturar anos distintos.
    quebras = pd.Series(False, index=df.index)
    dif_dias = df.index.to_series().diff().dt.days
    quebras |= dif_dias.fillna(1).ne(1)
    if 'cidade' in df.columns:
        quebras |= df['cidade'] != df['cidade'].shift(1)
    segmento = quebras.cumsum()

    # Preenche valores ausentes apenas dentro do mesmo segmento
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    for c in colunas_numericas:
        df[c] = df.groupby(segmento)[c].ffill().bfill()

    # Chuva total diária
    df['chuva_total'] = df[coluna_chuva]

    # Features de chuva (as mais preditivas)
    g_seg = df.groupby(segmento)
    df['chuva_media_3d'] = g_seg['chuva_total'].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    df['chuva_media_7d'] = g_seg['chuva_total'].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )

    # Sequência de dias consecutivos sem chuva (reinicia a cada segmento)
    df['dia_com_chuva'] = (df['chuva_total'] > 0).astype(int)
    troca = (df['dia_com_chuva'] != df['dia_com_chuva'].shift(1)) | quebras
    grupo = troca.cumsum()
    df['sequencia_dias_sem_chuva'] = df.groupby(grupo).cumcount()
    df.loc[df['dia_com_chuva'] == 1, 'sequencia_dias_sem_chuva'] = 0

    # Sazonalidade (ciclo anual - fundamental para chuva em Brasília)
    meses = df.index.month.values
    dias_do_ano = df.index.day_of_year.values
    df['seno_mes'] = np.sin(2 * np.pi * meses / 12)
    df['cosseno_mes'] = np.cos(2 * np.pi * meses / 12)
    df['seno_dia_ano'] = np.sin(2 * np.pi * dias_do_ano / 365)
    df['cosseno_dia_ano'] = np.cos(2 * np.pi * dias_do_ano / 365)

    # Temperatura e umidade (ajudam a identificar massa de ar)
    col_temp = _mapear_coluna(df, ('TEMPERATURA DO AR',))
    if col_temp is not None:
        df['temp_media'] = df[col_temp]
    col_temp_max = _mapear_coluna(df, ('TEMPERATURA MAXIMA',))
    col_temp_min = _mapear_coluna(df, ('TEMPERATURA MINIMA',))
    if col_temp_max is not None and col_temp_min is not None:
        df['amplitude_termica'] = df[col_temp_max] - df[col_temp_min]

    col_umid = _mapear_coluna(df, ('UMIDADE RELATIVA DO AR',))
    if col_umid is not None:
        df['umidade_media'] = df[col_umid]

    # Pressão (queda de pressão costuma preceder chuva)
    col_pressao = _mapear_coluna(df, ('PRESSAO ATMOSFERICA AO NIVEL',))
    if col_pressao is not None:
        df['pressao_media'] = df[col_pressao]
        df['pressao_variacao'] = df.groupby(segmento)[col_pressao].diff()

    return df


def preparar_dados(df, coluna_chuva=None, janela_dias=3):
    # Observação: espera-se que engenharia_features_diaria já tenha sido aplicada
    # pelo chamador (evita processar as features em duplicidade).
    if coluna_chuva is None:
        for c in df.columns:
            if c == 'chuva_total' or 'CHUVA' in c.upper():
                coluna_chuva = c
                break
    if coluna_chuva is None:
        coluna_chuva = 'chuva_total'

    valores_chuva = df[coluna_chuva].to_numpy(dtype=float)

    colunas_feature = [
        c for c in df.columns
        if c not in (coluna_chuva, 'horas_validas', 'cidade', 'dia_com_chuva')
    ]

    n_features = len(colunas_feature)
    if n_features == 0:
        matriz = valores_chuva.reshape(-1, 1)
        n_features = 1
    else:
        matriz = df[colunas_feature].to_numpy(dtype=float)

    X = []
    y = []
    datas_y = []
    cidades = df['cidade'].to_numpy() if 'cidade' in df.columns else None

    n = len(valores_chuva)
    for i in range(janela_dias, n):
        if cidades is not None and cidades[i] != cidades[i - janela_dias]:
            continue

        janela = matriz[i - janela_dias:i].reshape(-1)
        choveu = int(valores_chuva[i] > 0)

        X.append(janela)
        y.append(choveu)
        datas_y.append(df.index[i])

    return (
        np.array(X, dtype=float),
        np.array(y, dtype=float).reshape(-1, 1),
        colunas_feature,
        np.array(datas_y),
    )


def dividir_temporal(X, y, datas, proporcoes=(0.6, 0.2, 0.2)):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    datas = np.asarray(datas)

    n = len(X)
    a = int(n * proporcoes[0])
    b = a + int(n * proporcoes[1])

    X_treino = X[:a]
    y_treino = y[:a].reshape(-1, 1)
    datas_treino = datas[:a]

    X_val = X[a:b]
    y_val = y[a:b].reshape(-1, 1)
    datas_val = datas[a:b]

    X_teste = X[b:]
    y_teste = y[b:].reshape(-1, 1)
    datas_teste = datas[b:]

    return (X_treino, y_treino, datas_treino,
            X_val, y_val, datas_val,
            X_teste, y_teste, datas_teste)


def normalizar_treino_teste(X_treino, X_val, X_teste):
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0)
    desvio = np.where(desvio == 0, 1, desvio)

    X_treino_n = (X_treino - media) / desvio
    X_val_n = (X_val - media) / desvio
    X_teste_n = (X_teste - media) / desvio

    return X_treino_n, X_val_n, X_teste_n, media, desvio


def treinar_rede_completa(caminho_dados, epocas=8000, janela_dias=3,
                          colunas_extra=None, proporcoes=(0.6, 0.2, 0.2),
                          salvar_modelo='modelo.npz', verbose=True):
    if os.path.isdir(caminho_dados):
        df = carregar_todas_cidades_da_pasta(caminho_dados, colunas_extra=colunas_extra)
    else:
        df = carregar_dados(caminho_dados, colunas_extra=colunas_extra)

    if len(df) < 100:
        raise ValueError(
            f"Poucos dias disponíveis ({len(df)}). São necessários pelo menos 100 dias com dados completos."
        )

    df = engenharia_features_diaria(df)

    X, y, colunas_feature, datas = preparar_dados(df, janela_dias=janela_dias)

    if len(X) < 100:
        raise ValueError(
            f"Poucos exemplos para treinar ({len(X)}). São necessários pelo menos 100 exemplos."
        )

    (X_treino, y_treino, d_treino,
     X_val, y_val, d_val,
     X_teste, y_teste, d_teste) = dividir_temporal(X, y, datas, proporcoes=proporcoes)

    X_treino_n, X_val_n, X_teste_n, media, desvio = normalizar_treino_teste(
        X_treino, X_val, X_teste
    )

    n_entradas = X_treino_n.shape[1]

    rede = RedeNeural(
        entradas=n_entradas,
        ocultos=[32, 16],
        taxa_aprendizado=0.01,
        l2_lambda=0.001
    )

    # Inicializa bias de saída com log-odds da taxa base de chuva
    p_base = float(np.mean(y_treino))
    p_base = np.clip(p_base, 1e-4, 1 - 1e-4)
    rede.biases[-1][0, 0] = np.log(p_base / (1 - p_base))

    class_weight = calcular_class_weight(y_treino)
    # Amortece fortemente o class_weight para evitar colapso para prever tudo como chuva
    class_weight = {k: 1.0 + (v - 1.0) * 0.2 for k, v in class_weight.items()}

    rede.treinar(
        X_treino_n, y_treino,
        epocas=epocas,
        batch_size=128,
        X_val=X_val_n,
        y_val=y_val,
        paciencia=3000,
        decaimento_taxa=0.8,
        class_weight=class_weight,
        verbose=verbose,
        metrica_espera='auc',
        epocas_minimas=2000
    )

    probabilidades_val = rede.prever(X_val_n)
    threshold = encontrar_threshold_otimo(probabilidades_val, y_val)
    # Piso: nunca aceita threshold abaixo de 0.5 (usado igualmente em
    # treino/avaliação e na previsão final, para não haver inconsistência)
    threshold = max(threshold, 0.50)
    rede.threshold_otimo = threshold

    probabilidades_teste = rede.prever(X_teste_n)
    previsoes = (probabilidades_teste >= threshold).astype(int)

    vp = int(np.sum((y_teste == 1) & (previsoes == 1)))
    fp = int(np.sum((y_teste == 0) & (previsoes == 1)))
    fn = int(np.sum((y_teste == 1) & (previsoes == 0)))
    vn = int(np.sum((y_teste == 0) & (previsoes == 0)))

    acuracia = float(np.mean(previsoes == y_teste))
    precisao = vp / (vp + fp) if vp + fp > 0 else 0.0
    recall = vp / (vp + fn) if vp + fn > 0 else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall > 0 else 0.0

    auc = calcular_auc(probabilidades_teste, y_teste)
    brier = calcular_brier(probabilidades_teste, y_teste)

    classe_comum = int(np.mean(y_treino) >= 0.5)
    ac_base = float(np.mean(np.full_like(y_teste, classe_comum) == y_teste))

    resultado = {
        'total_dias': len(df),
        'total_exemplos': len(X),
        'treino': len(X_treino),
        'validacao': len(X_val),
        'teste': len(X_teste),
        'n_entradas': n_entradas,
        'features': colunas_feature,
        'acuracia': round(acuracia, 4),
        'acuracia_base': round(ac_base, 4),
        'melhoria': round(acuracia - ac_base, 4),
        'precisao': round(precisao, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'auc': round(auc, 4),
        'brier': round(brier, 4),
        'chuvas_teste': int(y_teste.sum()),
        'sem_chuva_teste': int(len(y_teste) - y_teste.sum()),
        'previsoes_chuva': int(previsoes.sum()),
        'previsoes_sem_chuva': int(len(previsoes) - previsoes.sum()),
        'threshold': round(threshold, 4),
        'class_weight_chuva': round(float(class_weight[1]), 4),
        'periodo_treino': f"{d_treino[0]} a {d_treino[-1]}",
        'periodo_val': f"{d_val[0]} a {d_val[-1]}",
        'periodo_teste': f"{d_teste[0]} a {d_teste[-1]}",
    }

    rede.media_treino = media
    rede.desvio_treino = desvio
    rede.colunas_feature = colunas_feature
    rede.colunas_extra = colunas_extra
    rede.janela_dias = janela_dias

    if salvar_modelo:
        rede.salvar(salvar_modelo)

    if verbose:
        print("\n" + "=" * 50)
        print("       RESUMO EXECUTIVO")
        print("=" * 50)
        print(f"Período treino: {resultado['periodo_treino']}")
        print(f"Período validação: {resultado['periodo_val']}")
        print(f"Período teste: {resultado['periodo_teste']}")
        print(f"Acurácia da rede neural: {acuracia:.1%}")
        print(f"Acurácia da referência (classe mais comum): {ac_base:.1%}")
        print(f"Melhoria sobre referência: {(acuracia - ac_base):.1%}")
        print(f"F1-score: {f1:.1%}")
        print(f"AUC: {auc:.3f}")
        print(f"Brier score: {brier:.3f}")
        print(f"Precisão: {precisao:.1%}")
        print(f"Recall: {recall:.1%}")
        print()
        print("Matriz de confusão (teste):")
        print(f"  VP (chuva acertada):  {vp}")
        print(f"  FP (falso alarme):    {fp}")
        print(f"  FN (chuva perdida):   {fn}")
        print(f"  VN (sem chuva acert.):{vn}")
        print("=" * 50 + "\n")

    return rede, resultado, (X_teste_n, y_teste, previsoes, probabilidades_teste, d_teste)


def prever_proximo_dia(caminho_modelo='modelo.npz', caminho_dados='data'):
    rede = RedeNeural.carregar(caminho_modelo)

    if os.path.isdir(caminho_dados):
        df = carregar_todas_cidades_da_pasta(caminho_dados, colunas_extra=rede.colunas_extra)
    else:
        df = carregar_dados(caminho_dados, colunas_extra=rede.colunas_extra)

    df = engenharia_features_diaria(df)

    # Usa as mesmas colunas do treino (salvas no modelo)
    colunas_feature = rede.colunas_feature
    janela_dias = rede.janela_dias if rede.janela_dias is not None else 3

    if colunas_feature is None:
        _, _, colunas_feature, _ = preparar_dados(df, janela_dias=janela_dias)
    else:
        # Verifica se todas as colunas existem no df
        colunas_faltantes = [c for c in colunas_feature if c not in df.columns]
        if colunas_faltantes:
            raise ValueError(
                f"Colunas do treino não encontradas nos dados: {colunas_faltantes}"
            )

    matriz = df[colunas_feature].to_numpy(dtype=float)
    datas = df.index

    if len(matriz) < janela_dias:
        raise ValueError(f"Não há dados suficientes (mínimo {janela_dias} dias).")

    # Preenche NaN com forward fill e depois com média da coluna
    matriz_df = pd.DataFrame(matriz, columns=colunas_feature, index=datas)
    matriz_df = matriz_df.ffill().bfill()
    for c in matriz_df.columns:
        if matriz_df[c].isna().any():
            matriz_df[c] = matriz_df[c].fillna(matriz_df[c].mean())

    ultima_data = datas[-1]
    janela = matriz_df.iloc[-janela_dias:].values.reshape(1, -1)

    media = rede.media_treino
    desvio = rede.desvio_treino
    if media is None or desvio is None:
        raise ValueError("O modelo não contém estatísticas de normalização.")

    janela_n = (janela - media) / desvio

    probabilidade = float(rede.prever(janela_n)[0][0])
    threshold = rede.threshold_otimo
    # Piso de seguranca: nao aceita threshold abaixo de 0.50
    threshold = max(threshold, 0.50)
    previsao_chuva = int(probabilidade >= threshold)

    return {
        'ultima_data_disponivel': str(ultima_data.date()),
        'probabilidade_chuva': round(probabilidade, 4),
        'probabilidade_percentual': round(probabilidade * 100, 2),
        'previsao_chuva': previsao_chuva,
        'previsao_texto': 'CHUVA' if previsao_chuva == 1 else 'SEM CHUVA',
        'threshold_usado': round(threshold, 4),
    }


def main(verbose=True):
    if len(sys.argv) < 2:
        print("Uso: python main.py <caminho_csv_ou_pasta> [--prever] [colunas_extra...]")
        print("Exemplo: python main.py data/")
        print("Exemplo com previsão: python main.py data/ --prever")
        sys.exit(1)

    caminho = sys.argv[1]
    modo_prever = False
    colunas_extra = []

    for arg in sys.argv[2:]:
        if arg == '--prever':
            modo_prever = True
        else:
            colunas_extra.append(arg)

    if modo_prever:
        resultado = prever_proximo_dia(
            caminho_modelo='modelo.npz',
            caminho_dados=caminho,
        )
        print("\n" + "=" * 50)
        print("       PREVISÃO PARA O PRÓXIMO DIA")
        print("=" * 50)
        print(f"Última data disponível: {resultado['ultima_data_disponivel']}")
        print(f"Probabilidade de chuva: {resultado['probabilidade_percentual']}%")
        print(f"Previsão: {resultado['previsao_texto']}")
        print(f"Threshold usado: {resultado['threshold_usado']}")
        print("=" * 50 + "\n")
        return

    rede, resultado, _ = treinar_rede_completa(
        caminho_dados=caminho,
        colunas_extra=colunas_extra if colunas_extra else None,
        verbose=verbose
    )

    # Previsão para o dia seguinte após o treino
    try:
        pred = prever_proximo_dia(
            caminho_modelo='modelo.npz',
            caminho_dados=caminho,
        )
        print("\n" + "=" * 50)
        print("       PREVISÃO PARA O PRÓXIMO DIA")
        print("=" * 50)
        print(f"Última data disponível: {pred['ultima_data_disponivel']}")
        print(f"Probabilidade de chuva: {pred['probabilidade_percentual']}%")
        print(f"Previsão: {pred['previsao_texto']}")
        print(f"Threshold usado: {pred['threshold_usado']}")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"\nNão foi possível gerar previsão para o próximo dia: {e}\n")


if __name__ == "__main__":
    modo_verbose = True
    if len(sys.argv) > 2 and sys.argv[-1] == "--quiet":
        modo_verbose = False

    main(verbose=modo_verbose)
