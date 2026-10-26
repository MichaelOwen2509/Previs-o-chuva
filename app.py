import os
import numpy as np
from flask import Flask, render_template, request, jsonify

from main import carregar_dados, preparar_dados
from rede_neural import RedeNeural


app = Flask(__name__)
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

def carregar_e_processar(caminho_arquivo):
    """Reutiliza a lógica de processamento do main.py"""
    chuva = carregar_dados(caminho_arquivo)
    X, y = preparar_dados(chuva)

    if len(X) < 30:
        raise ValueError(f"Poucos exemplos disponíveis para treinar a rede. São necessários pelo menos 30 dias com dados completos. CSV tem apenas {len(X)} exemplos.")

    limite = int(len(X) * 0.8)
    X_treino = X[:limite]
    X_teste = X[limite:]
    y_treino = y[:limite]
    y_teste = y[limite:]
    media = X_treino.mean()
    desvio = X_treino.std()
    if desvio == 0: desvio = 1
    X_treino_n = (X_treino - media) / desvio
    X_teste_n = (X_teste - media) / desvio

    rede = RedeNeural(entradas=3, ocultos=4, taxa_aprendizado=0.01)
    rede.treinar(X_treino_n, y_treino, epocas=2000)
    probabilidades = rede.prever(X_teste_n)
    previsoes = (probabilidades >= 0.5).astype(int)

    vp = int(np.sum((y_teste == 1) & (previsoes == 1)))
    fp = int(np.sum((y_teste == 0) & (previsoes == 1)))
    fn = int(np.sum((y_teste == 1) & (previsoes == 0)))
    prec = vp / (vp + fp) if vp + fp > 0 else 0
    recall = vp / (vp + fn) if vp + fn > 0 else 0
    acuracia = float(np.mean(previsoes == y_teste))
    classe_comum = int(np.mean(y_treino) >= 0.5)
    ac_base_score = float(np.mean(np.full_like(y_teste, classe_comum) == y_teste))

    return {
        "arquivo": os.path.basename(caminho_arquivo),
        "total_exemplos": len(X),
        "treino": len(X_treino),
        "teste": len(X_teste),
        "acuracia": round(acuracia, 4),
        "acuracia_base": round(ac_base_score, 4),
        "melhoria": round(acuracia - ac_base_score, 4),
        "precisao": round(prec, 4),
        "recall": round(recall, 4),
        "chuvas_teste": int(y_teste.sum()),
        "sem_chuva_teste": int(len(y_teste) - y_teste.sum()),
        "previsoes_chuva": int(previsoes.sum()),
        "previsoes_sem_chuva": int(len(previsoes) - previsoes.sum()),
        "prob_min": round(float(probabilidades.min()), 4),
        "prob_max": round(float(probabilidades.max()), 4),
    }

@app.route("/")
def index():
    arquivos = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv"))
    return render_template("index.html", arquivos=arquivos, titulo="Previsão de Chuva - Rede Neural")

@app.route("/analisar", methods=["POST"])
def analisar():
    try:
        action = request.form.get("action")
        if action == "upload":
            arquivo = request.files.get("csv_file")
            if not arquivo or arquivo.filename == "":
                return jsonify({"tipo": "erro", "mensagem": "Nenhum arquivo enviado. Por favor, selecione um arquivo .csv."}), 400
            nome_seguro = "".join(c for c in arquivo.filename if c.isalnum() or c in "._-").lstrip(".")
            if not nome_seguro.endswith(".csv") or nome_seguro == ".csv":
                return jsonify({"tipo": "erro", "mensagem": "Formato inválido. Por favor, envie apenas arquivos .csv."}), 400
            caminho = os.path.join(DATA_FOLDER, nome_seguro)
            arquivo.save(caminho)
            return jsonify({"tipo": "sucesso", "mensagem": "Arquivo salvo com sucesso: " + nome_seguro, "arquivo": nome_seguro})

        elif action == "selecionar":
            nome = request.form.get("arquivo", "")
            if not nome:
                return jsonify({"tipo": "erro", "mensagem": "Nome do arquivo não informado."}), 400
            if "/" in nome or "\\" in nome:
                return jsonify({"tipo": "erro", "mensagem": "Nome de arquivo inválido."}), 400
            caminho = os.path.join(DATA_FOLDER, nome)
            if not os.path.exists(caminho):
                return jsonify({"tipo": "erro", "mensagem": "Arquivo não encontrado. Ele pode ter sido excluído."}), 404
            try:
                resultado = carregar_e_processar(caminho)
                return jsonify({"tipo": "sucesso", "mensagem": "Análise concluída: " + nome, "resultado": resultado})
            except ValueError as ve:
                return jsonify({"tipo": "erro", "mensagem": str(ve)}), 400
            except Exception as e:
                return jsonify({"tipo": "erro", "mensagem": "Erro interno ao processar o arquivo: " + str(e)}), 500

        elif action == "excluir":
            nome = request.form.get("arquivo", "")
            if not nome:
                return jsonify({"tipo": "erro", "mensagem": "Nome do arquivo não informado."}), 400
            if "/" in nome or "\\" in nome:
                return jsonify({"tipo": "erro", "mensagem": "Nome inválido."}), 400
            caminho = os.path.join(DATA_FOLDER, nome)
            if not os.path.exists(caminho):
                return jsonify({"tipo": "erro", "mensagem": "Arquivo não encontrado."}), 404
            try:
                os.remove(caminho)
                return jsonify({"tipo": "sucesso", "mensagem": "Arquivo excluído: " + nome})
            except Exception as e:
                return jsonify({"tipo": "erro", "mensagem": "Erro ao excluir arquivo: " + str(e)}), 500

        return jsonify({"tipo": "erro", "mensagem": "Ação desconhecida recebida."}), 400

    except Exception as e:
        return jsonify({"tipo": "erro", "mensagem": "Erro inesperado no servidor: " + str(e)}), 500

@app.route("/resultado")
def resultado():
    return jsonify({"mensagem": "Acesse a página principal para iniciar uma nova análise."})


if __name__ == "__main__":
    # 5000 é reservada pelo Windows nesta máquina
    app.run(debug=True, port=8000)
