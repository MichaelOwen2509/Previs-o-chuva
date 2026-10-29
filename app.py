import os
import numpy as np
import math
from pathlib import Path
from flask import Flask, render_template, request, jsonify

from main import (
    carregar_todas_cidades_da_pasta,
    carregar_dados,
    treinar_rede_completa,
    prever_proximo_dia,
)
from rede_neural import RedeNeural


def _sanitize_json(obj):
    """Substitui NaN e Inf por None para tornar o dicionário serializável em JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    return obj


app = Flask(__name__)
DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_FOLDER, exist_ok=True)


@app.route("/")
def index():
    arquivos = sorted(f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".csv"))
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
            if not nome_seguro.lower().endswith(".csv") or nome_seguro == ".csv":
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
                # Treina sempre com TODOS os CSVs da pasta data/ (arquivo único tem
                # dados poucos e forte deslocamento sazonal treino/teste, o que
                # invertia as previsões). O arquivo selecionado apenas dispara a análise.
                arquivos_treino = [
                    f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".csv")
                ]
                if not arquivos_treino:
                    return jsonify({"tipo": "erro", "mensagem": "Nenhum arquivo CSV encontrado na pasta data/."}), 400
                _, resultado, _ = treinar_rede_completa(
                    caminho_dados=DATA_FOLDER,
                    salvar_modelo='modelo.npz',
                    verbose=False
                )
                return jsonify({
                    "tipo": "sucesso",
                    "mensagem": (
                        "Análise concluída (treinada com todos os "
                        f"{len(arquivos_treino)} arquivos de data/)."
                    ),
                    "resultado": _sanitize_json(resultado),
                })
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


@app.route("/prever", methods=["POST", "GET"])
def prever():
    try:
        # O modelo é treinado com todos os CSVs de data/, então a previsão do
        # próximo dia também usa a pasta inteira (última data disponível).
        resultado = prever_proximo_dia(
            caminho_modelo='modelo.npz',
            caminho_dados=DATA_FOLDER,
        )
        return jsonify({"tipo": "sucesso", "mensagem": "Previsão gerada", "resultado": _sanitize_json(resultado)})
    except FileNotFoundError as fe:
        return jsonify({"tipo": "erro", "mensagem": str(fe)}), 404
    except ValueError as ve:
        return jsonify({"tipo": "erro", "mensagem": str(ve)}), 400
    except Exception as e:
        return jsonify({"tipo": "erro", "mensagem": "Erro interno ao gerar previsão: " + str(e)}), 500


@app.route("/resultado")
def resultado():
    return jsonify({"mensagem": "Acesse a página principal para iniciar uma nova análise."})


if __name__ == "__main__":
    app.run(debug=True, port=8000)
