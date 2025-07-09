# ============================================
# Autor: Fernando Torres Ferreira da Silva
# Projeto: RecomNLP - Comunicação Aumentativa com IA
# Versão: v1.0
# Arquivo: api.py
# Data: 15/06/2025
# ============================================
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import uvicorn
import json
import pickle
import os
import numpy as np
from typing import List
from preprocessamento import gerar_sugestoes_gpt2
from fastapi.middleware.cors import CORSMiddleware
import vosk


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINE_TUNED_MODEL_PATH = os.path.join(BASE_DIR, "bert_finetuned_caa")
VOSK_MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-pt-0.3")
TRIE_PATH = os.path.join(BASE_DIR, "trie.pkl")
FREQUENCIA_PATH = os.path.join(BASE_DIR, "frequencia.pkl")
VOCAB_PATH = os.path.join(BASE_DIR, "vocabulario.pkl")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")

#FINE_TUNED_MODEL_PATH = "./bert_finetuned_caa"
#VOSK_MODEL_PATH = "/media/fernando/OS/Cursos/Mestrado/matricula mestrado 2025/Pesquisa/Projeto_Mestrado_v10/vosk-model-small-pt-0.3"
#TRIE_PATH = "trie.pkl" 
#FREQUENCIA_PATH = "frequencia.pkl" 

app = FastAPI(title="API de Recomendação (Fine-Tuned + Trie)", version="3.0")

# --- INÍCIO DA CONFIGURAÇÃO DO CORS ---
origins = [
    "http://localhost:3000",  
    "http://localhost:3001",  # Se a gente for usar outra porta para o React
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


"""
# Carregamento dos recursos
with open("trie.pkl", "rb") as f:
    trie = pickle.load(f)

with open("vocabulario.pkl", "rb") as f:
    vocabulario = pickle.load(f)

with open("frequencia.pkl", "rb") as f:
    frequencia = pickle.load(f)

embeddings = np.load("embeddings.npy")
"""
# --- Carregamento dos Recursos (objetos globais para a API) ---
app.state.sugestor_bert = None
app.state.trie = None
app.state.frequencia = None
app.state.vocabulario = None
app.state.embeddings = None 
app.state.vosk_model = None

@app.on_event("startup")
def carregar_modelos():
    """Carrega os modelos e recursos na inicialização da API."""
    print("[API Startup] Carregando modelos e recursos...")
    try:
        tokenizer_bert = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        model_bert = AutoModelForMaskedLM.from_pretrained(FINE_TUNED_MODEL_PATH)


        app.state.sugestor_bert = pipeline("fill-mask", model=model_bert, tokenizer=tokenizer_bert)
        print("[API Startup] Modelo BERT fine-tuned carregado.")
    except Exception as e:
        print(f"[API Startup ERRO] Falha ao carregar modelo BERT: {e}")

    try:
        with open(TRIE_PATH, "rb") as f:
            app.state.trie = pickle.load(f)
        with open(FREQUENCIA_PATH, "rb") as f:
            app.state.frequencia = pickle.load(f)
        print("[API Startup] Recursos da Trie carregados.")
    except Exception as e:
        print(f"[API Startup ERRO] Falha ao carregar recursos da Trie: {e}")

    try:
        app.state.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        print("[API Startup] Modelo Vosk carregado.")
    except Exception as e:
        print(f"[API Startup ERRO] Falha ao carregar modelo Vosk: {e}")


    try:
        with open(VOCAB_PATH, "rb") as f:
            app.state.vocabulario = pickle.load(f)
        app.state.embeddings = np.load(EMBEDDINGS_PATH)
        print("[API Startup] Vocabulário e Embeddings carregados.")
    except Exception as e:
        print(f"[API Startup ERRO] Falha ao carregar vocab/embeddings: {e}")

# Endpoints

@app.get("/sugestoes_hibrido/", response_model=dict, summary="Sugestões Híbridas (Trie + BERT Fine-Tuned)")
async def sugerir_hibrido(texto: str = Query(..., min_length=1), limite: int = 3):
    """
    Endpoint híbrido: autocompleta palavra atual com Trie ou prevê próxima palavra com BERT.
    """
    # Verifica se os modelos foram carregados corretamente
    if app.state.sugestor_bert is None and app.state.trie is None:
        return {"tipo": "erro", "sugestoes": ["Modelos não carregados"]}

    sugestoes = []
    tipo_sugestao = "nenhum"

    if texto.endswith(' '):
        # MODO BERT: Prever próxima palavra
        if app.state.sugestor_bert:
            try:
                texto_com_mascara = texto.strip() + " " + app.state.sugestor_bert.tokenizer.mask_token
                resultados = app.state.sugestor_bert(texto_com_mascara, top_k=limite)
                sugestoes = [res['token_str'].strip() for res in resultados]
                tipo_sugestao = "proxima_palavra"
            except Exception as e:
                print(f"[API Endpoint ERRO] Falha na predição BERT: {e}")
                
    else:
        # MODO TRIE: Autocompletar palavra atual
        if app.state.trie and app.state.frequencia:
            try:
                palavra_atual = texto.split(' ')[-1].lower()
                if palavra_atual:
                    candidatos = list(app.state.trie.iterkeys(palavra_atual))
                    candidatos_ordenados = sorted(candidatos, key=lambda p: app.state.frequencia.get(p, float('inf')))
                    sugestoes = candidatos_ordenados[:limite]
                    tipo_sugestao = "autocompletar"
            except Exception as e:
                print(f"[API Endpoint ERRO] Falha na busca da Trie: {e}")

    #return sugestoes
    return {"tipo": tipo_sugestao, "sugestoes": sugestoes}

@app.get("/sugestoes/", response_model=list, summary="Sugestões com base no prefixo")
def sugerir(prefixo: str = Query(..., min_length=1), limite: int = 5):
    """Sugere palavras que começam com o prefixo fornecido."""
    if not app.state.trie or not app.state.frequencia:
        return []

    prefixo_lower = prefixo.lower()
    
    if not app.state.trie.has_subtrie(prefixo_lower):
        return []
        # OU, se o frontend já espera o objeto:
        # return {"prefixo": prefixo, "sugestoes": []}

    candidatos = list(app.state.trie.iterkeys(prefixo_lower))
    candidatos_ordenados = sorted(candidatos, key=lambda p: app.state.frequencia.get(p, float('inf')))
    
    return candidatos_ordenados[:limite]

    # OU, se o frontend já espera o objeto:
    #return {"prefixo": prefixo, "sugestoes": candidatos[:limite]}

#@app.get("/sugestoes/", summary="Sugestões com base no prefixo")
#def sugerir(prefixo: str = Query(..., min_length=1), limite: int = 5):
#    """Sugere palavras que começam com o prefixo fornecido."""
#    candidatos = list(trie.iterkeys(prefixo.lower()))
#    candidatos = sorted(candidatos, key=lambda p: frequencia.get(p, 1.0), reverse=True)
#    return {"prefixo": prefixo, "sugestoes": candidatos[:limite]}


@app.get("/embedding/", response_model=dict, summary="Retorna o embedding de uma palavra")
def get_embedding(palavra: str):
    """Retorna o vetor de embedding da palavra, se existir no vocabulário."""
    if not app.state.vocabulario or app.state.embeddings is None:
        return {"erro": "Recursos de embedding não carregados."}

    palavra = palavra.lower()
    try:
        idx = app.state.vocabulario.index(palavra)
        vetor = app.state.embeddings[idx].tolist()
        return {"palavra": palavra, "embedding": vetor}
    except ValueError:
        return {"erro": f"A palavra '{palavra}' não está no vocabulário."}


@app.get("/frequencia/", response_model=dict, summary="Consulta a frequência estimada de uma palavra")
def get_frequencia(palavra: str):
    """Consulta a frequência aproximada da palavra (quanto menor, mais frequente)."""
    if not app.state.frequencia:
        return {"erro": "Recurso de frequência não carregado."}

    palavra = palavra.lower()
    freq = app.state.frequencia.get(palavra)
    if freq is not None:
        return {"palavra": palavra, "frequencia_invertida": freq}
    return {"erro": f"Palavra '{palavra}' não encontrada no léxico."}

@app.get("/sugestoes_gpt2/", response_model=list, summary="Sugestões com GPT-2")
def sugerir_com_gpt2(prefixo: str = Query(..., min_length=1), limite: int = 5):
    sugestoes = gerar_sugestoes_gpt2(prefixo, num_sugestoes=limite)
    #completas = [f"{prefixo}{s}" for s in sugestoes]
    return sugestoes
    # OU, se o frontend espera o objeto:
    # return {"prefixo": prefixo, "sugestoes": completas}

#@app.get("/sugestoes_hibrido/", summary="Sugestões com Trie + GPT-2")
#def endpoint_sugestoes(prefixo: str = Query(..., min_length=1), limite: int = 5):
#    sugestoes = sugerir_hibrido(prefixo, limite)
#    return sugestoes
    # OU, se o frontend espera o objeto:
    # return {"prefixo": prefixo, "sugestoes": sugestoes}

@app.websocket("/ws/sugestoes_por_voz")
async def websocket_endpoint_voz(websocket: WebSocket):
    """
    Endpoint WebSocket para transcrição em tempo real e sugestão.
    - Recebe chunks de áudio em bytes.
    - Transcreve com Vosk.
    - Ao obter uma frase final, gera sugestões com BERT fine-tuned.
    - Envia sugestões de volta para o cliente.
    """
    await websocket.accept()
    
    # Verifica se o modelo Vosk foi carregado
    if not app.state.vosk_model:
        await websocket.send_json({"tipo": "erro", "detalhe": "Modelo de reconhecimento de voz não está disponível."})
        await websocket.close()
        return    

    rec = vosk.KaldiRecognizer(app.state.vosk_model, 16000) # Assumindo 16000Hz
    rec.SetWords(True)

    print("[WebSocket] Cliente conectado para sugestões por voz.")

    try:
        while True:
            # Recebe o áudio em bytes do cliente
            data = await websocket.receive_bytes()

            # Alimenta o reconhecedor Vosk
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get("text"):
                    texto_transcrito = result["text"]
                    print(f"[WebSocket] Transcrição final: '{texto_transcrito}'")
                    
                    # Envia a transcrição de volta para o cliente (para exibir na tela)
                    await websocket.send_json({"tipo": "transcricao_final", "texto": texto_transcrito})

                    # Agora, usa o texto transcrito para gerar sugestões com BERT
                    if app.state.sugestor_bert:
                        try:
                            # Prepara o input para o pipeline 'fill-mask'
                            texto_com_mascara = texto_transcrito.strip() + " " + app.state.sugestor_bert.tokenizer.mask_token
                            resultados_bert = app.state.sugestor_bert(texto_com_mascara, top_k=3)
                            sugestoes = [res['token_str'].strip() for res in resultados_bert]
                            
                            # Envia as sugestões para o cliente
                            await websocket.send_json({"tipo": "sugestao_contextual", "sugestoes": sugestoes})
                            print(f"[WebSocket] Sugestões enviadas: {sugestoes}")

                        except Exception as e:
                            print(f"[WebSocket ERRO] Falha na predição BERT: {e}")
                            await websocket.send_json({"tipo": "erro", "detalhe": "Falha ao gerar sugestões."})

            else:
                # Opcional: Enviar transcrições parciais de volta para o cliente
                partial_result = json.loads(rec.PartialResult())
                if partial_result.get("partial"):
                    await websocket.send_json({"tipo": "transcricao_parcial", "texto": partial_result["partial"]})

    except WebSocketDisconnect:
        print("[WebSocket] Cliente desconectado.")
    except Exception as e:
        print(f"[WebSocket ERRO] Erro inesperado: {e}")
        await websocket.send_json({"tipo": "erro", "detalhe": "Ocorreu um erro no servidor."})
    finally:
        # Garante que o websocket seja fechado se sair do loop
        await websocket.close()


@app.websocket("/ws/transcricao_em_tempo_real")
async def websocket_transcricao(websocket: WebSocket):
    await websocket.accept()
    
    if not app.state.vosk_model:
        await websocket.send_json({"tipo": "erro", "detalhe": "Modelo de voz não disponível."})
        await websocket.close()
        return

    # Cria uma instância do reconhecedor Vosk para esta sessão WebSocket
    # A taxa de amostragem (16000) deve corresponder à do áudio enviado pelo frontend
    rec = vosk.KaldiRecognizer(app.state.vosk_model, 16000)

    print("[WebSocket] Cliente conectado.")
    try:
        # Loop infinito para receber dados de áudio
        while True:
            # Recebe um chunk de áudio em bytes do cliente
            audio_chunk = await websocket.receive_bytes()

            # Processa o chunk de áudio
            if rec.AcceptWaveform(audio_chunk):
                # O usuário fez uma pausa, temos um resultado final
                result = json.loads(rec.Result())
                if result.get("text"):
                    texto_final = result["text"]
                    print(f" Texto Final: {texto_final}")
                    
                    # Envia transcrição para o cliente
                    await websocket.send_json({"tipo": "transcricao_final", "texto": texto_final})
                
                    # Gera e envia sugestões contextuais
                    if app.state.sugestor_bert:
                        texto_com_mascara = texto_final.strip() + " " + app.state.sugestor_bert.tokenizer.mask_token
                        resultados_bert = app.state.sugestor_bert(texto_com_mascara, top_k=3)
                        sugestoes = [res['token_str'].strip() for res in resultados_bert]
                        await websocket.send_json({"tipo": "sugestao_contextual", "sugestoes": sugestoes})
                        print(f" Sugestões: {sugestoes}")

            else:
                # Ainda não há uma frase completa, envia resultado parcial
                partial_result = json.loads(rec.PartialResult())
                if partial_result.get("partial"):
                    await websocket.send_json({"tipo": "transcricao_parcial", "texto": partial_result["partial"]})

    except WebSocketDisconnect:
        print("[WebSocket] Cliente desconectado.")
    except Exception as e:
        print(f"[WebSocket ERRO] Erro: {e}")
    finally:
        # Garante que a conexão seja fechada
        await websocket.close()

# Execução local
if __name__ == "__main__":
    print("Iniciando servidor FastAPI na porta 8000...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 
