# ============================================
# Autor: Fernando Torres Ferreira da Silva
# Projeto: RecomNLP - Comunicação Aumentativa com IA
# Versão: v1.0
# Arquivo: executar_pipeline.py
# Data: 15/06/2025
# ============================================
import os
from transformers import AutoTokenizer, AutoModel, pipeline
from preprocessamento import (
    baixar_e_preparar_frequencia,
    preparar_vocab_e_trie,
    calcular_embeddings,
    carregar_dispositivo,
    executar_fine_tuning
)

# CONFIGS
#BERTimbau da NeuralMind https://huggingface.co/neuralmind/bert-base-portuguese-cased
BERT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
ARQUIVO_LEXICO = "lexporbr_alfa_txt.txt" 
MAX_PALAVRAS = 30000 
BATCH_SIZE = 32

DATASET_CAA_FILE = "dataset_caa.txt" 
FINE_TUNED_MODEL_PATH = "./bert_finetuned_caa"
EPOCHS_FINETUNE = 3

def main():
    print("INÍCIO DO PIPELINE DE GERAÇÃO DE RECURSOS")

    # 1. Baixando e Preparando a Frequência
    if not os.path.exists("frequencia.pkl"):
        baixar_e_preparar_frequencia(ARQUIVO_LEXICO)
    else:
        print("[PULADO] Frequência já processada.")

    # 2. Carregando o modelo BERT Tokenizer e Modelo
    print("[INFO] Carregando modelo BERT...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)

    # 3. Criando Vocabulário e Trie
    if not os.path.exists("trie.pkl") or not os.path.exists("vocabulario.pkl"):
        vocabulario = preparar_vocab_e_trie(tokenizer, max_palavras=MAX_PALAVRAS)
    else:
        print("[PULADO] Trie e vocabulário já existem.")
        import pickle
        with open("vocabulario.pkl", "rb") as f:
            vocabulario = pickle.load(f)

    # 4. Calculo e Geração dos Embeddings
    if not os.path.exists("embeddings.npy"):
        dispositivo = carregar_dispositivo()
        calcular_embeddings(vocabulario, tokenizer, model, BATCH_SIZE, dispositivo)
    else:
        print("[PULADO] Embeddings já gerados.")

# --- NOVO PASSO 5: Executando o Fine-Tuning ---
    print("\n--- PASSO 5: Verificando e Executando Fine-Tuning ---")
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        if not os.path.exists(DATASET_CAA_FILE):
            print(f"[ERRO] Dataset de fine-tuning '{DATASET_CAA_FILE}' não encontrado. Pulando etapa de fine-tuning.")
        else:
            executar_fine_tuning(
                base_model_name=BERT_MODEL_NAME,
                dataset_file=DATASET_CAA_FILE,
                output_path=FINE_TUNED_MODEL_PATH,
                epochs=EPOCHS_FINETUNE,
                batch_size=BATCH_SIZE
            )
    else:
        print(f"[PULADO] Modelo fine-tuned já existe em '{FINE_TUNED_MODEL_PATH}'.")

    print("PIPELINE CONCLUÍDO")

if __name__ == "__main__":
    main()
