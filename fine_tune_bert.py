# ============================================
# Autor: Fernando Torres Ferreira da Silva
# Projeto: RecomNLP - Comunicação Aumentativa com IA
# Versão: v1.0
# Arquivo: fine_tune_bert.py
# Data: 06/07/2025
# ============================================
#fine_tune_bert.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import os

# --- 1. CONFIGURAÇÕES ---
BERT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
DATASET_FILE = "dataset_caa.txt"  
FINE_TUNED_MODEL_PATH = "./bert_finetuned_caa" 

# --- 2. CARREGAR MODELO E TOKENIZER ---
print("Carregando modelo e tokenizer...")

model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

# --- 3. CARREGAR E PREPARAR O DATASET ---
print(f"Carregando dataset de '{DATASET_FILE}'...")

dataset = load_dataset("text", data_files={"train": DATASET_FILE})

# Função para tokenizar o dataset
def tokenize_function(examples):
    
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

print("Tokenizando o dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# --- 4. CONFIGURAR O TREINAMENTO ---
print("Configurando o treinamento...")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15 # 15% das palavras serão mascaradas, que é o padrão do BERT
)

# Argumentos do treinamento
training_args = TrainingArguments(
    output_dir=FINE_TUNED_MODEL_PATH,   # Diretório de saída
    overwrite_output_dir=True,          # Sobrescreve se já existir
    num_train_epochs=3,                 # Número de vezes para passar pelo dataset. 3-5 é um bom começo.
    per_device_train_batch_size=8,      # Número de exemplos por lote. Ajuste de acordo com a memória da sua GPU.
    save_steps=10_000,                  # Salva um checkpoint a cada X passos
    save_total_limit=2,                 # Mantém apenas os 2 últimos checkpoints
    prediction_loss_only=True,          # Acelera o cálculo ao não fazer predições durante o treinamento
    logging_steps=500,                  # Mostra o log de perda a cada 500 passos
)

# O Trainer abstrai todo o loop de treinamento
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    
)

# --- 5. INICIAR O FINE-TUNING ---
print("Iniciando o fine-tuning...")
trainer.train()

# --- 6. SALVAR O MODELO FINAL ---
print("Fine-tuning concluído. Salvando o modelo final...")
trainer.save_model(FINE_TUNED_MODEL_PATH)
tokenizer.save_pretrained(FINE_TUNED_MODEL_PATH) # Salva o tokenizer junto

print(f"Modelo salvo em '{FINE_TUNED_MODEL_PATH}'")
