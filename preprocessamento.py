# ============================================
# Autor: Fernando Torres Ferreira da Silva
# Projeto: RecomNLP - Comunicação Aumentativa com IA
# Versão: v1.0
# Arquivo: preprocessamento.py
# Data: 15/06/2025
# https://wandb.ai/fernandotfs-usp-universidade-de-s-o-paulo
# https://github.com/fertorresfs/vox_recom_nlp
# 
# ============================================
import os
import pickle
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer,
    AutoModelForMaskedLM, DataCollatorForLanguageModeling,
    TrainingArguments, Trainer
)
from pygtrie import CharTrie
from datasets import load_dataset

def carregar_dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# --- FUNÇÃO NOVA PARA LER O DATASET CAA ---
def carregar_frases_caa(caminho_arquivo_json: str) -> set:
    """
    Lê um dataset no formato JSON Lines e extrai as frases únicas em português.
    """
    print(f"[INFO] Carregando frases do dataset CAA de '{caminho_arquivo_json}'...")
    frases_pt_unicas = set()
    try:
        with open(caminho_arquivo_json, 'r', encoding='utf-8') as f:
            for i, linha in enumerate(f):
                if not linha.strip():
                    continue
                try:
                    data = json.loads(linha)
                    texto_pt = data.get('text_pt', '').strip()
                    if texto_pt:
                        frases_pt_unicas.add(texto_pt)
                except json.JSONDecodeError:
                    print(f"AVISO: Linha {i+1} do dataset CAA não é um JSON válido e será ignorada.")
    except FileNotFoundError:
        print(f"AVISO: Arquivo do dataset CAA não encontrado em '{caminho_arquivo_json}'.")
    
    print(f"[OK] {len(frases_pt_unicas)} frases únicas carregadas do dataset CAA.")
    return frases_pt_unicas

# --- MODIFICADO para retornar o dicionário e ser mais flexível ---
def baixar_e_preparar_frequencia(arquivo_lexico, arquivo_saida="frequencia.pkl"):
    """Processa o léxico de frequência e retorna um dicionário."""
    print("[INFO] Processando o léxico de frequência...")
    freq_dict = {}

    if not os.path.exists(arquivo_lexico):
        print(f"[ERRO] Arquivo de léxico '{arquivo_lexico}' não encontrado.")
        return freq_dict # Retorna dicionário vazio

    try:
        with open(arquivo_lexico, 'r', encoding='latin1', errors='ignore') as f:
            lexico_raw = f.readlines()

        for i, linha in enumerate(lexico_raw[1:], start=2): # Pula cabeçalho
            linha = linha.strip()
            if not linha:
                continue
            
            colunas = linha.split('\t')
            if len(colunas) > 0:
                palavra = colunas[0].strip().lower()
                if palavra.isalpha() and palavra not in freq_dict:
                    # Usando a posição no arquivo como score (menor é mais frequente)
                    freq_dict[palavra] = i

        print(f"[OK] {len(freq_dict)} palavras processadas do léxico.")
    except Exception as e:
        print(f"[ALERTA] Falha ao processar léxico: {e}")
    
    # Salvar ainda é uma opção, mas o retorno é o principal
    with open(arquivo_saida, "wb") as f_out:
        pickle.dump(freq_dict, f_out)
    print(f"[SALVO] Frequência salva em '{arquivo_saida}'.")
    return freq_dict
    
"""
def baixar_e_preparar_frequencia(arquivo_lexico, arquivo_saida="frequencia.pkl"):
    ""#"Processa o léxico de frequência de palavras e salva em pickle.""#"
    print("[INFO] Processando o léxico de frequência...")
    freq_dict = {}

    if not os.path.exists(arquivo_lexico):
        print(f"[ERRO] Arquivo '{arquivo_lexico}' não encontrado.")
        with open(arquivo_saida, "wb") as f_out:
            pickle.dump(freq_dict, f_out)
        return

    try:
        with open(arquivo_lexico, 'r', encoding='latin1', errors='ignore') as f: #utf-8
            lexico_raw = f.readlines() # f.read().splitlines()

        # Ignorar cabeçalho (1ª linha)
        for i, linha in enumerate(lexico_raw[1:], start=2):
            linha = linha.strip()
            if not linha:
                continue  # pula linhas vazias

            colunas = linha.split('\t')  # separa pelas TABs

            if len(colunas) > 0:
                palavra = colunas[0].strip().lower()  # coluna "ortografia"

                if palavra.isalpha() and palavra not in freq_dict:
                    freq_dict[palavra] = i
                    #freq_dict[palavra] = 1.0 / (len(freq_dict) + 1)

        #for line in lexico_raw[1:]:
        #    parts = line.split().split('\t')
        #    if len(parts) > 0:
        #        palavra = parts[0].strip().lower()
        #        if palavra.isalpha() and palavra not in freq_dict:  
        #            freq_dict[palavra] = 1.0 / (len(freq_dict) + 1)
                    # tambem é possivel usar a frequencia real assim:
                    #frequencia_real = float(parts[3].replace(',', '.'))
                    #freq_dict[palavra] = frequencia_real

        #for line in lexico_raw[1:]:
        #    parts = line.split()
        #    if len(parts) > 1 and parts[1].isalpha():
        #        palavra = parts[1].lower()
        #        freq_dict[palavra] = 1.0 / (len(freq_dict) + 1)

        print(f"[OK] {len(freq_dict)} palavras processadas no léxico.")
    except Exception as e:
        print(f"[ALERTA] Falha ao processar léxico: {e}")
    finally:
        with open(arquivo_saida, "wb") as f_out:
            pickle.dump(freq_dict, f_out)
        print(f"[SALVO] Frequência salva em '{arquivo_saida}'.")
"""

def preparar_vocab_e_trie(tokenizer, max_palavras=30000, arquivo_vocab="vocabulario.pkl", arquivo_trie="trie.pkl"):
    vocabulario = [
        palavra for palavra in tokenizer.get_vocab().keys()
        if palavra.isalpha() and len(palavra) > 2
    ][:max_palavras]

    trie = CharTrie()
    for i, palavra in enumerate(vocabulario):
        trie[palavra.lower()] = i

    with open(arquivo_vocab, "wb") as f:
        pickle.dump(vocabulario, f)

    with open(arquivo_trie, "wb") as f:
        pickle.dump(trie, f)

    print(f"[SALVO] Trie salva em '{arquivo_trie}', vocabulário em '{arquivo_vocab}'.")

    return vocabulario


def calcular_embeddings(vocabulario, tokenizer, model, batch_size, dispositivo, arquivo_saida="embeddings.npy"):
    embeddings = []
    model.to(dispositivo)
    model.eval()

    with torch.no_grad():
        for i in range(0, len(vocabulario), batch_size):
            batch = vocabulario[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, return_tensors="pt").to(dispositivo)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)

            if (i + batch_size) % 500 == 0 or i == 0:
                print(f"[PROCESSO] {i + batch_size}/{len(vocabulario)} palavras processadas...")

    embeddings_matrix = np.vstack(embeddings)
    np.save(arquivo_saida, embeddings_matrix)
    print(f"[SALVO] Embeddings salvos em '{arquivo_saida}'.")


def gerar_sugestoes_gpt2(prefixo, num_sugestoes=5, max_tokens=10):
    model_name = "pierreguillou/gpt2-small-portuguese"  
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    input_text = prefixo
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Gerar saídas
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + max_tokens,
            num_return_sequences=num_sugestoes,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    sugestoes = []
    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        proximo = decoded[len(prefixo):].strip().split()[0]  # pega a próxima palavra
        if proximo.isalpha():  # evita pontuação ou números
            sugestoes.append(proximo)

    return list(set(sugestoes))[:num_sugestoes]


#def sugerir_hibrido(prefixo: str, limite: int = 5):
#    trie = CharTrie()
#    frequencia = {}

#    prefixo = prefixo.lower()

    # Primeiro tenta no Trie
#    try:
#        candidatos = list(trie.iterkeys(prefixo))
#        candidatos = sorted(candidatos, key=lambda p: frequencia.get(p, 1.0), reverse=True)
#    except KeyError:
#        candidatos = []

    # Se poucos resultados ou nenhum, chama GPT-2
#    if len(candidatos) < limite:
#        novas = gerar_sugestoes_gpt2(prefixo, num_sugestoes=limite * 2)
#        novas_completas = [prefixo + s for s in novas if len(s) > 1 and s not in candidatos]

        # Atualiza o Trie com novas palavras geradas
#        for palavra in novas_completas:
#            trie[palavra] = True
#            frequencia[palavra] = frequencia.get(palavra, 1.0) + 1  # leve incremento

#        candidatos.extend(novas_completas)

#    return sorted(set(candidatos))[:limite]


# --- FUNÇÃO MESTRA NOVA ---
def executar_preprocessamento_completo(
    caminho_modelo: str,
    caminho_lexico: str,
    caminho_dataset_caa: str,
    caminho_saida_trie="trie.pkl",
    caminho_saida_frequencia="frequencia.pkl"
):
    """
    Orquestra todo o processo de pré-processamento, combinando o vocabulário
    do tokenizer com as frases do dataset CAA para criar os artefatos finais.
    """
    print("\n--- INICIANDO PRÉ-PROCESSAMENTO COMPLETO E UNIFICADO ---")

    # 1. Carregar as fontes de vocabulário
    print("\n[PASSO 1/4] Carregando fontes de vocabulário...")
    tokenizer = AutoTokenizer.from_pretrained(caminho_modelo)
    vocab_tokenizer = set(tokenizer.get_vocab().keys())
    frases_caa = carregar_frases_caa(caminho_dataset_caa)

    # 2. Combinar vocabulários
    print("\n[PASSO 2/4] Combinando vocabulários...")
    vocabulario_combinado = vocab_tokenizer.union(frases_caa)
    
    # Filtro opcional para remover itens muito curtos ou não-alfabéticos
    vocabulario_filtrado = {
        item for item in vocabulario_combinado if any(c.isalpha() for c in item) and len(item) > 1
    }
    print(f"  - Vocabulário do Tokenizer: {len(vocab_tokenizer)} itens")
    print(f"  - Frases do Dataset CAA: {len(frases_caa)} itens")
    print(f"  - Vocabulário Final Combinado e Filtrado: {len(vocabulario_filtrado)} itens")

    # 3. Construir Trie e Mapa de Frequência
    print("\n[PASSO 3/4] Construindo Trie e Mapa de Frequência...")
    frequencia_base = baixar_e_preparar_frequencia(caminho_lexico, arquivo_saida=caminho_saida_frequencia)
    
    # Damos prioridade máxima (score mais baixo) para as frases do CAA
    for frase in frases_caa:
        if frase.lower() not in frequencia_base:
            frequencia_base[frase.lower()] = 0 # 0 é o score de maior prioridade

    trie = CharTrie()
    for item in vocabulario_filtrado:
        # Chave minúscula para busca, valor com a capitalização original
        trie[item.lower()] = item
    
    # 4. Salvar artefatos finais
    print("\n[PASSO 4/4] Salvando artefatos finais...")
    with open(caminho_saida_trie, "wb") as f:
        pickle.dump(trie, f)
    print(f"[OK] Trie combinada salva em '{caminho_saida_trie}'.")

    with open(caminho_saida_frequencia, "wb") as f:
        pickle.dump(frequencia_base, f)
    print(f"[OK] Mapa de frequência combinado salvo em '{caminho_saida_frequencia}'.")
    
    print("\n--- PRÉ-PROCESSAMENTO COMPLETO FINALIZADO ---")

def executar_fine_tuning(base_model_name, dataset_file, output_path, epochs=3, batch_size=8):
    """
    Executa o processo de fine-tuning de um modelo BERT na tarefa de Masked Language Modeling (MLM).
    """
    print(f"[INFO] Iniciando fine-tuning do modelo '{base_model_name}'.")

    # 1. Carregar modelo e tokenizer base
    print("[INFO] Carregando modelo e tokenizer base...")
    model = AutoModelForMaskedLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 2. Carregar e preparar o dataset
    print(f"[INFO] Carregando dataset de '{dataset_file}'...")
    dataset = load_dataset("text", data_files={"train": dataset_file})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("[INFO] Tokenizando o dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 3. Configurar o treinamento
    print("[INFO] Configurando o DataCollator e os Argumentos de Treinamento...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )

    # 4. Iniciar o treinamento
    print("[INFO] Treinamento iniciado...")
    trainer.train()

    # 5. Salvar o modelo final
    print("[INFO] Treinamento concluído. Salvando o modelo final...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[OK] Modelo fine-tuned salvo em '{output_path}'.")