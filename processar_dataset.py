import json
from typing import Dict, List

def processar_dataset_caa(caminho_arquivo: str) -> Dict[str, List[str]]:
    """
    Lê um dataset no formato JSON Lines, extrai, trata e separa as frases
    em português e inglês.

    Args:
        caminho_arquivo (str): O caminho para o arquivo .json do dataset.

    Returns:
        Dict[str, List[str]]: Um dicionário contendo duas chaves: 'portugues' e 'ingles',
                              cada uma com uma lista de frases únicas.
    """
    print(f"Iniciando o processamento do arquivo: {caminho_arquivo}")
    
    # Usamos sets para armazenar as frases e garantir a unicidade de forma eficiente
    frases_pt_unicas = set()
    frases_en_unicas = set()

    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            for i, linha in enumerate(f):
                # Ignora linhas em branco que podem existir no arquivo
                if not linha.strip():
                    continue
                
                try:
                    # Carrega cada linha como um objeto JSON separado
                    data = json.loads(linha)
                    
                    # Extrai e trata a frase em português
                    texto_pt = data.get('text_pt', '').strip()
                    if texto_pt:  # Adiciona apenas se não estiver vazio
                        frases_pt_unicas.add(texto_pt)

                    # Extrai e trata a frase em inglês
                    texto_en = data.get('text', '').strip()
                    if texto_en:
                        frases_en_unicas.add(texto_en)
                        
                except json.JSONDecodeError:
                    print(f"AVISO: Linha {i+1} não é um JSON válido e será ignorada.")
    
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return {'portugues': [], 'ingles': []}
    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado ao ler o arquivo: {e}")
        return {'portugues': [], 'ingles': []}

    print("Processamento concluído com sucesso.")
    # Converte os sets de volta para listas antes de retornar
    return {
        'portugues': sorted(list(frases_pt_unicas)),
        'ingles': sorted(list(frases_en_unicas))
    }

def salvar_vocabulario_para_recomendador(lista_frases: List[str], caminho_saida: str):
    """
    Salva uma lista de frases em um arquivo de texto, com uma frase por linha.
    Este formato é ideal para ser usado como um vocabulário pelo recomendador.

    Args:
        lista_frases (List[str]): A lista de frases a ser salva.
        caminho_saida (str): O nome do arquivo .txt de saída.
    """
    try:
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            for frase in lista_frases:
                f.write(f"{frase}\n")
        print(f"Vocabulário salvo com sucesso em '{caminho_saida}'.")
    except Exception as e:
        print(f"ERRO: Falha ao salvar o arquivo de vocabulário: {e}")

# --- BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # Define o nome do arquivo de entrada e de saída
    ARQUIVO_DATASET = r"C:\Mestrado_Projetos\Pesquisa\Projeto_Mestrado_v10\dataset_pt\aactext_test_pt_br.json"
    ARQUIVO_SAIDA_VOCAB = "vocabulario_pt_br.txt"

    # 1. Processa o dataset para extrair as frases
    dados_processados = processar_dataset_caa(ARQUIVO_DATASET)
    
    # Pega a lista de frases em português
    frases_em_portugues = dados_processados['portugues']

    if frases_em_portugues:
        print("-" * 50)
        print(f"Total de {len(frases_em_portugues)} frases únicas em português encontradas.")
        
        # Exibe as 5 primeiras frases como exemplo
        print("Amostra das frases:")
        for frase in frases_em_portugues[:5]:
            print(f"  - {frase}")
        print("-" * 50)

        # 2. Salva as frases em português no formato de vocabulário para o recomendador
        salvar_vocabulario_para_recomendador(frases_em_portugues, ARQUIVO_SAIDA_VOCAB)

        print("\nPróximos passos:")
        print(f"1. A lista de frases está disponível na variável 'frases_em_portugues' para popular a Trie em memória.")
        print(f"2. O arquivo '{ARQUIVO_SAIDA_VOCAB}' foi criado e pode ser usado como um vocabulário externo.")
    else:
        print("Nenhuma frase em português foi encontrada no dataset.")