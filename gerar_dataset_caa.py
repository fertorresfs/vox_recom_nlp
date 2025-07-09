# ============================================
# Autor: Fernando Torres Ferreira da Silva
# Projeto: RecomNLP - Comunicação Aumentativa com IA
# Versão: v1.0
# Arquivo: gerar_dataset_caa.py
# Data: 06/07/2025
# ============================================
import random
from itertools import product
from tqdm import tqdm

# Listas de palavras para preencher os templates
acoes = [
    "quero", "gostaria de", "preciso", "posso", "estou com vontade de", "não quero"
]

objetos = [
    "comer", "beber", "dormir", "brincar", "assistir televisão", "ir ao banheiro",
    "tomar banho", "escovar os dentes", "ficar em silêncio", "usar o tablet",
    "falar com você", "me sentar", "me levantar", "deitar um pouco", "sair", "voltar para casa"
]

alimentos = [
    "água", "suco", "leite", "pizza", "arroz", "fruta", "sorvete", "biscoito", "pão"
]

sentimentos = [
    "feliz", "triste", "com medo", "cansado", "nervoso", "entediado", "sozinho", "calmo", "animado"
]

lugares = [
    "quarto", "sala", "cozinha", "banheiro", "escola", "parque", "consultório", "cama", "sofá"
]

pessoas = [
    "mamãe", "papai", "professora", "amigo", "médico", "terapeuta", "irmão", "colega"
]

# Templates com slots
templates = [
    "{acao} {objeto}.",
    "{acao} {objeto} com {pessoa}.",
    "estou me sentindo {sentimento}.",
    "quero ir para o {lugar}.",
    "não gosto de {alimento}.",
    "gosto de {alimento}.",
    "quero ver a {pessoa}.",
    "posso ir ao {lugar}?",
    "você pode me ajudar a {objeto}?",
    "estou com fome de {alimento}.",
]

# Função para preencher os templates
def gerar_frases(quantidade=10000):
    frases = set()
    with tqdm(total=quantidade, desc="Gerando frases") as pbar:
        while len(frases) < quantidade:
            template = random.choice(templates)
            frase = template.format(
                acao=random.choice(acoes),
                objeto=random.choice(objetos),
                alimento=random.choice(alimentos),
                sentimento=random.choice(sentimentos),
                lugar=random.choice(lugares),
                pessoa=random.choice(pessoas)
            )
            if frase not in frases:
                frases.add(frase)
                pbar.update(1)
        return sorted(frases)

# Gerar frases e salvar
frases_geradas = gerar_frases(900)

with open("dataset_caa.txt", "w", encoding="utf-8") as f:
    for frase in frases_geradas:
        f.write(frase + "\n")

print(f" Dataset gerado com {len(frases_geradas)} frases em 'dataset_caa.txt'")
