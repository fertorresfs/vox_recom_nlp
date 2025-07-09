# VoxRecomNLP: Teclado Virtual com Recomendação Textual para CAA

 <!-- Substitua com um GIF ou imagem do seu projeto em ação -->

## 📜 Sobre o Projeto

O **RecomNLP** é um protótipo de sistema de **Comunicação Aumentativa e Alternativa (CAA)** projetado para auxiliar pessoas com necessidades complexas de comunicação. O sistema consiste em um teclado virtual personalizável (desenvolvido em React) que se integra a um poderoso motor de recomendação textual (backend em FastAPI e Python).

O objetivo principal é acelerar a comunicação e reduzir o esforço de digitação, oferecendo sugestões de palavras contextualmente relevantes em tempo real. A arquitetura é híbrida, combinando a velocidade de estruturas de dados clássicas (Trie) com o poder contextual de modelos de linguagem modernos como BERT e GPT-2.

Este projeto foi desenvolvido como parte de uma pesquisa de mestrado no Instituto de Ciências Matemáticas e de Computação (ICMC) da Universidade de São Paulo (USP).

## ✨ Funcionalidades Principais

-   **Teclado Virtual Personalizável:** Interface desenvolvida em React, permitindo ao usuário customizar o layout das teclas.
-   **Motor de Recomendação Híbrido:**
    -   **Autocompletar com Trie:** Sugere a completação da palavra atual com base em um léxico de frequência, oferecendo respostas rápidas.
    -   **Previsão de Próxima Palavra com BERT (Fine-tuned):** Utiliza um modelo BERTimbau (com fine-tuning opcional em um corpus de CAA) para prever a próxima palavra com base no contexto da frase.
-   **Transcrição de Voz em Tempo Real:** Um endpoint WebSocket utiliza o modelo **Vosk** para transcrever a fala do usuário continuamente, permitindo uma comunicação multimodal.
-   **API de Alta Performance:** Backend construído com **FastAPI**, garantindo baixa latência para as sugestões.
-   **Pipeline de Pré-processamento:** Um conjunto de scripts para processar léxicos, gerar embeddings, criar a estrutura Trie e realizar o fine-tuning dos modelos.

## 🛠️ Tecnologias Utilizadas

### Backend (Python)
-   **Framework:** FastAPI, Uvicorn
-   **Modelos de PLN:** Transformers (BERTimbau, GPT-2), Vosk
-   **Estruturas de Dados:** Pygtrie
-   **Comunicação em Tempo Real:** WebSockets
-   **Processamento de Dados:** NumPy, aiofiles

### Frontend (JavaScript/React)
-   **Framework:** React.js
-   **Gerenciamento de Estado:** (Mencione se usa Redux, Context API, etc.)
-   **Estilização:** Styled Components
-   **Comunicação com API:** Fetch API, WebSocket API

### Ferramentas de Desenvolvimento
-   **Ambiente:** Python `venv`
-   **Gerenciador de Pacotes:** `pip`, `npm`
-   **Controle de Versão:** Git, GitHub

## 🚀 Instalação e Execução

Siga os passos abaixo para configurar e rodar o projeto localmente.

### Pré-requisitos
-   Python 3.9+
-   Node.js e npm
-   Git

### 1. Clonar o Repositório
```bash
git clone https://github.com/seu-usuario/recomnlp.git
cd recomnlp
```
2. Configuração do Backend (Servidor Python)
a. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```
b. Instale as dependências Python:
```bash
pip install -r requirements.txt
```
Nota: O requirements.txt deve conter todas as bibliotecas, como fastapi, uvicorn, transformers, torch, vosk, pygtrie, websockets, accelerate, etc.

c. Baixe o Modelo Vosk:
Baixe o modelo em português (small ou large) do site oficial do Vosk e descompacte-o na raiz do projeto, na pasta vosk-model-small-pt-0.3.

d. Execute o Pipeline de Pré-processamento:
Este passo é crucial e só precisa ser executado uma vez (ou sempre que quiser atualizar os recursos). Ele irá gerar os arquivos .pkl, .npy e o modelo fine-tuned.
```bash
python executar_pipeline.py
```
Nota: Para o fine-tuning, certifique-se de que o arquivo dataset_caa.txt existe. Você pode gerá-lo com python gerar_dataset_caa.py.

e. Inicie o Servidor FastAPI:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
O backend estará rodando em http://localhost:8000.

3. Configuração do Frontend (Aplicação React)
a. Navegue até o diretório do cliente:
```bash
cd client
```
b. Instale as dependências JavaScript:
```bash
npm install
```
c. Inicie a Aplicação React:
```bash
npm start
```
O frontend estará acessível em http://localhost:3000 e se conectará automaticamente ao backend.

API Endpoints
A API expõe os seguintes endpoints principais:

GET /sugestoes_hibrido/: Endpoint principal para sugestões de texto. Recebe um parâmetro texto.

GET /sugestoes/: Sugestões baseadas apenas na Trie.

GET /frequencia/: Retorna a frequência de uma palavra.

GET /embedding/: Retorna o vetor de embedding de uma palavra.

WebSocket /ws/transcricao_em_tempo_real: Endpoint para transcrição e sugestão de voz em tempo real.

📂 Estrutura do Projeto
```bash
.
├── client/              # Código do frontend React
│   ├── public/
│   └── src/
├── vosk-model-small-pt-0.3/ # Modelo de reconhecimento de voz
├── bert_finetuned_caa/    # Modelo BERT após fine-tuning
├── api.py               # Servidor FastAPI com os endpoints
├── executar_pipeline.py   # Orquestrador do pré-processamento
├── preprocessamento.py    # Funções de lógica de PLN e fine-tuning
├── gerar_dataset_caa.py   # Script para gerar dataset sintético
├── dataset_caa.txt      # Dataset para fine-tuning
├── requirements.txt     # Dependências Python
└── README.md
```
📈 Trabalhos Futuros
Integração com Rastreamento Ocular: Implementar o controle do teclado via WebGazer.js.

Aprimoramento dos Modelos: Realizar fine-tuning com dados de usuários reais para personalização.

Testes de Usabilidade: Conduzir estudos formais com o público-alvo para validar a eficácia do sistema.

Otimização de Performance: Quantização dos modelos para rodar em dispositivos com menos recursos.

Deploy: Empacotar a aplicação com Docker para facilitar o deploy.

🤝 Contribuições
Contribuições são bem-vindas! Se você tiver ideias para novas funcionalidades, melhorias ou correções de bugs, sinta-se à vontade para abrir uma issue ou enviar um pull request.

Faça um fork do projeto.

Crie uma nova branch (git checkout -b feature/minha-feature).

Faça o commit das suas alterações (git commit -m 'Adiciona minha-feature').

Faça o push para a branch (git push origin feature/minha-feature).

Abra um Pull Request.

📄 Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

📧 Contato
Fernando Torres Ferreira da Silva - fernandotfs@usp.br

Link do Projeto: https://github.com/fertorresfs/vox_recom_nlp
