# Markdown RAG

Projeto para construir uma aplicação RAG local que utilizará arquivos Markdown como base de conhecimento. Nesta etapa, o repositório contém apenas a estrutura inicial e uma API FastAPI com endpoint de verificação de saúde.

## Estrutura atual

```text
pipe_rag/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── __init__.py
│   │   ├── rag/
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── main.py
│   ├── .env.example
│   └── requirements.txt
├── material/
├── .gitignore
└── README.md
```

## Preparação do backend

Entre na pasta do backend:

```bash
cd backend
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual no Linux ou macOS:

```bash
source .venv/bin/activate
```

No Windows (PowerShell), use:

```powershell
source .venv/Scripts/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Crie o arquivo local de variáveis de ambiente:

```bash
cp .env.example .env
```

Edite `.env` quando uma chave real for necessária. O arquivo é ignorado pelo Git e não deve ser versionado.

## Executando

A partir da pasta `backend`, inicie a API:

```bash
uvicorn app.main:app --reload
```

A API estará disponível em `http://127.0.0.1:8000`. Para testar o endpoint de saúde, acesse `http://127.0.0.1:8000/health` ou execute:

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

## Indexação dos documentos

O processo de indexação executa o fluxo:

```text
Markdown -> chunks -> embeddings -> ChromaDB
```

Com o ambiente virtual ativo e a partir da pasta `backend`, execute a indexação incremental:

```bash
python -m scripts.index_material
```

O script consulta no Chroma os IDs já existentes antes de gerar embeddings. Somente chunks ausentes são enviados ao Google, e cada batch é persistido imediatamente. Se a execução for interrompida, execute o mesmo comando novamente para continuar dos chunks pendentes.

Para remover todos os registros anteriores e reconstruir completamente a collection:

```bash
python -m scripts.index_material --clear
```

Por padrão, o ChromaDB persiste os dados em `backend/data/chroma` na collection `markdown_docs`. Esses valores podem ser alterados por `CHROMA_PATH` e `CHROMA_COLLECTION` no arquivo `.env`.

O tamanho dos batches, os intervalos e as tentativas após limite de quota também podem ser configurados:

```env
EMBEDDING_BATCH_SIZE=50
EMBEDDING_BATCH_DELAY_SECONDS=35
EMBEDDING_RETRY_DELAY_SECONDS=65
EMBEDDING_MAX_RETRIES=3
```

O tamanho do batch e o intervalo entre batches podem ser sobrescritos pela linha de comando:

```bash
python -m scripts.index_material --batch-size 25 --delay 40
```

A estratégia incremental atual identifica chunks por `relative_path::chunk_index`. Uma alteração de conteúdo que mantenha o mesmo ID não é detectada; nesse caso, use `--clear`. Uma versão futura poderá incorporar um hash do conteúdo para detectar essas alterações automaticamente.

O `upsert` atualiza IDs processados, mas chunks antigos podem permanecer se um arquivo for removido ou passar a gerar menos chunks. A reconstrução com `--clear` também resolve esse caso durante o desenvolvimento.

## Busca semântica

Para consultar diretamente os chunks já indexados:

```bash
python -m scripts.search_material "Como funciona uma rede neural?"
```

O limite padrão é configurado por `RAG_TOP_K=5` no `.env`. Ele pode ser sobrescrito no terminal:

```bash
python -m scripts.search_material "Como funciona uma rede neural?" --top-k 3
```

O comando gera o embedding da consulta e mostra os chunks mais próximos, suas distâncias e previews limitados a 500 caracteres. Como a collection pode estar parcialmente indexada, resultados relevantes ainda podem estar ausentes.

## Respostas com RAG

Para gerar uma resposta baseada nos documentos recuperados:

```bash
python -m scripts.ask_material "O que é backpropagation?"
```

Também é possível ajustar a quantidade de chunks e exibir informações de diagnóstico:

```bash
python -m scripts.ask_material "Explique Ridge e Lasso" --top-k 5 --debug
```

O fluxo executado é:

```text
Pergunta -> embedding -> retrieval -> contexto -> Gemini -> resposta
```

O modelo de geração é configurado por `GOOGLE_CHAT_MODEL` e recebe instruções para responder somente com o contexto documental. A qualidade da resposta depende dos chunks disponíveis; uma collection parcialmente indexada pode não conter o documento necessário.

## API de chat

Com o backend em execução, envie uma pergunta para `POST /chat`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"O que é backpropagation?","top_k":3}'
```

O campo `top_k` é opcional e aceita valores entre 1 e 20. A resposta contém apenas o texto gerado e os metadados públicos das fontes. A documentação interativa está disponível em `http://127.0.0.1:8000/docs`.

Por padrão, o CORS permite o frontend em `http://localhost:5173`. O endereço pode ser alterado com `FRONTEND_ORIGIN` no `.env`.

## Frontend

O frontend usa React, TypeScript, Vite e TailwindCSS. Para executar a aplicação completa, inicie primeiro o backend:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

Em outro terminal, prepare e inicie o frontend:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

A interface estará disponível em `http://localhost:5173` e enviará as perguntas para `POST /chat`. A URL do backend pode ser alterada com `VITE_API_URL` no arquivo `frontend/.env`; nenhuma chave da API Google é utilizada ou exposta pelo frontend.

Para validar o TypeScript e gerar a versão de produção:

```bash
cd frontend
npm run build
```
