# Chat With Your Documents

A simple RAG (Retrieval Augmented Generation) chatbot. Upload a PDF or DOCX file and ask questions about it. The answers come only from your document.

## Features

- Upload PDF and DOCX files
- Ask questions in a chat screen
- See the source chunks used for each answer
- Delete your data with one button

## How It Works

Ingestion (runs once per file):

```
file -> extract text -> split into chunks -> create embeddings -> store in Qdrant
```

Query (runs for every question):

```
question -> create embedding -> search Qdrant for top 4 chunks -> build prompt -> Groq LLM -> answer
```

## Tech Stack

- Python
- Streamlit for the user interface
- pypdf and python-docx to read files
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- Qdrant Cloud as the vector database
- Groq API for the LLM

## Project Structure

```
Chatbot/
  app.py             Chat screen
  rag.py             All RAG logic
  requirements.txt   Python packages
  .env               Secret keys (never upload this file)
  .gitignore
```

## Setup

1. Install the packages:

```bash
pip install -r requirements.txt
```

2. Create a free API key at console.groq.com.

3. Create a free cluster at cloud.qdrant.io and copy the cluster URL and API key.

4. Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_groq_key
QDRANT_URL=https://your-cluster-id.region.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key
```

## Run

```bash
streamlit run app.py
```

The first start can take a few minutes because the embedding model is downloaded.

## Deploy

1. Push the code to a private GitHub repository. Do not push the `.env` file.
2. Go to share.streamlit.io and create a new app from the repository.
3. Add the three keys in the Secrets settings of the app.
4. Limit the app to trusted email addresses in the Sharing settings.

## Settings

You can change these values at the top of `rag.py`:

| Name | Meaning | Default |
| --- | --- | --- |
| CHUNK_SIZE | Characters in one chunk | 800 |
| CHUNK_OVERLAP | Characters repeated between chunks | 150 |
| TOP_K | Number of chunks sent to the LLM | 4 |
| LLM_MODEL | Groq model name | groq_model |

## Limits

- Scanned PDFs do not work because they have no text (OCR is needed).
- Text inside Word tables is not read.
- Each question is answered alone, so the bot has no chat memory yet.