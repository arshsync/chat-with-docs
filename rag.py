import os
import uuid
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from groq import Groq

load_dotenv()

COLLECTION = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

embedder = SentenceTransformer(EMBED_MODEL)
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def create_collection():
      """Create the Qdrant collection if it does not already exist."""
      if not qdrant.collection_exists(COLLECTION):
            qdrant.create_collection(
                  collection_name=COLLECTION,
                  vectors_config=VectorParams(
                        size=embedder.get_embedding_dimension(),
                        distance=Distance.COSINE
                  ),
            )


create_collection()

# Extract text from the file
def load_text(file):
      name = file.name.lower()
      if name.endswith(".pdf"):
            reader = PdfReader(file)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
      if name.endswith(".docx"):
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
      raise ValueError("Only pdf and DOCX files are supported.")

# Split the text into chunks
def split_into_chunks(text):
      chunks = []
      start = 0
      while start < len(text):
            chunk = text[start:start + CHUNK_SIZE].strip()
            if chunk:
                  chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
      return chunks

# Embed the chunks and store them in Qdrant
def add_document(file):
      text = load_text(file)
      chunks = split_into_chunks(text)

      if not chunks:
            return 0

      vectors = embedder.encode(chunks)

      points = []

      for i, (chunk, vector) in enumerate(zip(chunks, vectors)):

            points.append(
                  PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file.name}-{i}")),
                        vector = vector.tolist(),
                        payload = {"text": chunk, "source": file.name, "chunk_index": i},
                  )
            )

      qdrant.upsert(collection_name=COLLECTION, points=points)
      return len(chunks)

# Retrieve the best chunks for a question
def retrieve(question):
      query_vector = embedder.encode(question).tolist()

      result = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=TOP_K,
      )
      return result.points

# Build the prompt and ask groq
def answer_question(question):
      points = retrieve(question)

      if not points:
            return "Please upload a document first", []

      context = "\n\n".join(
        f"[Source: {p.payload['source']}]\n{p.payload['text']}" for p in points
      )

      system_prompt = (
            "You answer questions using only the context given by the user."
            "If the answer is not in the context, say that you could not find it in the documents."
            "Do not use outside knowledge"
            "Do not hallucinate."
            "Do not invent information just say I cannot help with your query."
      )

      user_prompt = f"Context: \n{context}\n\nQuestion: {question}"

      response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages = [
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt},
            ],
            temperature=0
      )

      return response.choices[0].message.content, points


def has_documents():
    if not qdrant.collection_exists(COLLECTION):
          create_collection()
    return qdrant.count(collection_name=COLLECTION).count > 0