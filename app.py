import tempfile
from docx import Document
import fitz  # PyMuPDF
import os
import streamlit as st
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Pinecone
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

cloud = 'aws'
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "document-chatbot-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=768,
        metric='cosine',
        spec=spec
    )

# connect to index
index = pc.Index(index_name)
# view index stats
index_stats = index.describe_index_stats()

model = SentenceTransformer('all-mpnet-base-v2')

# Function to embed text using SentenceTransformer


def embed_text(text, file_name):
    doc_chunks = text.split('. ')
    vectors = []
    for i, doc_chunk in enumerate(doc_chunks):
        embedding = model.encode(doc_chunk).tolist()
        vectors.append({
            'id': f'{file_name}_chunk_{i}',
            'values': embedding,
            'metadata': {"file_name": file_name, 'text': doc_chunk}
        })
    return vectors

# Function to extract text based on file type


def extract_text(file, file_type):
    if file_type == "pdf":
        reader = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in reader:
            text += page.get_text()
        return text
    elif file_type == "docx":
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc = Document(tmp_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        os.remove(tmp_path)
        return text
    elif file_type == "txt":
        return file.read().decode("utf-8")
    return ""


# Streamlit interface to upload documents and search
st.set_page_config(page_title="Chat with Documents - Pinecone & OpenAI GPT-4",
                   page_icon="🌲", layout="wide", initial_sidebar_state="expanded")
st.title("Chat with Documents - Pinecone & OpenAI GPT-4")
uploaded_file = st.file_uploader(
    "Upload a document", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_type = uploaded_file.type.split('/')[-1]
    file_name = uploaded_file.name
    text = extract_text(uploaded_file, file_type)
    # print(text)
    if text:
        embeddings = embed_text(text, file_name)
        index.delete(ids=[file_name])
        index.upsert(embeddings, namespace=index_name)
        st.write("File uploaded and processed successfully.")
    else:
        st.write("Unsupported file type or empty content.")


def search_documents(query):
    contexts = []
    query_embeddings = model.encode(query).tolist()
    # print("query_embeddings==", query_embeddings)
    results = index.query(namespace=index_name,  vector=query_embeddings,
                          top_k=5, include_metadata=True)
    # print("results==", results)
    if "matches" in results:
        # print("Search results==1==", results["matches"])
        for result in results["matches"]:
            contexts.append(result)
    else:
        print("No search results found.")

    return contexts


query = st.text_input("Enter your query:")
if query:
    retrieved_texts = []
    results = search_documents(query)
    st.write("Results:")
    for match in results:
        retrieved_texts.append(match["metadata"]["text"])

    context = "\n".join(retrieved_texts)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a useful assistant. Use the assistant's content to answer the user's query \
        Summarize your answer using the 'text' and cite the 'file_name' metadata in your reply."},
            {"role": "assistant", "content": context},
            {"role": "user", "content": query}
        ]
    )
    answer = response.choices[0].message.content
    st.write(answer)
