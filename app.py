import tempfile
from docx import Document
import fitz  # PyMuPDF
import os
import streamlit as st
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

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
print("Index Stats:===", index_stats)

model = SentenceTransformer('all-mpnet-base-v2')

# Function to embed text


def embed_text(text, file_name):
    records = [{"text": text, "file_name": file_name}]
    # print("records==", records)
    for record in records:
        record['embedding'] = model.encode(record['text']).tolist()

    # print("records==1==", records[0]['embedding'])
    return records[0]['embedding']

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


# Streamlit interface
# Streamlit page configuration
st.set_page_config(page_title="Chat with Documents - Pinecone & Google LLM",
                   page_icon="🌲", layout="wide", initial_sidebar_state="expanded")

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
        index.upsert([({"id": file_name, "values": embeddings,
                     "metadata": {"file_name": file_name, "text": text, "file_type": file_type}})], namespace=index_name)

        st.write("File uploaded and processed successfully.")
    else:
        st.write("Unsupported file type or empty content.")


def search_documents(query):
    contexts = []
    query_embeddings = model.encode(query).tolist()
    print("query_embeddings==", query_embeddings)
    results = index.query(namespace=index_name,    vector=query_embeddings,
                          top_k=20, include_metadata=True)
    print("results==", results)
    if "matches" in results:
        print("Search results==1==", results["matches"])
        for result in results["matches"]:
            contexts.append(result)

    else:
        print("No search results found.")

    print("contexts==", contexts)
    return contexts


query = st.text_input("Enter your query:")
if query:
    results = search_documents(query)
    st.write("Results:")
    print("results==2==", results)
    for match in results:
        print(match)
        st.write(match["metadata"]["text"])
