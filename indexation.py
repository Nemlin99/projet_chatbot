import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOllama

# les dossiers et la creation d 'un index
DOCS_FOLDER = "C:\doc_juridique"
FAISS_PATH = "faiss_index"

#j'utilise nomic-embed-text de ollama pour l'indexation 
embedding = OllamaEmbeddings(model="nomic-embed-text")

#  Charger les documents
def load_documents(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            continue  # Ignorer les fichiers non supportés
        
        documents.extend(loader.load())
    
    return documents

# Charger et afficher les documents

docs = load_documents(DOCS_FOLDER)

# Découper les documents en petits morceaux
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Créer et sauvegarder l’index FAISS
vector_store = FAISS.from_documents(split_docs, embedding)
vector_store.save_local(FAISS_PATH)