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
FAISS_PATH = "faiss_index"


#  Interface Streamlit 
st.title("🤖 SGCI Bot")
st.write("Discutez avec votre assistant IA !")

# Stocker l'historique des conversations
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Saisie utilisateur
user_input = st.chat_input("Posez votre question ici :")
if user_input:
    # Ajouter l'entrée utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)    

    #  Définir l’embedding nomic-embed-text pour recherche dans FAISS
    embedding_search = OllamaEmbeddings(model="nomic-embed-text")

    #  Charger l'index FAISS
    if os.path.exists(f"{FAISS_PATH}/index.faiss"):
        vector_store = FAISS.load_local(FAISS_PATH, embeddings=embedding_search, allow_dangerous_deserialization=True)
    else:
        st.error(" Aucun index FAISS trouvé ! ")

    #  Définir le modèle de chat Llama pour la génération de réponses
    chat_model = ChatOllama(model="llama3.2:latest")

    #  Ajouter la mémoire conversationnelle
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Définir la chaîne de conversation avec recherche dans FAISS
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    # Appel à la chaîne de conversation pour obtenir une réponse
    response = qa_chain.run(user_input)
    bot_reply = response

    # Ajouter la réponse du bot à l'historique
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
