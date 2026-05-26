import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_core.documents import Document

# 1. Page Configuration
st.set_page_config(page_title="KnowledgeVault AI", page_icon="🧠", layout="centered")
st.title("🧠 KnowledgeVault: Multi-Data Brain")
st.markdown("---")

# 2. Secret Keys Load Karein
load_dotenv()

# 3. Initialize Models (Cache hata diya taake Windows par hang na ho)
def init_models():
    llm = ChatGroq(model="llama-3.1-8b-instant")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = init_models()

# 4. Advanced Multi-Format Loading Logic
@st.cache_resource
def load_data_folder():
    if not os.path.exists("data"):
        return None
    
    all_docs = []
    
    # Text aur CSV files ke liye loaders
    text_loader = DirectoryLoader("data", glob="./*.txt", loader_cls=TextLoader)
    csv_loader = DirectoryLoader("data", glob="./*.csv", loader_cls=CSVLoader)
    
    # Error handling ke sath files load karna
    try:
        all_docs.extend(text_loader.load())
    except:
        pass
        
    try:
        all_docs.extend(csv_loader.load())
    except:
        pass
    
    # Excel (.xlsx) files ke liye custom logic
    for file in os.listdir("data"):
        if file.endswith(".xlsx"):
            file_path = os.path.join("data", file)
            try:
                df = pd.read_excel(file_path)
                for index, row in df.iterrows():
                    content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    all_docs.append(Document(page_content=content, metadata={"source": file}))
            except Exception as e:
                st.warning(f"Is file ko parhne mein masla aaya: {file}")

    if not all_docs:
        return None

    # Splitting logic
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_docs)
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

with st.spinner("Dimaagh load ho raha hai..."):
    vector_db = load_data_folder()

if vector_db:
    st.success("✅ Agent ne aapke 'data' folder ki files ko samajh liya hai!")
else:
    st.error("❌ 'data' folder mein koi files nahi milin. Barae meharbani files daalein.")

# 5. Chat Interface (Strict RAG Prompt)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Apne data se kuch bhi poochein..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Data dhoond raha hoon..."):
            if vector_db:
                docs = vector_db.similarity_search(prompt_input, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
            else:
                context = "Koi data available nahi hai."
            
            # The STRICT Zanjeer (No Hallucinations)
            system_prompt = f"""Tum ek Expert Data Analyst ho.
            Sirf niche diye gaye CONTEXT ko parh kar sawal ka jawab do.
            
            SAKHT HIDAYAT (STRICT RULES):
            1. Jawab bilkul TO-THE-POINT aur direct do. 
            2. Apne dimaagh mein kya soch rahe ho, ya tumne kya ignore kiya hai, aisi koi lambi kahani ya safai mat do. 
            3. Seedha data batao. Jo sawal pucha jaye, sirf uska jawab do.
            4. Roman Urdu istemal karo.
            
            CONTEXT: {context}
            SAWAL: {prompt_input}"""
            
            response = llm.invoke(system_prompt)
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})