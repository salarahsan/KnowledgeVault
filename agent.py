import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader 

# 1. Secret Keys Load Karein
load_dotenv()

# 2. AI Model & Embeddings Setup
llm = ChatGroq(model="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("✅ KnowledgeVault Agent Ready Hai!")

# 3. File Loading & Vector DB Setup (Sirf aik baar hoga)
if os.path.exists("info.txt"):
    loader = TextLoader("info.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    vector_db = FAISS.from_documents(chunks, embeddings)
    print("🧠 Memory Loaded: Agent ne aapki file ka data yaad kar liya hai.")
else:
    print("❌ Error: info.txt file nahi mili!")
    exit()

print("\n💬 KnowledgeVault Chat Bot Zinda Hai! (Exit likh kar band karein)")

# 4. The Continuous Chat Loop 🔄
while True:
    query = input("\n👤 Aapka Sawal: ")
    
    # Exit condition
    if query.lower() in ["exit", "quit", "bye", "band karo"]:
        print("👋 Allah Hafiz Salar bhai! Agli baar milte hain.")
        break
    
    if not query.strip():
        continue

    # Database se relevant data nikalna
    docs = vector_db.similarity_search(query)
    context = docs[0].page_content
    
    # Groq AI se response lena
    prompt = f"Context: {context}\nSawal: {query}\nJawab (Urdu/Hindi Mix):"
    response = llm.invoke(prompt)
    
    print(f"🤖 AI ka Jawab: {response.content}")