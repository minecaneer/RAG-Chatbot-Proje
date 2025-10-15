# HÜCRE 8: streamlit_app.py dosyasını oluşturma (KURALA UYGUN VE SON VERSİYON)
%%writefile streamlit_app.py
import streamlit as st
import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.dataframe import DataFrameLoader # Yeni eklendi

# --- Konfigürasyonlar ve Sabitler ---
CHROMA_PATH = "cancer_pubmed_db"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Streamlit Cloud'da secret ayarını kontrol eder.
    st.error("GEMINI_API_KEY ortam değişkeni bulunamadı. Lütfen Streamlit Secrets'ı kontrol edin.")
    st.stop()

def create_and_persist_vector_db():
    """Veritabanı yoksa HuggingFace'ten veriyi çekip ChromaDB'yi yeniden oluşturur."""
    st.info("Veritabanı bulunamadı. ChromaDB, Hugging Face verileri kullanılarak yeniden oluşturuluyor...")

    # 1. Veri Setlerini Yükleme ve Kısıtlama (1000 satır)
    # Bu, Streamlit Cloud'un hızlı çalışması için kritiktir.
    breast_data = load_dataset("Gaborandi/breast_cancer_pubmed_abstracts", split="train") 
    lung_data = load_dataset("Gaborandi/Lung_Cancer_pubmed_abstracts", split="train") 
    combined_data = concatenate_datasets([breast_data, lung_data])
    df = combined_data.to_pandas()
    df = df.sample(n=1000, random_state=42) # 1000 satırla sınırla
    df = df[['abstract', 'title']].dropna() 

    # 2. Chunking ve Dokümanlara Çevirme
    loader = DataFrameLoader(df, page_content_column="abstract")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    # 3. Embedding Modelini Yükleme
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    
    # 4. ChromaDB'yi Oluşturma
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    # Persist etmeye gerek yok, Streamlit Cloud'da yeniden oluşturulacaktır.
    st.success(f"ChromaDB, {len(chunks)} parça ile başarıyla yeniden oluşturuldu.")
    return vector_store

@st.cache_resource
def setup_rag_chain():
    """RAG zincirini kurar ve döndürür."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )

    # 1. Veritabanını Yükle veya Yeniden Oluştur (GitHub'a yüklenmediği için)
    if not os.path.exists(CHROMA_PATH):
        # Dosya yoksa yeniden oluştur
        vector_store = create_and_persist_vector_db()
    else:
        # Dosya varsa yükle (Bu kod sadece Colab'de çalışırken tetiklenir)
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
    
    # 2. LLM ve Retriever'ı Kur
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=GEMINI_API_KEY
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 3. RAG Zincirini Oluştur
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa_chain

# --- Streamlit Arayüzü (Aynı kalır) ---
st.set_page_config(page_title="CancerLit-RAG Chatbot", layout="wide")
st.title("🧬 CancerLit-RAG: Kanser Biyolojisi Chatbot")
st.markdown("PubMed makale özetleri üzerine kurulmuş, **Gemini** destekli $\text{RAG}$ (Retrieval Augmented Generation) sistemi. Spesifik sorularınıza bilimsel ve kaynak destekli cevaplar sağlar.")

qa_chain = setup_rag_chain()

# Sohbet geçmişi ve kullanıcı girişi (önceki gibi devam eder...)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Kanser biyolojisi veya genetiği hakkında bir soru sorun (Örn: BRCA1 mutasyonu nedir?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Bilimsel literatür taranıyor ve cevap üretiliyor..."):
        result = qa_chain.invoke({"query": prompt})
        response = result['result']
        sources = result['source_documents']
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
            if sources:
                st.subheader("💡 Cevap Kaynakları (İlk 3 Kaynak):")
                for i, doc in enumerate(sources):
                    title = doc.metadata.get('title', 'Başlık Yok')
                    st.markdown(f"**{i+1}. Kaynak Başlığı:** {title[:150]}...")

    st.session_state.messages.append({"role": "assistant", "content": response})
