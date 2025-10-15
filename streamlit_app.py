import streamlit as st
import os

from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Konfigürasyonlar ---
# Colab'de tanımlanan API anahtarını Streamlit uygulamasına yükleme
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY ortam değişkeni bulunamadı. Lütfen Colab Secrets'ı kontrol edin.")
    st.stop()

# ChromaDB veritabanı yolu (Colab'de oluşturduğumuz klasör)
CHROMA_PATH = "cancer_pubmed_db"

@st.cache_resource
def setup_rag_chain():
    """RAG zincirini kurar ve döndürür."""
    # 1. Embedding Modelini Yükle
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_API_KEY
    )
    
    # 2. Vektör Veritabanını Yükle
    if not os.path.exists(CHROMA_PATH):
        st.error("ChromaDB veritabanı dosyaları ('cancer_pubmed_db') bulunamadı. Lütfen 5. Hücreyi (Embedding) çalıştırın ve dosyaları GitHub'a yükleyin.")
        st.stop()
        
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    
    # 3. LLM ve Retriever'ı Kur
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=GEMINI_API_KEY
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 4. RAG Zincirini Oluştur
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- Streamlit Arayüzü ---

st.set_page_config(page_title="CancerLit-RAG Chatbot", layout="wide")
st.title("🧬 CancerLit-RAG: Kanser Biyolojisi Chatbot")
st.markdown("PubMed makale özetleri üzerine kurulmuş, **Gemini** destekli $\text{RAG}$ (Retrieval Augmented Generation) sistemi. Spesifik sorularınıza bilimsel ve kaynak destekli cevaplar sağlar.")

# RAG zincirini bir kez kur
qa_chain = setup_rag_chain()

# Sohbet geçmişini başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sohbet geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı girişi
if prompt := st.chat_input("Kanser biyolojisi veya genetiği hakkında bir soru sorun (Örn: BRCA1 mutasyonu nedir?)..."):
    # Kullanıcı mesajını geçmişe ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Bilimsel literatür taranıyor ve cevap üretiliyor..."):
        # RAG zincirini çağır
        result = qa_chain.invoke({"query": prompt})
        response = result['result']
        sources = result['source_documents']
        
        # Cevabı göster
        with st.chat_message("assistant"):
            st.markdown(response)
            
            # Kaynakları göster (Proje gereksinimi için kritik)
            if sources:
                st.subheader("💡 Cevap Kaynakları (İlk 3 Kaynak):")
                for i, doc in enumerate(sources):
                    title = doc.metadata.get('title', 'Başlık Yok')
                    st.markdown(f"**{i+1}. Kaynak Başlığı:** {title[:150]}...")
                    #st.caption(f"İçerik Parçası: {doc.page_content[:200]}...")

    # Asistan cevabını geçmişe ekle
    st.session_state.messages.append({"role": "assistant", "content": response})
