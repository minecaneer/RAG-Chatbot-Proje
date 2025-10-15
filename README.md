# RAG-Chatbot-Proje
"Akbank GenAI Bootcamp: Kanser Biyolojisi literatüründen güvenilir bilgi çekmek için tasarlanmış RAG (Retrieval Augmented Generation) tabanlı, Gemini destekli Türkçe/İngilizce soru-cevap sistemi.

# 🧬 RAG Tabanlı Kanser Biyolojisi Literatür Sorgulama Sistemi (CancerLit-RAG)

[cite_start]Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiştir[cite: 1].
[cite_start]Projenin temel amacı [cite: 9][cite_start], bilimsel güvenilirliği yüksek, makale özetlerinden oluşan bir bilgi havuzu kullanarak **RAG (Retrieval Augmented Generation)** temelli bir sohbet robotu geliştirmektir[cite: 2]. Chatbot, Kanser Biyolojisi alanındaki spesifik sorulara doğru ve kaynak destekli cevaplar sunarak Büyük Dil Modellerinin (LLM) halüsinasyon riskini azaltmayı hedefler.

## 💾 Veri Seti Hakkında Bilgi
...
* **İçerik:** Toplam **18316** makale özeti birleştirilmiştir. Her özet, RAG için birincil bilgi kaynağı olarak kullanılmıştır.
* **Hazırlık:** Metinler, RAG sistemine beslenmeden önce **RecursiveCharacterTextSplitter (Chunk Boyutu: 1000, Overlap: 150)** ile daha küçük ve yönetilebilir parçalara ayrılmıştır.

## 🛠️ Kullanılan Yöntemler ve Çözüm Mimarisi
...
* **RAG Çerçevesi:** LangChain
* **Vektör Veritabanı:** ChromaDB

## 📊 Elde Edilen Sonuçlar (Özet)

[cite_start]Bu projede RAG mimarisinin bilimsel alanda doğru bilgi sağlama potansiyeli gösterilmiştir.
* Chatbot, spesifik genetik mekanizmalar ve kanser biyolojisi terimleri hakkında kaynak destekli cevaplar üreterek geleneksel LLM'lerin halüsinasyonlarını etkili bir şekilde azaltmıştır.
* [Proje testlerinden elde edilen Nicel Bir Sonuç (Örn: X.XX F1 skoru) veya Kalitatif Bir Çıkarım (Örn: Türkçe Cevap Kalitesi)]

---

## 🔗 Uygulamanın Web Bağlantısı (Deploy Linki)

Projenin canlı web arayüzüne aşağıdaki linkten ulaşabilirsiniz:

[https://Sizin-Web-Linkiniz.streamlit.app/](https://Sizin-Web-Linkiniz.streamlit.app/)


## 🚀 Kodunuzun Çalışma Kılavuzu (Setup)

Projenin yerel veya bulut ortamında (Google Colab) çalıştırılması için aşağıdaki adımlar takip edilmelidir.

1.  **Bağımlılıkların Kurulumu:** Projenin tüm gereksinimleri `requirements.txt` dosyasında listelenmiştir. Bu dosya kullanılarak kurulum yapılmalıdır:
    ```bash
    pip install -r requirements.txt
    ```
2.  **API Anahtarının Ayarlanması:** Gemini API'nin kullanılabilmesi için `GEMINI_API_KEY` ortam değişkeni ayarlanmalıdır.
    ```bash
    export GEMINI_API_KEY="AIzaSy..." # Yerel ortamda
    # Colab'de ise Colab Secrets (Sırlar) özelliği kullanılmalıdır.
    ```
3.  **Çalıştırma:** Ana RAG pipeline ve Streamlit arayüzü, aşağıdaki dosya çalıştırılarak başlatılır:
    ```bash
    python streamlit_app.py 
    # VEYA Colab Notebook'u çalıştırılmalıdır: cancer_rag_chatbot.ipynb
    ```
