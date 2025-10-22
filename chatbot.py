import streamlit as st
import os
from google.oauth2 import service_account

# Gerekli LangChain importları
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# -------------------------------
# 1. Service Account ve Ortam Ayarları
# -------------------------------
SERVICE_ACCOUNT_FILE = r"C:\Users\Hanım\Desktop\GenAIMuzikAsistani\service-account.json"

if not os.path.exists(SERVICE_ACCOUNT_FILE):
    st.error(f"Hata: Service Account JSON dosyası bulunamadı. Lütfen '{SERVICE_ACCOUNT_FILE}' dosyasını kontrol edin.")
    st.stop()

# Credentials oluştur
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)

# -------------------------------
# 2. RAG Pipeline Kurulumu
# -------------------------------
@st.cache_resource
def setup_rag_pipeline(pdf_path):
    # Streamlit, spinner içeriğini otomatik olarak gösterir.
    with st.spinner("Müzik teorisi belgeleri yükleniyor ve vektör veritabanı hazırlanıyor..."):
        if not os.path.exists(pdf_path):
            st.error(f"Hata: Veri dosyası bulunamadı. Lütfen '{pdf_path}' dosyasını proje klasörüne koyun.")
            return None

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Chunking (Parçalama)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # Embedding ve Vektör Veritabanı
        # Yeni hata "unexpected model name format" nedeniyle,
        # model adını tam API yoluyla ('models/') güncelliyoruz.
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            credentials=credentials
        )

        try:
            vectordb = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            return retriever
        except Exception as e:
            st.error(f"Vektör veritabanı oluşturulurken bir hata oluştu: {e}")
            return None


# -------------------------------
# 3. Chatbot Fonksiyonu
# -------------------------------
def main():
    st.set_page_config(page_title="🎼 Müzik Notaları RAG Chatbotu", layout="wide")
    st.title("🎵 Müzik Notaları Asistanı (RAG Destekli)")

    PDF_DOSYA_ADI = "Temel Müzik Eğitimi.pdf"
    retriever = setup_rag_pipeline(PDF_DOSYA_ADI)
    if retriever is None:
        return

    # LLM (Google Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        credentials=credentials
    )

    # Prompt Tanımı
    prompt = ChatPromptTemplate.from_template("""
    Aşağıdaki bağlamı kullanarak soruyu yanıtla. Yanıt Türkçe, kısa ve doğru olmalı. 
    Eğer bağlamda soruya cevap verecek yeterli bilgi yoksa, "Elimdeki müzik teorisi bilgileri bu soruyu tam olarak yanıtlamak için yetersizdir." de.

    Bağlam:
    {context}

    Soru:
    {input}
    """)

    # Stuff Chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # RAG zinciri
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Streamlit oturum durumu başlatma
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Mesajları göster
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcı girişi
    if user_query := st.chat_input("🎤 Müzikle ilgili bir soru sor..."):
        # Kullanıcı mesajını ekle
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Cevabı al ve göster
        with st.chat_message("assistant"):
            with st.spinner("Cevap aranıyor ve oluşturuluyor..."):
                try:
                    response = qa_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    
                    # Kaynakları cevaba ekle
                    sources_info = "\n\n**📚 Kaynaklar:**\n"
                    for i, doc in enumerate(response["context"]):
                        src = doc.metadata.get("source", "Bilinmiyor")
                        page = doc.metadata.get("page", "Bilinmiyor")
                        sources_info += f"- Sayfa {page}: {src.split('/')[-1]}\n"
                    
                    full_response = answer + sources_info
                    
                    st.markdown(answer)
                    st.markdown(sources_info)
                    
                    # Asistan mesajını oturum durumuna ekle
                    st.session_state["messages"].append({"role": "assistant", "content": full_response})

                except Exception as e:
                    error_message = f"Cevap oluşturulurken bir hata oluştu: {e}"
                    st.error(error_message)
                    st.session_state["messages"].append({"role": "assistant", "content": error_message})


# -------------------------------
# 4. Uygulama Başlat
# -------------------------------
if __name__ == "__main__":
    main()
