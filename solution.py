import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from streamlit_mic_recorder import mic_recorder
import requests
import pdfplumber
import io
import os
import time

# --- GOOGLE GENAI IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate 

# Google Drive & Auth
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ==========================================
# 0. CONFIG & SECRETS
# ==========================================
st.set_page_config(page_title="CampusMind AI", page_icon="🎓", layout="wide")

# --- AUTH SETUP ---
try:
    if "GOOGLE_API_KEY" in st.secrets: 
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 
except Exception as e:
    st.error(f"🚨 Secrets Error: {e}")

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except: return None

def upload_to_drive(file_path, file_name):
    try:
        if "gcp_service_account" not in st.secrets: return "Error: Secrets missing"
        key_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(key_dict, scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
        media = MediaFileUpload(file_path, mimetype='application/pdf')
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e: return f"Error: {e}"

@st.cache_resource
class GlobalMemory:
    def __init__(self):
        self.files = []

def get_global_memory():
    return GlobalMemory()

def update_global_files_from_drive():
    memory = get_global_memory()
    try:
        if "gcp_service_account" in st.secrets:
            key_dict = st.secrets["gcp_service_account"]
            creds = service_account.Credentials.from_service_account_info(key_dict, scopes=['https://www.googleapis.com/auth/drive'])
            service = build('drive', 'v3', credentials=creds)
            query = f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"
            results = service.files().list(q=query, pageSize=3, fields="files(id, name, createdTime)", orderBy="createdTime desc", supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
            memory.files = results.get('files', [])
    except: pass

if not get_global_memory().files:
    update_global_files_from_drive()

# --- GEMINI FUNCTIONS ---
def transcribe_audio_gemini(audio_bytes):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Transcribe this audio exactly.",
            {"mime_type": "audio/webm", "data": audio_bytes}
        ])
        return response.text
    except:
        return ""

# --- INDEX HANDLING (VERSION 8 - INVINCIBLE) ---
INDEX_NAME = "faiss_index_v8"

def get_vector_store_batched(text_chunks):
    # Using 'models/text-embedding-004' (Newer, usually better limits)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # TINY BATCH SIZE TO SURVIVE FREE TIER
    batch_size = 5 
    total_chunks = len(text_chunks)
    progress_text = "Vectorizing... (Auto-throttling enabled)"
    my_bar = st.progress(0, text=progress_text)
    
    vector_store = None
    
    # Function to add batch with retry logic
    def add_batch_with_retry(store, batch_texts):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if store is None:
                    if os.path.exists(INDEX_NAME):
                        try:
                            store = FAISS.load_local(INDEX_NAME, embeddings, allow_dangerous_deserialization=True)
                            store.add_texts(batch_texts)
                        except:
                            store = FAISS.from_texts(batch_texts, embedding=embeddings)
                    else:
                        store = FAISS.from_texts(batch_texts, embedding=embeddings)
                else:
                    store.add_texts(batch_texts)
                return store # Success
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str:
                    st.warning(f"Rate limit hit. Cooling down for 65 seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(65) # Wait out the minute limit
                    continue
                else:
                    raise e # Real error, don't retry
        return store

    # Process all chunks
    for i in range(0, total_chunks, batch_size):
        batch = text_chunks[i : i + batch_size]
        
        vector_store = add_batch_with_retry(vector_store, batch)
        
        # Update UI
        percent_complete = min(1.0, (i + batch_size) / total_chunks)
        my_bar.progress(percent_complete, text=f"Processing... {int(percent_complete*100)}%")
        
        # Standard safety sleep
        time.sleep(1.0)
        
    if vector_store:
        vector_store.save_local(INDEX_NAME)
    my_bar.empty()

def get_conversational_chain():
    prompt_template = """
    Answer based on the Context provided.
    Context: {context}
    Question: {question}
    Answer:
    """
    # gemini-pro is the safest choice for RAG
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ==========================================
# 2. UI SETUP
# ==========================================
st.markdown("""<style>
.stApp { background-color: #050913; color: white; } 
.glass-card { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.1); }
</style>""", unsafe_allow_html=True)

lottie_admin = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")
if "chat_history" not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=64)
    st.title("CampusMind")
    selected = option_menu("Nav", ["Student Chat", "Admin Portal"], icons=['chat', 'cloud'], default_index=0)

# ==========================================
# PAGE 1: CHAT
# ==========================================
if selected == "Student Chat":
    st.title("CampusMind AI")
    
    memory = get_global_memory()
    if memory.files:
        st.write("📄 **Recent Circulars:**")
        cols = st.columns(3)
        for i, f in enumerate(memory.files[:3]):
            with cols[i]:
                st.markdown(f"<div class='glass-card'>{f['name'][:30]}...</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 8])
    with c1:
        audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder')
    with c2:
        voice_text = ""
        if audio:
            with st.spinner("Listening..."):
                voice_text = transcribe_audio_gemini(audio['bytes'])
        user_question = st.text_input("Ask a question", value=voice_text)

    if user_question:
        with st.spinner("Thinking..."):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                if os.path.exists(INDEX_NAME):
                    new_db = FAISS.load_local(INDEX_NAME, embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    res = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    st.success(res['output_text'])
                else:
                    st.warning("⚠️ No circulars uploaded yet! Go to Admin Portal.")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# PAGE 2: ADMIN
# ==========================================
if selected == "Admin Portal":
    st.title("Admin Upload")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Upload"):
        if pdf_docs:
            with st.spinner("Starting processing..."):
                raw_text = ""
                for pdf in pdf_docs:
                    with pdfplumber.open(pdf) as f:
                        for page in f.pages: raw_text += page.extract_text()
                    upload_to_drive(pdf.name, pdf.name)
                
                get_global_memory().files = [{"name": p.name} for p in pdf_docs] + get_global_memory().files
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(raw_text)
                
                # CALL THE NEW INVINCIBLE FUNCTION
                get_vector_store_batched(chunks)
                
                st.success("Knowledge Base Updated Successfully!")
                time.sleep(1)
                st.rerun()
