import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests
import pdfplumber
import io
import os
import time

# --- STANDARD IMPORTS ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate 

# Google Drive & Auth
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# --- GOOGLE GEMINI (The "Google AI" Part) ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ==========================================
# 0. THEME ENGINE
# ==========================================
def force_dark_mode():
    config_dir = ".streamlit"
    config_path = os.path.join(config_dir, "config.toml")
    if not os.path.exists(config_dir): os.makedirs(config_dir)
    config_content = """
[theme]
base = "dark"
primaryColor = "#00C853"
backgroundColor = "#050913"
secondaryBackgroundColor = "#0b0f1f"
textColor = "#f5f7fb"
font = "sans serif"
    """
    if not os.path.exists(config_path):
        with open(config_path, "w") as f: f.write(config_content)
        st.rerun()

force_dark_mode()

# ==========================================
# 1. SETUP & SECRETS
# ==========================================
st.set_page_config(page_title="CampusMind AI", page_icon="🎓", layout="wide")

try:
    if "OPENAI_API_KEY" in st.secrets: 
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 
except FileNotFoundError: st.error("🚨 Secrets file not found!")

# ==========================================
# 2. HACKATHON CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #050913; }
    ::-webkit-scrollbar-thumb { background: #00C853; border-radius: 10px; }
    
    .stApp { background: radial-gradient(circle at top left, #1a2a4f 0, #050913 40%, #000000 100%); color: #f5f7fb !important; }
    [data-testid="stMain"] { background: transparent !important; }
    section[data-testid="stSidebar"] { background: rgba(5, 9, 19, 0.95); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    
    /* AUDIO INPUT STYLING */
    .stAudioInput { margin-top: 10px; }
    div[data-testid="stAudioInput"] { background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 10px; border: 1px solid rgba(255, 255, 255, 0.1); }
    
    .glass-card { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(25px); border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 24px; color: #fff; }
    .answer-box-container { background: rgba(0, 200, 83, 0.04); border-radius: 12px; border: 2px solid #00C853; padding: 24px; margin-top: 30px; color: #ffffff !important; box-shadow: 0 0 50px rgba(0, 200, 83, 0.1); }
    .answer-title { color: #00ffc3; font-size: 20px; font-weight: 800; display: flex; align-items: center; gap: 12px; }
    .answer-content { font-size: 17px; line-height: 1.7; margin-top: 15px; color: #eef2f6; }
    
    .stTextInput input { background: rgba(255, 255, 255, 0.05) !important; color: #fff !important; border-radius: 8px; padding: 16px; border: 1px solid rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS
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

def get_global_memory(): return GlobalMemory()

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

if not get_global_memory().files: update_global_files_from_drive()

# --- SELF-HEALING GEMINI TRANSCRIPTION ---
def transcribe_audio_gemini(audio_bytes):
    # Safety settings to prevent blocks
    safety = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE
    }
    
    # Try 1: Standard Flash (Fastest)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            ["Transcribe this audio exactly. Output only the English text.", {"mime_type": "audio/wav", "data": audio_bytes}],
            safety_settings=safety
        )
        return response.text
    except:
        # Try 2: Versioned Flash (If 404 occurs on alias)
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-001")
            response = model.generate_content(
                ["Transcribe this audio exactly. Output only the English text.", {"mime_type": "audio/wav", "data": audio_bytes}],
                safety_settings=safety
            )
            return response.text
        except:
            # Try 3: Pro (Most powerful backup)
            try:
                model = genai.GenerativeModel("gemini-1.5-pro")
                response = model.generate_content(
                    ["Transcribe this audio exactly. Output only the English text.", {"mime_type": "audio/wav", "data": audio_bytes}],
                    safety_settings=safety
                )
                return response.text
            except Exception as e:
                st.error(f"Google AI Error: {e}")
                return ""

# --- OPENAI INTELLIGENCE ---
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if os.path.exists("faiss_index"):
        try:
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            vector_store.add_texts(text_chunks)
        except: vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    else: vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question based ONLY on the provided Context.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

lottie_admin = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ==========================================
# 4. UI STRUCTURE
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=64)
    st.markdown("## CampusMind")
    selected = option_menu("Nav", ["Student Chat", "Admin Portal", "About"], icons=['chat', 'cloud', 'info'], default_index=0)

# ==========================================
# PAGE 1: STUDENT CHAT
# ==========================================
if selected == "Student Chat":
    st.markdown("<h1 style='text-align: center;'>CampusMind AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #00ffc3;'>⚡ Powered by OpenAI & Google Gemini</p>", unsafe_allow_html=True)
    
    # Recent Circulars
    memory = get_global_memory()
    if memory.files:
        st.write("📄 **Recent Circulars:**")
        cols = st.columns(3)
        for i, f in enumerate(memory.files[:3]):
            with cols[i]:
                st.markdown(f"<div class='glass-card'>{f['name'][:30]}...</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # --- NATIVE AUDIO INPUT (Reliable) ---
    col_audio, col_text = st.columns([1, 2])
    
    voice_query = ""
    
    with col_audio:
        st.markdown("**🎙️ Record Voice Query:**")
        audio_value = st.audio_input("Record")
        
        if audio_value:
            with st.spinner("Processing with Google Gemini..."):
                # Read bytes
                audio_bytes = audio_value.read()
                # Send to Gemini (Self-Healing Function)
                voice_query = transcribe_audio_gemini(audio_bytes)
                
                if voice_query:
                    st.success(f"Did you say: '{voice_query}'?")
    
    with col_text:
        st.markdown("**💬 Or Type:**")
        user_input = st.text_input("Question", value=voice_query, placeholder="Ask about exams, fees...")

    # Logic: Prefer voice query if it exists
    final_question = voice_query if voice_query else user_input

    if final_question:
        if "last_answered" not in st.session_state or st.session_state.last_answered != final_question:
            with st.spinner("Thinking..."):
                try:
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    if os.path.exists("faiss_index"):
                        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                        docs = new_db.similarity_search(final_question, k=10)
                        chain = get_conversational_chain()
                        res = chain.invoke({"input_documents": docs, "question": final_question}, return_only_outputs=True)
                        
                        st.session_state.last_answered = final_question
                        
                        st.markdown(f"""
                        <div class="answer-box-container">
                            <div class="answer-title">🤖 CampusMind Answer</div>
                            <div class="answer-content">{res['output_text']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No database found. Please upload circulars.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# PAGE 2: ADMIN PORTAL
# ==========================================
if selected == "Admin Portal":
    st.title("Admin Upload")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Process & Upload"):
        if pdf_docs:
            with st.spinner("Processing..."):
                text = ""
                for pdf in pdf_docs:
                    with pdfplumber.open(pdf) as f:
                        for page in f.pages:
                            t = page.extract_text()
                            if t: text += t
                    upload_to_drive(pdf.name, pdf.name)
                
                memory = get_global_memory()
                for pdf in pdf_docs: memory.files.insert(0, {"name": pdf.name})
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                get_vector_store(chunks)
                st.success("Updated!")
                time.sleep(1)
                st.rerun()

# ==========================================
# PAGE 3: ABOUT
# ==========================================
if selected == "About":
    st.title("About")
    st.info("Built with Streamlit, OpenAI, and Google Gemini.")
