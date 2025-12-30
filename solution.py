import streamlit as st
import os

# ==========================================
# 0. THEME CONFIG (Standard Light Mode)
# ==========================================
st.set_page_config(page_title="CampusMind AI", page_icon="🎓", layout="wide")

# ==========================================
# 1. IMPORTS
# ==========================================
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests
import pdfplumber
import io
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate 
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ==========================================
# 2. CLEAN PROFESSIONAL CSS (Black Text)
# ==========================================
def inject_light_mode_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        /* --- 1. FORCE WHITE BACKGROUND (Fail-Safe) --- */
        [data-testid="stAppViewContainer"], .stApp {
            background: #ffffff !important;
            color: #1a1a1a !important; /* DARK BLACK TEXT */
        }
        [data-testid="stHeader"] { background: rgba(255,255,255,0.9) !important; }
        [data-testid="stSidebar"] { 
            background: #f8f9fa !important; 
            border-right: 1px solid #e0e0e0;
        }

        /* --- 2. TEXT VISIBILITY (Black Text Everywhere) --- */
        h1, h2, h3, h4, h5, h6 { color: #111111 !important; font-weight: 800 !important; }
        p, div, span, label { color: #333333 !important; }
        
        /* --- 3. INPUT FIELDS (Clean Borders) --- */
        .stTextInput input {
            color: #000000 !important;
            background: #ffffff !important;
            border: 1px solid #ced4da !important;
            border-radius: 8px !important;
            padding: 12px !important;
        }
        .stTextInput input:focus {
            border-color: #00C853 !important;
            box-shadow: 0 0 0 2px rgba(0, 200, 83, 0.2);
        }

        /* --- 4. CARDS (Shadows instead of Transparency) --- */
        .info-card {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 15px;
            transition: transform 0.2s;
        }
        .info-card:hover { transform: translateY(-3px); border-color: #00C853; }

        /* --- 5. AUDIO WIDGET FIX --- */
        /* Forces the audio widget to look correct on white background */
        div[data-testid="stAudioInput"] {
            margin-top: 5px;
        }
        div[data-testid="stAudioInput"] button {
            background-color: #f1f3f4 !important;
            color: #00C853 !important;
            border: 1px solid #e0e0e0 !important;
            width: 50px; height: 50px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
        }
        div[data-testid="stAudioInput"] button:hover {
            background-color: #e8f5e9 !important;
            transform: scale(1.05);
        }

        /* --- 6. ANSWER BOX --- */
        .answer-box-container {
            background: #e8f5e9; /* Light Green Bg */
            border-radius: 12px;
            border: 1px solid #00C853;
            padding: 25px; margin-top: 20px;
            box-shadow: 0 4px 15px rgba(0, 200, 83, 0.1);
        }
        .answer-title { color: #1b5e20 !important; font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; }
        .answer-content { font-size: 16px; line-height: 1.6; color: #1a1a1a !important; margin-top: 10px; }

        /* --- 7. HERO TEXT --- */
        .hero-title {
            font-family: 'Inter', sans-serif;
            font-size: 48px; font-weight: 800;
            background: -webkit-linear-gradient(45deg, #1a1a1a, #43a047);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .hero-badge {
            background: #e8f5e9; color: #2e7d32 !important;
            font-weight: 700; padding: 5px 12px; border-radius: 15px;
            font-size: 12px; border: 1px solid #c8e6c9;
            display: inline-block; margin-bottom: 15px;
        }
        
        /* HIDE FOOTER */
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

inject_light_mode_css()

# ==========================================
# 3. SECRETS & SETUP
# ==========================================
try:
    if "OPENAI_API_KEY" in st.secrets: 
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 
except: pass

# ==========================================
# 4. LOGIC
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

def get_valid_gemini_model():
    return "models/gemini-1.5-flash"

def transcribe_audio_gemini(audio_bytes):
    try:
        model_name = get_valid_gemini_model()
        model = genai.GenerativeModel(model_name)
        safety = {HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE}
        response = model.generate_content(
            ["Transcribe this audio exactly. Output only the English text.", {"mime_type": "audio/wav", "data": audio_bytes}],
            safety_settings=safety
        )
        return response.text
    except: return ""

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
# 5. UI LAYOUT
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=64)
    st.markdown("### CampusMind")
    st.caption("HYBRID AI CORE")
    
    selected = option_menu(
        "Navigation", ["Student Chat", "Admin Portal", "About"],
        icons=['chat-dots', 'cloud-upload', 'info-circle'],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"background-color": "transparent", "padding": "0"},
            "icon": {"color": "#555", "font-size": "16px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "6px 0px", "color": "#333"},
            "nav-link-selected": {"background-color": "#e8f5e9", "color": "#1b5e20"},
        }
    )

# --- STUDENT CHAT ---
if selected == "Student Chat":
    st.markdown("""
    <div style="text-align:center; padding: 40px 0;">
        <span class="hero-badge">⚡ ONLINE · 24/7</span>
        <h1 class="hero-title">CampusMind AI</h1>
        <p style="font-size: 18px; color: #555;">Voice-first campus intelligence.</p>
    </div>
    """, unsafe_allow_html=True)

    # Recent Circulars
    memory = get_global_memory()
    if memory.files:
        st.markdown("##### Recent Circulars")
        cols = st.columns(3)
        for i, f in enumerate(memory.files[:3]):
            with cols[i]:
                st.markdown(f"""
                <div class="info-card">
                    <div style="color:#00C853; font-size:11px; font-weight:800; letter-spacing:1px;">NEW UPLOAD</div>
                    <div style="color:#111; font-weight:600; font-size:14px; margin-top:5px;">{f['name'][:30]}...</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 💬 Ask Question")

    # --- INPUT ---
    c1, c2 = st.columns([1, 10], vertical_alignment="bottom")
    
    voice_query = ""
    with c1:
        audio_value = st.audio_input("Mic", label_visibility="collapsed")
        if audio_value:
            with st.spinner(" "):
                voice_query = transcribe_audio_gemini(audio_value.read())
    
    with c2:
        default_val = voice_query if voice_query else ""
        user_input = st.text_input("Message", value=default_val, placeholder="Ask about exams, fees, circulars...", label_visibility="collapsed")

    final_question = voice_query if voice_query else user_input

    # --- ANSWER ---
    col_left, col_right = st.columns([7, 3])
    if final_question:
        if "last_answered" not in st.session_state: st.session_state.last_answered = ""
        
        if st.session_state.last_answered != final_question:
            with col_left:
                with st.spinner("Searching documents..."):
                    try:
                        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                        if os.path.exists("faiss_index"):
                            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                            docs = new_db.similarity_search(final_question, k=10)
                            chain = get_conversational_chain()
                            res = chain.invoke({"input_documents": docs, "question": final_question}, return_only_outputs=True)
                            
                            st.session_state.last_answered = final_question
                            response = res['output_text']
                            
                            st.session_state.chat_history.append({"role": "User", "text": final_question})
                            st.session_state.chat_history.append({"role": "AI", "text": response})
                            
                            st.markdown(f"""
                            <div class="answer-box-container">
                                <div class="answer-title">🤖 CampusMind Answer</div>
                                <div class="answer-content">{response}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else: st.warning("Knowledge base empty.")
                    except: st.error("Context not found.")
    
    # --- HISTORY ---
    with col_right:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        st.markdown("<div style='padding:15px; background:#f8f9fa; border-radius:10px; border:1px solid #eee;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:12px; color:#555; margin-bottom:10px;'>HISTORY</div>", unsafe_allow_html=True)
        if st.session_state.chat_history:
            for item in reversed(st.session_state.chat_history[-4:]): 
                lbl = "You" if item["role"] == "User" else "AI"
                clr = "#00C853" if lbl == "AI" else "#333"
                st.markdown(f"<div style='font-size:12px; margin-bottom:8px; border-bottom:1px solid #eee; padding-bottom:5px;'><b style='color:{clr}'>{lbl}</b>: {item['text'][:40]}...</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- ADMIN ---
if selected == "Admin Portal":
    c1, c2 = st.columns([3, 7]) 
    with c1:
        if lottie_admin: st_lottie(lottie_admin, height=150)
    with c2:
        st.title("Admin Upload")
        st.markdown('<p style="color:#555;">Upload PDFs to the Knowledge Base.</p>', unsafe_allow_html=True)

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Upload to Cloud"):
        if pdf_docs:
            with st.status("Uploading...", expanded=True):
                text = ""
                for pdf in pdf_docs:
                    with pdfplumber.open(pdf) as f:
                        for page in f.pages:
                            t = page.extract_text()
                            if t: text += t
                    upload_to_drive(pdf.name, pdf.name)
                
                memory = get_global_memory()
                for pdf in pdf_docs: memory.files.insert(0, {"name": pdf.name, "id": "local_upload"})
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                get_vector_store(chunks)
                st.success("Success!")
                time.sleep(1)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- ABOUT ---
if selected == "About":
    st.title("About")
    st.markdown("""
    <div class="info-card">
        <h3>CampusMind AI</h3>
        <p>Built for the GDG Hackathon using Google Gemini + OpenAI.</p>
    </div>
    """, unsafe_allow_html=True)
