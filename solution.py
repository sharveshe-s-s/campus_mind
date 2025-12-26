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
# 2. HACKATHON WINNING CSS (PREMIUM UI)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #050913; }
    ::-webkit-scrollbar-thumb { background: #00C853; border-radius: 10px; }

    /* LAYOUT & ANIMATION */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
        animation: pageFadeIn 0.6s ease-out;
    }
    @keyframes pageFadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stApp {
        background: radial-gradient(circle at top left, #1a2a4f 0, #050913 40%, #000000 100%);
        color: #f5f7fb !important;
    }
    [data-testid="stMain"] { background: transparent !important; }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background: rgba(5, 9, 19, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
    }
    .sidebar-title { font-weight: 800; font-size: 24px; color: #fff; letter-spacing: 0.05em; }
    .sidebar-subtitle { font-size: 12px; color: rgba(255,255,255,0.6); letter-spacing: 0.1em; text-transform: uppercase; }

    /* HERO */
    .hero-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center; padding: 40px 0 30px 0;
    }
    .shimmer-text {
        font-weight: 800; font-size: 56px;
        background: linear-gradient(120deg, #ffffff 30%, #00ffc3 50%, #00C853 70%);
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: shine 6s linear infinite; text-shadow: 0 0 30px rgba(0, 200, 83, 0.2);
        margin: 10px 0; line-height: 1.1;
    }
    @keyframes shine { to { background-position: 200% center; } }
    
    .hero-badge {
        display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px;
        border-radius: 8px; background: rgba(0, 200, 83, 0.15);
        border: 1px solid rgba(0, 255, 140, 0.3); font-size: 12px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em; color: #00ffc3;
    }

    /* GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(25px);
        border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 20px;
        transition: transform 0.2s ease; color: #ffffff !important; 
    }
    .glass-card:hover { transform: translateY(-5px); border-color: rgba(0, 200, 83, 0.4); }

    /* --- GEMINI-LIKE INPUT BAR --- */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important; color: #fff !important;
        border-radius: 25px; /* Rounded pill shape like Gemini */
        padding: 15px 25px; font-size: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.2s ease;
    }
    .stTextInput input:focus {
        border-color: #00C853 !important; background: rgba(0, 200, 83, 0.05) !important;
        box-shadow: 0 0 0 2px rgba(0, 200, 83, 0.25);
    }

    /* AUDIO WIDGET INTEGRATION */
    /* This makes the audio widget look like a button next to the text box */
    div[data-testid="stAudioInput"] {
        background: transparent !important;
        border: none !important;
        padding: 5px !important;
        margin-top: 5px !important;
    }
    div[data-testid="stAudioInput"] button {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        color: #00C853 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        display: flex; align-items: center; justify-content: center;
    }
    div[data-testid="stAudioInput"] button:hover {
        background: rgba(0, 200, 83, 0.2) !important;
        border-color: #00C853 !important;
    }

    /* ANSWER BOX */
    .answer-box-container {
        background: rgba(0, 200, 83, 0.04); border-radius: 16px; border: 1px solid rgba(0, 200, 83, 0.3);
        padding: 30px; margin-top: 30px; color: #ffffff !important;
        box-shadow: 0 0 60px rgba(0, 200, 83, 0.08); position: relative; word-wrap: break-word;
    }
    .answer-title { color: #00ffc3; font-size: 22px; font-weight: 700; display: flex; align-items: center; gap: 12px; margin-bottom: 15px;}
    .answer-content { font-size: 17px; line-height: 1.8; color: #eef2f6; }

    /* HISTORY */
    .history-card {
        background: rgba(255, 255, 255, 0.02); border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08); padding: 16px;
        max-height: 400px; overflow-y: auto;
    }
    .history-item {
        padding: 12px 16px; background: rgba(255, 255, 255, 0.04);
        border-radius: 8px; margin-bottom: 10px; font-size: 14px;
        border-left: 3px solid #00C853;
    }
    
    .chip {
        display: inline-flex; align-items: center; padding: 6px 14px; border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1); font-size: 13px; color: #fff;
        gap: 8px; background: rgba(255,255,255,0.05);
    }
    
    /* PROCESS BUTTON */
    .stButton button.process-btn {
        min-width: 100% !important; padding: 14px 40px !important; 
        font-size: 16px; font-weight: 700; background: linear-gradient(135deg, #00C853, #00e676); color: white;
        border: none; box-shadow: 0 8px 25px rgba(0, 200, 83, 0.3); border-radius: 8px;
    }
    .stButton button.process-btn:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0, 200, 83, 0.4); }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
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

# --- INTELLIGENT MODEL SELECTOR (SELF-HEALING) ---
def get_valid_gemini_model():
    """Finds a working model name available to your specific API key."""
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        preferred_order = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-pro",
            "models/gemini-1.5-pro-001"
        ]
        
        for model_name in preferred_order:
            if model_name in available_models: return model_name
        
        for model_name in available_models:
            if "1.5" in model_name: return model_name
            
        return "models/gemini-1.5-flash"
    except:
        return "models/gemini-1.5-flash"

# --- GOOGLE GEMINI AUDIO TRANSCRIPTION ---
def transcribe_audio_gemini(audio_bytes):
    try:
        model_name = get_valid_gemini_model()
        model = genai.GenerativeModel(model_name)
        safety = {HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE}
        
        # Audio input via new method
        response = model.generate_content(
            ["Transcribe this audio exactly. Output only the English text.", {"mime_type": "audio/wav", "data": audio_bytes}],
            safety_settings=safety
        )
        return response.text
    except Exception as e: 
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
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=64)
    st.markdown('<div class="sidebar-title">CampusMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Smart Campus Copilot</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected = option_menu(
        "Navigation", ["Student Chat", "Admin Portal", "About"],
        icons=['chat-dots', 'cloud-upload', 'info-circle'],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"background-color": "transparent", "padding": "0"},
            "icon": {"color": "#c0c7df", "font-size": "18px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "6px 0px"},
            "nav-link-selected": {"background-color": "#00C853"},
        }
    )
    st.markdown("---")
    st.caption("v2.0 · Hackathon Edition")

# ==========================================
# PAGE 1: STUDENT CHAT
# ==========================================
if selected == "Student Chat":
    
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚡ Campus-ready · 24/7</div>
        <h1 class="shimmer-text">CampusMind AI</h1>
        <p class="hero-tagline">Ask about exams, circulars, or anything on campus — get instant, tailored answers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Recent Circulars
    memory = get_global_memory()
    if memory.files:
        st.markdown("##### <span style='font-weight:700; color:#fff;'>Recent Circulars</span>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for i, f in enumerate(memory.files[:3]):
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color: #00ffc3; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">New Circular</div>
                    <div style="font-size: 15px; font-weight: 600; color: #ffffff !important; line-height: 1.4; word-wrap: break-word;">{f['name'][:50]}...</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("##### <span style='font-weight:700; color:#fff;'>💬 Ask Anything</span>", unsafe_allow_html=True)
    
    # --- GEMINI-STYLE INPUT BAR ---
    # We use columns to place the Mic next to the Text Box
    input_col1, input_col2 = st.columns([1, 10], vertical_alignment="bottom")
    
    voice_query = ""
    
    with input_col1:
        # Native Audio Input - Styled to look like a button
        audio_value = st.audio_input("Mic", label_visibility="collapsed")
        
        if audio_value:
            with st.spinner(" "):
                audio_bytes = audio_value.read()
                voice_query = transcribe_audio_gemini(audio_bytes)
    
    with input_col2:
        # Text Input
        default_val = voice_query if voice_query else ""
        user_input = st.text_input("Text Input", value=default_val, placeholder="Ask anything about the campus...", label_visibility="collapsed")

    # Main Logic
    final_question = voice_query if voice_query else user_input

    # Response Area (Two Columns: Answer | History)
    c_left, c_right = st.columns([7, 3])

    if final_question:
        if "last_answered" not in st.session_state: st.session_state.last_answered = ""
        
        if st.session_state.last_answered != final_question:
            with c_left:
                with st.spinner("🧠 Analyzing..."):
                    try:
                        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                        if os.path.exists("faiss_index"):
                            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                            docs = new_db.similarity_search(final_question, k=10)
                            chain = get_conversational_chain()
                            res = chain.invoke({"input_documents": docs, "question": final_question}, return_only_outputs=True)
                            
                            st.session_state.last_answered = final_question
                            response_text = res['output_text']
                            
                            # Add to History
                            st.session_state.chat_history.append({"role": "User", "text": final_question})
                            st.session_state.chat_history.append({"role": "AI", "text": response_text})
                            
                            st.markdown(f"""
                            <div class="answer-box-container">
                                <div class="answer-title">
                                    <span style="font-size: 24px;">🤖</span><span>CampusMind Answer</span>
                                </div>
                                <div style="font-size:14px; color:rgba(255,255,255,0.5);">Context-aware • From uploaded docs</div>
                                <hr style="border-color: rgba(0, 200, 83, 0.3); margin: 16px 0;">
                                <div class="answer-content">{response_text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Knowledge base empty. Please upload circulars in the Admin Portal.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    # History Sidebar (Right Column)
    with c_right:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True) # Spacer
        st.markdown("<div class='history-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:14px; color:#fff; margin-bottom:12px; letter-spacing:1px;'>RECENT TURNS</div>", unsafe_allow_html=True)
        
        if st.session_state.chat_history:
            for item in reversed(st.session_state.chat_history[-4:]): # Show last 4 items
                label = "You" if item["role"] == "User" else "AI"
                st.markdown(f"<div class='history-item'><div style='font-size:11px; color:#00C853; font-weight:bold; margin-bottom:4px;'>{label}</div>{item['text'][:80]}...</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:13px;color:rgba(255,255,255,0.6); font-style:italic;'>No history yet.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PAGE 2: ADMIN PORTAL
# ==========================================
if selected == "Admin Portal":
    c1, c2 = st.columns([3, 7]) 
    with c1:
        if lottie_admin: st_lottie(lottie_admin, height=180)
    with c2:
        st.title("Admin Upload")
        st.markdown('<p style="color:#c0c7df;font-size:16px;">Securely upload circulars and instantly refresh the AI\'s knowledge base.</p>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="chip">📁 <span>Upload PDF circulars</span></div><br><br>', unsafe_allow_html=True)
    pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True, type=['pdf'])
    
    st.write("")
    
    if st.button("Process & Upload", key="process_btn"):
        if pdf_docs:
            with st.status("Processing...", expanded=True):
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
                
                st.success("✅ Knowledge base updated successfully!")
                time.sleep(1)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Inject Button CSS
    st.markdown("""
    <script>
        const buttons = window.parent.document.querySelectorAll('button');
        buttons.forEach(btn => {
            if (btn.innerText === 'Process & Upload') {
                btn.classList.add('process-btn');
            }
        });
    </script>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 3: ABOUT
# ==========================================
if selected == "About":
    st.title("About")
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-bottom:12px; font-weight: 800;">CampusMind AI</h3>
        <p style="color:#c0c7df;font-size:15px;line-height:1.6;margin-bottom:20px;">
            A next‑gen smart campus assistant built for the Innovation Hackathon. It uses advanced AI to provide instant, accurate answers from official campus documents.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
            <div class="chip">💻 Streamlit Frontend</div>
            <div class="chip">🧠 OpenAI GPT‑4o</div>
            <div class="chip">🎤 Google Gemini Audio</div>
            <div class="chip">🔍 FAISS Vector DB</div>
            <div class="chip">☁️ Google Drive API</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
