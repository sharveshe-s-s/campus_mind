import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from streamlit_mic_recorder import mic_recorder
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

# OpenAI Client
from openai import OpenAI

# ==========================================
# 0. THEME ENGINE (Force Dark Mode)
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
    if "OPENAI_API_KEY" in st.secrets: os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 
except FileNotFoundError: st.error("🚨 Secrets file not found!")

# ==========================================
# 2. HACKATHON WINNING CSS (STRICT ENFORCEMENT)
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

    /* --- CENTERED HERO TITLE (No Image) --- */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 60px 0 40px 0; /* More padding for cleaner look */
    }
    .shimmer-text {
        font-weight: 800;
        font-size: 64px; /* Larger Text */
        background: linear-gradient(120deg, #ffffff 30%, #00ffc3 50%, #00C853 70%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 6s linear infinite;
        text-shadow: 0 0 30px rgba(0, 200, 83, 0.2);
        margin: 15px 0;
        line-height: 1.1;
    }
    @keyframes shine { to { background-position: 200% center; } }

    .hero-tagline { font-size: 20px; color: rgba(235, 241, 255, 0.9); }
    
    .hero-badge {
        display: inline-flex; align-items: center; gap: 8px; padding: 8px 20px;
        border-radius: 999px; background: rgba(0, 200, 83, 0.15);
        border: 1px solid rgba(0, 255, 140, 0.3); font-size: 13px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em; color: #00ffc3;
    }

    /* NAVIGATION PILLS */
    .nav-link {
        border-radius: 8px !important; margin: 4px 0 !important;
        font-size: 15px !important; font-weight: 500 !important; color: #c0c7df !important;
        transition: all 0.2s ease-out !important;
    }
    .nav-link:hover { background: rgba(255, 255, 255, 0.08) !important; color: #fff !important; }
    .nav-link-selected {
        background: linear-gradient(135deg, #00C853, #009624) !important;
        color: #ffffff !important; box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
    }

    /* INPUTS */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.05) !important; color: #fff !important;
        border-radius: 12px; padding: 16px 20px 16px 50px; font-size: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.2s ease;
    }
    .stTextInput input:focus {
        border-color: #00C853 !important; background: rgba(0, 200, 83, 0.05) !important;
        box-shadow: 0 0 0 3px rgba(0, 200, 83, 0.25);
    }

    /* MIC BUTTON */
    div[data-testid="stButton"] button {
        border-radius: 12px !important; width: 54px; height: 54px;
        background: linear-gradient(135deg, #00C853, #009624);
        border: none; color: #fff; font-size: 24px;
        box-shadow: 0 6px 15px rgba(0, 200, 83, 0.3);
    }
    div[data-testid="stButton"] button:hover { transform: translateY(-2px); }

    /* --- STRICT BUTTON SQUASH FIX --- */
    /* This targets ALL buttons in the Admin portal specifically via the custom class below */
    .stButton button {
        white-space: nowrap !important;
        width: auto !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* PROCESS BUTTON STYLE */
    .stButton button.process-btn {
        min-width: 300px !important; /* MASSIVE MIN-WIDTH to force rectangle */
        padding: 14px 40px !important; 
        font-size: 16px; font-weight: 700;
        background: linear-gradient(135deg, #00C853, #00e676); color: white;
        border-radius: 12px !important; border: none;
        box-shadow: 0 8px 25px rgba(0, 200, 83, 0.3);
    }
    .stButton button.process-btn:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0, 200, 83, 0.4); }

    /* GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(25px);
        border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.08); padding: 24px;
        transition: transform 0.2s ease;
        color: #ffffff !important; 
    }
    .glass-card:hover { transform: translateY(-5px); border-color: rgba(0, 200, 83, 0.4); }

    /* ANSWER BOX (Self-contained) */
    .answer-box-container {
        background: rgba(0, 200, 83, 0.04);
        border-radius: 16px;
        border: 2px solid #00C853; 
        padding: 24px;
        margin-top: 30px;
        color: #ffffff !important;
        box-shadow: 0 0 50px rgba(0, 200, 83, 0.1);
        position: relative;
        word-wrap: break-word;
    }
    .answer-title { color: #00ffc3; font-size: 20px; font-weight: 800; display: flex; align-items: center; gap: 12px; }
    .answer-content { font-size: 17px; line-height: 1.7; margin-top: 15px; color: #eef2f6; }

    /* HISTORY */
    .history-card {
        background: rgba(255, 255, 255, 0.02); border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08); padding: 16px;
        max-height: 350px; overflow-y: auto;
    }
    .history-item {
        padding: 12px 16px; background: rgba(255, 255, 255, 0.04);
        border-radius: 10px; margin-bottom: 10px; font-size: 14px;
    }
    
    .chip {
        display: inline-flex; align-items: center; padding: 6px 14px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.1); font-size: 13px; color: #fff;
        gap: 8px; background: rgba(255,255,255,0.05);
    }
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

def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

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

def get_recent_circulars():
    try:
        if "gcp_service_account" not in st.secrets: return []
        key_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(key_dict, scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=creds)
        query = f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"
        results = service.files().list(q=query, pageSize=3, fields="files(id, name, createdTime)", orderBy="createdTime desc", supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
        return results.get('files', [])
    except: return []

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
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

# 3b. SESSION STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

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
    
    # --- HERO SECTION (NO IMAGE, CENTERED) ---
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚡ Campus-ready · 24/7</div>
        <h1 class="shimmer-text">CampusMind AI</h1>
        <p class="hero-tagline">Ask about exams, circulars, or anything on campus — get instant, tailored answers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # --- RECENT UPDATES ---
    st.markdown("##### <span style='font-weight:700; color:#fff;'>Recent Circulars</span>", unsafe_allow_html=True)
    with st.spinner("Syncing latest updates..."):
        recent_files = get_recent_circulars()
        
    if recent_files:
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for i, file in enumerate(recent_files):
            fname = file.get('name', 'Untitled Circular')
            with cols[i]:
                # Force white color on filename via inline CSS
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color: #00ffc3; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;">New Circular</div>
                    <div style="font-size: 15px; font-weight: 600; color: #ffffff !important; line-height: 1.4; word-wrap: break-word;">{fname[:50]}...</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recent circulars uploaded yet. Try asking general campus questions!")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CHAT UI ---
    st.markdown("##### <span style='font-weight:700; color:#fff;'>💬 Ask Anything</span>", unsafe_allow_html=True)
    
    left_col, right_col = st.columns([7, 3])

    with left_col:
        with st.container():
            c_mic, c_input = st.columns([1, 8])
            with c_mic:
                audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format="webm", just_once=True)
            with c_input:
                voice_text = ""
                if audio:
                    with st.spinner("Transcribing..."):
                        try:
                            audio_file = io.BytesIO(audio['bytes'])
                            audio_file.name = "audio.webm"
                            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                            transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                            voice_text = transcript.text
                        except: pass
                default_val = voice_text if voice_text else ""
                user_question = st.text_input("Search", value=default_val, placeholder="Ex: When are the exams? What does the latest circular say?", label_visibility="collapsed")

    with right_col:
        st.markdown("<div class='history-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:14px; color:#fff; margin-bottom:12px;'>RECENT TURNS</div>", unsafe_allow_html=True)
        if st.session_state.chat_history:
            for item in st.session_state.chat_history[-3:]:
                label = "You" if item["role"] == "User" else "AI"
                st.markdown(f"<div class='history-item'><span class='label'>{label}</span><br>{item['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:13px;color:rgba(255,255,255,0.6);'>Your conversation history will appear here.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if user_question:
        with st.spinner("🧠 Analyzing your question..."):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                if os.path.exists("faiss_index"):
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    
                    res = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    full_response = res['output_text']
                    
                    st.session_state.chat_history.append({"role": "User", "text": user_question})
                    st.session_state.chat_history.append({"role": "AI", "text": full_response})

                    # --- BOX FIX: SINGLE PLACEHOLDER UPDATE ---
                    answer_placeholder = st.empty()
                    accumulated_text = ""
                    words = full_response.split(" ")
                    
                    for word in words:
                        accumulated_text += word + " "
                        answer_placeholder.markdown(f"""
                        <div class="answer-box-container">
                            <div class="answer-title">
                                <span style="font-size: 24px;">🤖</span><span>CampusMind Answer</span>
                            </div>
                            <div class="answer-sub">Context-aware · From your uploaded circulars</div>
                            <hr style="border-color: rgba(0, 200, 83, 0.3); margin: 16px 0;">
                            <div class="answer-content">{accumulated_text}▌</div>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.04)
                    
                    answer_placeholder.markdown(f"""
                    <div class="answer-box-container">
                        <div class="answer-title">
                            <span style="font-size: 24px;">🤖</span><span>CampusMind Answer</span>
                        </div>
                        <div class="answer-sub">Context-aware · From your uploaded circulars</div>
                        <hr style="border-color: rgba(0, 200, 83, 0.3); margin: 16px 0;">
                        <div class="answer-content">{full_response}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.warning("⚠️ Knowledge base empty. Please upload circulars in the Admin Portal.")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# PAGE 2: ADMIN PORTAL
# ==========================================
if selected == "Admin Portal":
    # --- FIX: HUGE COLUMN WIDTH FOR BUTTON ---
    # Changing layout to [3, 7] to give button plenty of room
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
    
    # Button with specific ID for styling
    if st.button("Process & Upload", key="process_btn", help="Click to process and upload documents"):
        if pdf_docs:
            with st.status("Processing...", expanded=True):
                text = ""
                for pdf in pdf_docs:
                    with pdfplumber.open(pdf) as pdf_file:
                        for page in pdf_file.pages:
                            t = page.extract_text()
                            if t: text += t
                    with open(pdf.name, "wb") as f: f.write(pdf.getbuffer())
                    upload_to_drive(pdf.name, pdf.name)
                    if os.path.exists(pdf.name): os.remove(pdf.name)
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(text)
                get_vector_store(chunks)
                st.success("✅ Knowledge base updated successfully!")
                time.sleep(1)
                st.rerun()
        else:
            st.warning("Please select at least one PDF file.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Inject Class
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
            <div class="chip">🔍 FAISS Vector DB</div>
            <div class="chip">☁️ Google Drive API</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
