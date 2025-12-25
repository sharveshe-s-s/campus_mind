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
    """
    Writes a config file to force Streamlit into Dark Mode.
    """
    config_dir = ".streamlit"
    config_path = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        
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
        with open(config_path, "w") as f:
            f.write(config_content)
        st.rerun()

force_dark_mode()

# ==========================================
# 1. SETUP & SECRETS
# ==========================================
st.set_page_config(page_title="CampusMind AI", page_icon="🎓", layout="wide")

try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 
except FileNotFoundError:
    st.error("🚨 Secrets file not found!")

# ==========================================
# 2. HACKATHON WINNING CSS (UPDATED)
# ==========================================
st.markdown("""
<style>
    /* GLOBAL LAYOUT TUNING + PAGE TRANSITION */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 1180px !important;
        animation: pageFadeIn 0.45s ease-out;
    }
    @keyframes pageFadeIn {
        from { opacity: 0; transform: translateY(6px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* APP BACKGROUND – DEEP NEON CAMPUS */
    .stApp {
        background: radial-gradient(circle at top left, #182848 0, #020024 35%, #000000 100%);
        color: #f5f7fb !important;
    }

    [data-testid="stMain"] {
        background: transparent !important;
    }

    /* SIDEBAR – GLASS PANEL */
    section[data-testid="stSidebar"] {
        background: radial-gradient(circle at top, rgba(0, 200, 83, 0.12), rgba(0, 0, 0, 0.82));
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(18px);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }

    .sidebar-title {
        font-weight: 700;
        font-size: 22px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #E0FFEE;
    }
    .sidebar-subtitle {
        font-size: 11px;
        color: rgba(255,255,255,0.65);
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    /* SHIMMER TITLE IN HERO */
    .shimmer-text {
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff 0%, #77ffb5 40%, #60a5ff 60%, #ffffff 100%);
        background-size: 220% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
    }
    @keyframes shine {
        to { background-position: 220% center; }
    }

    .hero-tagline {
        font-size: 16px;
        color: rgba(235, 241, 255, 0.85);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(0, 200, 83, 0.18), rgba(0, 200, 150, 0.04));
        border: 1px solid rgba(0, 255, 140, 0.35);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #9fffdc;
        margin-bottom: 8px;
    }

    /* OPTION MENU – NEON PILLS */
    .css-1d391kg, .css-12oz5g7 {
        background: transparent !important;
    }
    .nav-link {
        border-radius: 999px !important;
        margin: 4px 0 !important;
        padding: 0.5rem 0.85rem !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #d0d7ff !important;
        border: 1px solid transparent !important;
        transition: all 0.18s ease-out !important;
    }
    .nav-link:hover {
        background: radial-gradient(circle at top left, rgba(0, 200, 83, 0.18), rgba(0, 0, 0, 0.4));
        border-color: rgba(0, 255, 160, 0.28) !important;
        transform: translateX(2px);
    }
    .nav-link-selected {
        background: linear-gradient(120deg, #00C853, #009624) !important;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(0, 200, 83, 0.55);
    }

    /* TEXT INPUT – MODERN SEARCH BAR */
    .stTextInput > div > div {
        position: relative;
    }
    .stTextInput input {
        background: rgba(4, 8, 20, 0.96) !important;
        color: #f1f5ff !important;
        border-radius: 999px;
        padding: 12px 18px 12px 46px;
        font-size: 16px;
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.6), 0 18px 40px rgba(0, 0, 0, 0.7);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }
    .stTextInput input::placeholder {
        color: rgba(176, 189, 220, 0.7) !important;
        font-size: 14px;
    }
    .stTextInput input:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.85);
    }
    .stTextInput input:focus {
        border-color: #00C853 !important;
        box-shadow: 0 0 0 1px rgba(0, 200, 83, 0.7), 0 18px 40px rgba(0, 200, 83, 0.22);
        background: rgba(7, 14, 36, 0.98) !important;
    }
    .stTextInput > div > div:before {
        content: "🔍";
        position: absolute;
        left: 14px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 17px;
        opacity: 0.85;
    }

    /* MIC BUTTON – GLOWING NEON CIRCLE */
    div[data-testid="stButton"] button {
        border-radius: 50% !important;
        width: 44px;
        height: 44px;
        padding: 0;
        background: radial-gradient(circle at 30% 20%, #00ffc6 0, #00C853 40%, #006622 85%);
        border: 1px solid rgba(0, 255, 170, 0.7);
        color: #02030a;
        font-size: 22px;
        transition: all 0.22s ease;
        position: relative;
        left: 4px;
        top: 4px;
        box-shadow: 0 0 0 0 rgba(0, 255, 170, 0.5);
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-1px) scale(1.05);
        box-shadow: 0 0 18px 4px rgba(0, 255, 170, 0.55);
    }
    div[data-testid="stButton"] button:active {
        transform: scale(0.97) translateY(1px);
        box-shadow: 0 0 8px 2px rgba(0, 200, 120, 0.7);
    }

    /* PROCESS BUTTON (ADMIN) */
    .stButton button.process-btn {
        border-radius: 999px !important;
        width: auto !important;
        height: auto !important;
        white-space: nowrap !important;
        min-width: 190px;
        padding: 11px 32px !important;
        background: linear-gradient(120deg, #00C853, #00e676, #00C853);
        background-size: 220% auto;
        color: white;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 12px 30px rgba(0, 200, 83, 0.35);
        border: none;
        transition: all 0.22s ease, background-position 0.4s ease;
    }
    .stButton button.process-btn:hover {
        background-position: 100% center;
        transform: translateY(-2px);
        box-shadow: 0 18px 40px rgba(0, 200, 83, 0.6);
    }
    .stButton button.process-btn:active {
        transform: translateY(0px) scale(0.99);
        box-shadow: 0 6px 18px rgba(0, 150, 63, 0.7);
    }

    /* GLASS CARDS */
    .glass-card {
        background: radial-gradient(circle at top left, rgba(255, 255, 255, 0.12), rgba(3, 7, 18, 0.95));
        backdrop-filter: blur(16px);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.10);
        padding: 18px 18px 16px 18px;
        margin-bottom: 15px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.9);
        position: relative;
        overflow: hidden;
    }
    .glass-card:before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 10% 0%, rgba(0, 255, 170, 0.18), transparent 55%);
        opacity: 0.9;
        pointer-events: none;
    }
    .glass-card h3, .glass-card p, .glass-card li {
        color: #f5f7fb !important;
    }
    .glass-card-accent {
        width: 42px;
        height: 3px;
        border-radius: 999px;
        background: linear-gradient(90deg, #00f5a0, #00d9f5);
        margin-bottom: 8px;
    }

    .section-label {
        font-size: 14px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: rgba(196, 210, 255, 0.88);
    }

    /* ANSWER BOX */
    .answer-box-container {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.88), rgba(6, 20, 36, 0.96));
        border-radius: 18px;
        border: 1px solid rgba(0, 255, 170, 0.4);
        padding: 22px 22px 12px 22px;
        margin-top: 24px;
        color: #ffffff !important;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.9);
        position: relative;
        overflow: hidden;
    }
    .answer-box-container:before {
        content: "";
        position: absolute;
        inset: -40%;
        background: radial-gradient(circle at 0% 0%, rgba(0, 255, 170, 0.38), transparent 55%);
        opacity: 0.6;
        mix-blend-mode: screen;
        pointer-events: none;
    }
    .answer-title {
        color: #9effc9;
        font-size: 18px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .answer-title span.icon {
        font-size: 18px;
    }
    .answer-sub {
        font-size: 12px;
        color: rgba(210, 226, 255, 0.78);
        text-transform: uppercase;
        letter-spacing: 0.16em;
    }

    /* CHAT HISTORY PANEL */
    .history-card {
        background: radial-gradient(circle at top right, rgba(0, 255, 170, 0.12), rgba(5, 7, 20, 0.95));
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        padding: 10px 12px;
        font-size: 12px;
        max-height: 260px;
        overflow-y: auto;
        box-shadow: 0 14px 32px rgba(0,0,0,0.8);
    }
    .history-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: rgba(186, 205, 255, 0.9);
        margin-bottom: 4px;
    }
    .history-item {
        padding: 6px 8px;
        border-radius: 10px;
        background: rgba(0,0,0,0.42);
        margin-top: 4px;
        color: #dfe7ff;
    }
    .history-item span.label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(152, 214, 255, 0.9);
    }

    /* SMALL BADGE CHIPS */
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.16);
        font-size: 11px;
        color: rgba(225,231,255,0.9);
        gap: 6px;
        background: rgba(10,16,32,0.9);
    }

    .glass-card ul {
        padding-left: 1.1rem;
    }
    .glass-card li {
        margin-bottom: 4px;
        font-size: 14px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def stream_text(text):
    """Yields text character by character/word by word to simulate typing."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

def upload_to_drive(file_path, file_name):
    try:
        if "gcp_service_account" not in st.secrets:
            return "Error: Secrets missing"
        key_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
        media = MediaFileUpload(file_path, mimetype='application/pdf')
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        return file.get('id')
    except Exception as e:
        return f"Error: {e}"

def get_recent_circulars():
    try:
        if "gcp_service_account" not in st.secrets:
            return []
        key_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive']
        )
        service = build('drive', 'v3', credentials=creds)
        query = f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            pageSize=3,
            fields="files(id, name, createdTime)",
            orderBy="createdTime desc",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        return results.get('files', [])
    except:
        return []

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

# ASSETS
lottie_student_ai = load_lottieurl("https://lottie.host/99977635-960c-4432-9987-440077149909/99977635-960c-4432-9987-440077149909.json")
lottie_admin = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")

# ------------------------------------------
# 3b. SESSION STATE FOR CHAT HISTORY (UI ONLY)
# ------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "User"/"AI", "text": "..."}

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=52)
    st.markdown('<div class="sidebar-title">CampusMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Smart Campus Copilot</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected = option_menu(
        "Navigation",
        ["Student Chat", "Admin Portal", "About"],
        icons=['chat-dots', 'cloud-upload', 'info-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "icon": {"color": "#7df8c9", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "2px 0px",
            },
            "nav-link-selected": {"background-color": "#00C853"},
        }
    )
    st.markdown("---")
    st.caption("v2.0 · Hackathon Edition")

# ==========================================
# PAGE 1: STUDENT CHAT
# ==========================================
if selected == "Student Chat":
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if lottie_student_ai:
            st_lottie(lottie_student_ai, height=220, key="hero_anim")
    with col2:
        st.markdown(
            "<div style='display:flex;flex-direction:column;justify-content:center;height:220px;'>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='hero-badge'>⚡ Campus-ready · 24/7</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<h1 class='shimmer-text' style='font-size: 46px; margin-bottom: 6px;'>CampusMind AI</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p class='hero-tagline'>Ask about exams, circulars, or anything on campus — get instant, tailored answers in seconds.</p>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown("##### <span class='section-label'>Recent Circulars</span>", unsafe_allow_html=True)
    with st.spinner("Syncing latest updates..."):
        recent_files = get_recent_circulars()
        
    if recent_files:
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for i, file in enumerate(recent_files):
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="glass-card-accent"></div>
                    <div style="color: #9fffdc; font-weight: 600; font-size: 14px; text-transform: uppercase; letter-spacing: 0.12em;">
                        New Circular
                    </div>
                    <div style="margin-top: 6px; font-size: 15px; color: #ffffff;">
                        {file['name'][:40]}...
                    </div>
                    <div style="font-size: 12px; color: #a3b3d6; margin-top: 8px;">
                        🎧 Tap the mic and ask: <span style="font-style:italic;">“What is this circular about?”</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recent circulars uploaded yet. Try asking general campus questions!")

    st.markdown("---")

    st.markdown("##### 💬 Ask Anything")
    
    # Chat input + history layout
    left_col, right_col = st.columns([3, 1])

    with left_col:
        with st.container():
            c_mic, c_input = st.columns([1, 11])
            
            with c_mic:
                audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format="webm", just_once=True)
                
            with c_input:
                voice_text = ""
                if audio:
                    with st.spinner("Transcribing your voice..."):
                        try:
                            audio_file = io.BytesIO(audio['bytes'])
                            audio_file.name = "audio.webm"
                            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                            transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                            voice_text = transcript.text
                        except:
                            pass
                
                default_val = voice_text if voice_text else ""
                user_question = st.text_input(
                    "Search",
                    value=default_val,
                    placeholder="Ex: When are the exams? What does the latest circular say?",
                    label_visibility="collapsed"
                )

    # Conversation history (right column) – UI only
    with right_col:
        st.markdown("<div class='history-card'>", unsafe_allow_html=True)
        st.markdown("<div class='history-title'>Recent Turns</div>", unsafe_allow_html=True)
        if st.session_state.chat_history:
            # show last 4 turns (user+ai combined)
            for item in st.session_state.chat_history[-4:]:
                label = "You" if item["role"] == "User" else "AI"
                st.markdown(
                    f"<div class='history-item'><span class='label'>{label}</span><br>{item['text']}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div style='font-size:12px;color:rgba(200,213,255,0.8);margin-top:4px;'>Start asking questions to see them appear here.</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    if user_question:
        with st.spinner("🧠 Analyzing your question..."):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                if os.path.exists("faiss_index"):
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    response = chain.invoke(
                        {"input_documents": docs, "question": user_question},
                        return_only_outputs=True
                    )
                    
                    full_response = response['output_text']

                    # update history (UI context only)
                    st.session_state.chat_history.append({"role": "User", "text": user_question})
                    st.session_state.chat_history.append({"role": "AI", "text": full_response})

                    st.markdown("""
                        <div class="answer-box-container">
                            <div class="answer-title">
                                <span class="icon">🤖</span>
                                <span>CampusMind Answer</span>
                            </div>
                            <div class="answer-sub">Context-aware · From your uploaded circulars</div>
                            <hr style="border-color: rgba(255,255,255,0.18); margin: 12px 0 4px 0;">
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.write_stream(stream_text(full_response))
                    
                else:
                    st.warning("⚠️ Knowledge base empty. Please upload circulars in the Admin Portal to unlock campus‑aware answers.")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# PAGE 2: ADMIN PORTAL
# ==========================================
if selected == "Admin Portal":
    c1, c2 = st.columns([1, 3])
    with c1:
        if lottie_admin:
            st_lottie(lottie_admin, height=160)
    with c2:
        st.title("Admin Upload")
        st.markdown(
            '<p style="color:#c7d2ff;font-size:14px;">Securely upload circulars and instantly refresh what students can ask CampusMind.</p>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="chip">📁 <span>Upload PDF circulars</span></div><br><br>',
        unsafe_allow_html=True
    )
    pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True, type=['pdf'])
    
    st.write("")
    
    if st.button("Process & Upload", key="process_btn", help="Click to process and upload documents"):
        if pdf_docs:
            with st.status("Processing and training CampusMind...", expanded=True):
                text = ""
                for pdf in pdf_docs:
                    with pdfplumber.open(pdf) as pdf_file:
                        for page in pdf_file.pages:
                            t = page.extract_text()
                            if t:
                                text += t
                    with open(pdf.name, "wb") as f:
                        f.write(pdf.getbuffer())
                    upload_to_drive(pdf.name, pdf.name)
                    if os.path.exists(pdf.name):
                        os.remove(pdf.name)
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(text)
                get_vector_store(chunks)
                st.success("✅ Circulars uploaded and knowledge base updated!")
                time.sleep(1)
                st.rerun()
        else:
            st.warning("Please select at least one PDF file before processing.")
    st.markdown('</div>', unsafe_allow_html=True)

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
        <div class="glass-card-accent"></div>
        <h3 style="margin-bottom:4px;">CampusMind AI</h3>
        <p style="color:#d4e3ff;font-size:14px;margin-bottom:14px;">
            A next‑gen smart campus assistant built for the 2024 Innovation Hackathon, designed to feel like a friendly AI desk in your pocket.
        </p>
        <ul>
            <li><b>Frontend:</b> Streamlit with custom theme and glassmorphism.</li>
            <li><b>AI Brain:</b> OpenAI GPT‑4o‑Mini with LangChain QA pipeline.</li>
            <li><b>Vector DB:</b> FAISS for semantic circular search.</li>
            <li><b>Storage:</b> Google Drive API for secure circular management.</li>
        </ul>
        <p style="color:#b5c2ff;font-size:13px;margin-top:10px;">
            Optimized for hackathon demos: fast onboarding, clean navigation, and visually striking dark mode for projector displays.
        </p>
    </div>
    """, unsafe_allow_html=True)
