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
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#161b22"
textColor = "#fafafa"
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
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 
except FileNotFoundError:
    st.error("🚨 Secrets file not found!")

# ==========================================
# 2. HACKATHON WINNING CSS
# ==========================================
st.markdown("""
<style>
    /* 1. BACKGROUND */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    /* 2. SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.4);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* 3. INPUT BOX (White Box, Black Text - High Visibility) */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 12px;
        padding: 15px;
        font-size: 16px;
        border: 2px solid #ccc;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextInput input:focus {
        border-color: #00C853 !important;
        box-shadow: 0 0 15px rgba(0, 200, 83, 0.5);
    }

    /* 4. MIC BUTTON */
    div[data-testid="stButton"] button {
        border-radius: 50%;
        width: 55px;
        height: 55px;
        background: rgba(0, 200, 83, 0.2);
        border: 2px solid #00C853;
        color: #00C853;
        font-size: 24px;
        transition: all 0.3s ease;
    }
    div[data-testid="stButton"] button:hover {
        background: #00C853;
        color: white;
        transform: scale(1.1);
    }
    
    /* 5. PROCESS BUTTON OVERRIDE (For Admin) */
    .stButton button.process-btn {
        width: auto;
        border-radius: 8px;
    }

    /* 6. GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* 7. QUICK ACTION PILLS */
    .quick-pill {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        padding: 5px 15px;
        border-radius: 20px;
        margin-right: 10px;
        font-size: 14px;
        border: 1px solid rgba(255,255,255,0.2);
        color: #ddd;
    }

    /* 8. ANSWER BOX */
    .answer-box {
        background: rgba(0, 0, 0, 0.6);
        border-left: 6px solid #00C853;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        color: #ffffff !important;
    }
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

# --- ASSETS ---
# New "Futuristic AI" Animation for Student Chat
lottie_student_ai = load_lottieurl("https://lottie.host/020cc52c-7472-4632-841f-82559b95427d/21H5gH1p7E.json") 
# Admin Laptop Animation
lottie_admin = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")

# ==========================================
# 4. SIDEBAR MENU
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
    selected = option_menu(
        "CampusMind",
        ["Student Chat", "Admin Portal", "About"],
        icons=['chat-dots', 'cloud-upload', 'info-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#00C853"},
        }
    )
    st.markdown("---")
    st.caption("Hackathon Edition v2.0")

# ==========================================
# PAGE 1: STUDENT CHAT (THE "BEST WEBSITE" LOOK)
# ==========================================
if selected == "Student Chat":
    
    # --- HERO SECTION (Split Layout) ---
    # Left: Cool Animation | Right: Welcome Text
    col_hero_1, col_hero_2 = st.columns([1, 2])
    
    with col_hero_1:
        if lottie_student_ai: 
            st_lottie(lottie_student_ai, height=200, key="ai_anim")
    
    with col_hero_2:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        st.markdown("<h1 style='font-size: 48px; margin-bottom: 0;'>CampusMind AI</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px; margin-top: 10px;'>
            <span style='background: #00C853; width: 10px; height: 10px; border-radius: 50%; display: inline-block;'></span>
            <span style='color: #bbb; font-size: 16px;'>System Online • Knowledge Base Active</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- RECENT UPDATES (Dashboard Widgets) ---
    st.subheader("📢 Live Circulars")
    
    with st.spinner("Syncing with Admin Office..."):
        recent_files = get_recent_circulars()
        
    if recent_files:
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for i, file in enumerate(recent_files):
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color: #00C853; font-weight: bold; font-size: 18px; margin-bottom: 5px;">📄 Update {i+1}</div>
                    <div style="font-size: 14px; color: white;">{file['name'][:25]}...</div>
                    <div style="font-size: 12px; color: #888; margin-top: 5px;">Click mic to ask about this</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recent circulars found on Drive.")

    # --- COMMAND CENTER (The Chat) ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # "Quick Prompts" Visuals
    st.markdown("""
    <div style="margin-bottom: 10px;">
        <span class="quick-pill">📅 Exams</span>
        <span class="quick-pill">🚌 Bus Routes</span>
        <span class="quick-pill">💰 Fee Structure</span>
        <span class="quick-pill">📝 Revaluation</span>
    </div>
    """, unsafe_allow_html=True)

    # Chat Interface Columns
    c_mic, c_input = st.columns([1, 10], gap="small")
    
    with c_mic:
        st.write("") # Alignment push
        audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format="webm", just_once=True)
        
    with c_input:
        voice_text = ""
        if audio:
            try:
                audio_file = io.BytesIO(audio['bytes'])
                audio_file.name = "audio.webm"
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                voice_text = transcript.text
            except: pass
            
        default_val = voice_text if voice_text else ""
        user_question = st.text_input("Search", value=default_val, placeholder="Ask anything about the campus...", label_visibility="collapsed")

    # --- AI ANSWER SECTION ---
    if user_question:
        with st.spinner("🧠 Processing query..."):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                if os.path.exists("faiss_index"):
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    
                    st.markdown(f"""
                    <div class="answer-box">
                        <h3 style="color: #00C853; margin: 0;">🤖 Answer</h3>
                        <hr style="border-color: rgba(255,255,255,0.2); margin: 15px 0;">
                        <p style="font-size: 18px; line-height: 1.6; color: white;">{response['output_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Knowledge Base Empty.")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# PAGE 2: ADMIN PORTAL (Touch Nothing!)
# ==========================================
if selected == "Admin Portal":
    c1, c2 = st.columns([1, 3])
    with c1:
        if lottie_admin: st_lottie(lottie_admin, height=150)
    with c2:
        st.title("Admin Upload")
        st.write("Securely upload circulars.")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True, type=['pdf'])
    
    st.write("")
    if st.button("Process & Upload"):
        if pdf_docs:
            with st.status("Processing...", expanded=True):
                text = ""
                for pdf in pdf_docs:
                    with pdfplumber.open(pdf) as pdf_file:
                        for page in pdf_file.pages:
                            t = page.extract_text()
                            if t: text += t
                    # Upload
                    with open(pdf.name, "wb") as f: f.write(pdf.getbuffer())
                    upload_to_drive(pdf.name, pdf.name)
                    if os.path.exists(pdf.name): os.remove(pdf.name)
                
                # Train
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(text)
                get_vector_store(chunks)
                st.success("Success!")
                time.sleep(1)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: ABOUT
# ==========================================
if selected == "About":
    st.title("About")
    st.markdown("""
    <div class="glass-card">
        <h3>CampusMind AI</h3>
        <p style="color:white;">Built for the 2024 Innovation Hackathon.</p>
    </div>
    """, unsafe_allow_html=True)
