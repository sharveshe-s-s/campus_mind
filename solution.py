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

# --- 0. AUTO-FIX THEME (THE MAGIC FIX) ---
# This block checks if a config file exists. If not, it creates one to FORCE DARK MODE.
def setup_theme_config():
    config_dir = ".streamlit"
    config_path = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        
    # Only write if file doesn't exist to prevent overwriting custom changes
    if not os.path.exists(config_path):
        config_content = """
[theme]
base = "dark"
primaryColor = "#00C853"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#ffffff"
font = "sans serif"
        """
        with open(config_path, "w") as f:
            f.write(config_content)
        # Rerun to apply changes immediately
        st.rerun()

setup_theme_config()

# --- 1. CONFIGURATION & SECRETS ---
st.set_page_config(page_title="CampusMind AI", page_icon="🎓", layout="wide")

# LOAD SECRETS
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 

except FileNotFoundError:
    st.error("🚨 Secrets file not found!")

# --- 2. ADVANCED CSS (POLISH & ALIGNMENT) ---
st.markdown("""
<style>
    /* 1. TEXT INPUT STYLING (The "Search Bar" Look) */
    .stTextInput input {
        background-color: #1e293b !important;
        color: white !important;
        border: 2px solid #334155;
        border-radius: 50px; /* Pill shape */
        padding: 15px 25px;
        font-size: 16px;
    }
    .stTextInput input:focus {
        border-color: #00C853 !important;
        box-shadow: 0 0 15px rgba(0, 200, 83, 0.3);
    }
    
    /* 2. MIC BUTTON HACK */
    /* This targets the mic recorder widget to make it round and centered */
    .stButton button {
        border-radius: 50% !important;
        width: 55px !important;
        height: 55px !important;
        padding: 0 !important;
        background-color: #1e293b !important;
        border: 2px solid #334155 !important;
        color: #00C853 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        border-color: #00C853 !important;
        color: white !important;
        background-color: #00C853 !important;
        transform: scale(1.1);
    }

    /* 3. CARDS */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }

    /* 4. CHAT MESSAGE STYLING */
    .chat-box {
        background-color: #1e293b;
        border-left: 5px solid #00C853;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Hide standard Streamlit header/footer for cleaner look */
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---

def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

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

# --- 4. ASSETS ---
lottie_robot = load_lottieurl("https://lottie.host/5a919f2d-304b-4b15-9c8b-30234157d6b3/2k2k2k2k2k.json") 

# --- 5. SIDEBAR ---
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
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "5px", "--hover-color": "#334155"},
            "nav-link-selected": {"background-color": "#00C853"},
        }
    )
    st.markdown("---")
    st.caption("Powered by OpenAI GPT-4o")

# --- PAGE 1: STUDENT CHAT ---
if selected == "Student Chat":
    
    # HERO SECTION
    col1, col2 = st.columns([1, 4])
    with col1:
        if lottie_robot: st_lottie(lottie_robot, height=120, key="anim")
    with col2:
        st.markdown("<h1 style='padding-top: 10px; margin-bottom: 0;'>CampusMind AI</h1>", unsafe_allow_html=True)
        st.caption("Ask about Exams, Bus Routes, Fees, and Official Circulars.")

    # RECENT UPDATES (The "Glass" Cards)
    st.write("")
    st.markdown("##### 📢 Recent Circulars")
    
    recent_files = get_recent_circulars()
    if recent_files:
        cols = st.columns(3)
        for i, file in enumerate(recent_files):
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card">
                    <span style="font-size: 20px;">📄</span>
                    <div style="font-weight: bold; margin-top: 5px; font-size: 14px;">{file['name'][:20]}...</div>
                    <div style="font-size: 12px; color: #aaa;">Tap to copy name</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        # A nice placeholder instead of an error message
        st.info("Everything is up to date! No new circulars recently.")

    st.markdown("---")
    
    # SEARCH INTERFACE (Aligned Perfectly)
    st.markdown("##### 💬 Ask Anything")
    
    # Using columns to align the mic button and input box horizontally
    c1, c2 = st.columns([1, 12], gap="small")
    
    with c1:
        # Mic Button
        audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format="webm", just_once=True)
    
    with c2:
        # Voice Logic
        voice_text = ""
        if audio:
            with st.spinner("Processing voice..."):
                try:
                    audio_file = io.BytesIO(audio['bytes'])
                    audio_file.name = "audio.webm"
                    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                    voice_text = transcript.text
                except: pass
        
        # Search Box
        default_text = voice_text if voice_text else ""
        user_question = st.text_input("Query", value=default_text, placeholder="Ex: When does the semester start?", label_visibility="collapsed")

    # ANSWER SECTION
    if user_question:
        with st.spinner("🧠 Analyzing database..."):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                if os.path.exists("faiss_index"):
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    
                    st.markdown(f"""
                    <div class="chat-box">
                        <h4 style="color: #00C853; margin-top: 0;">🤖 Answer:</h4>
                        <p style="font-size: 17px; line-height: 1.6;">{response['output_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Knowledge base is empty. Please upload circulars in Admin Portal.")
            except Exception as e:
                st.error(f"Error: {e}")

# --- PAGE 2: ADMIN PORTAL ---
if selected == "Admin Portal":
    st.title("Admin Portal")
    st.write("Upload PDF circulars here.")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    pdf_docs = st.file_uploader("Select PDFs", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Upload & Train"):
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
                st.success("Success! The AI has been updated.")
                time.sleep(1)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE 3: ABOUT ---
if selected == "About":
    st.title("About")
    st.markdown("""
    <div class="glass-card">
        <h3>CampusMind AI</h3>
        <p>Built for the 2024 Innovation Hackathon.</p>
        <p><b>Stack:</b> Streamlit, OpenAI GPT-4o, FAISS, Google Drive API.</p>
    </div>
    """, unsafe_allow_html=True)
