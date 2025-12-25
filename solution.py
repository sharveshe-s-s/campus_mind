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
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#161b22"
textColor = "#fafafa"
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
# 2. HACKATHON WINNING CSS (UPDATED WITH ANIMATIONS)
# ==========================================
st.markdown("""
<style>
    /* --- ANIMATION: ENTRY SLIDE UP --- */
    .block-container {
        animation: slideUpFade 0.8s ease-out;
    }
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- ANIMATION: SHIMMER TEXT FOR TITLE --- */
    .shimmer-text {
        font-weight: bold;
        background: linear-gradient(45deg, #ffffff, #a8c0ff, #ffffff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }

    /* 1. MAIN BACKGROUND: Deep Cyberpunk Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    /* 2. SIDEBAR POLISH */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.4);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* 3. INPUT BOX - HIGH VISIBILITY + NEW HOVER GLOW */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 12px;
        padding: 12px 12px 12px 50px; 
        font-size: 16px;
        border: 2px solid transparent;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    /* Hover Effect added here */
    .stTextInput input:hover {
        transform: scale(1.01);
    }
    .stTextInput input:focus {
        border-color: #00C853 !important;
        box-shadow: 0 0 15px rgba(0, 200, 83, 0.5);
        transform: scale(1.01);
    }

    /* 4. MIC BUTTON (Integrated look) */
    div[data-testid="stButton"] button {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        padding: 0;
        background: transparent;
        border: none;
        color: #00C853;
        font-size: 20px;
        transition: all 0.3s ease;
        position: relative;
        left: 5px;
        top: 2px;
        z-index: 1;
    }
    div[data-testid="stButton"] button:hover {
        color: #009624;
        transform: scale(1.1);
    }

    /* 5. PROCESS BUTTON (FIXED ALIGNMENT) */
    /* Added white-space: nowrap to prevent squashing */
    .stButton button.process-btn {
        border-radius: 30px !important;
        width: auto !important;
        height: auto !important;
        white-space: nowrap !important; /* <--- KEY FIX */
        min-width: 160px;               /* <--- KEY FIX */
        padding: 10px 30px !important;
        background: linear-gradient(90deg, #00C853 0%, #009624 100%);
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
    }
    .stButton button.process-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 200, 83, 0.5);
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

    /* 7. AI ANSWER BOX */
    .answer-box-container {
        background: rgba(0, 0, 0, 0.6);
        border-left: 6px solid #00C853;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        color: #ffffff !important;
    }

    /* 8. HIDE STREAMLIT UI ELEMENTS */
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
    except: return None

def stream_text(text):
    """Yields text character by character/word by word to simulate typing."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05) # Adjust typing speed here

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

# ASSETS
lottie_student_ai = load_lottieurl("https://lottie.host/99977635-960c-4432-9987-440077149909/99977635-960c-4432-9987-440077149909.json")
lottie_admin = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")

# ==========================================
# 4. SIDEBAR
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
    st.caption("v2.0 | Hackathon Edition")

# ==========================================
# PAGE 1: STUDENT CHAT (Winner UI)
# ==========================================
if selected == "Student Chat":
    
    # --- HERO HEADER WITH ANIMATION & SHIMMER TITLE ---
    col1, col2 = st.columns([1, 2])
    with col1:
        if lottie_student_ai: 
            st_lottie(lottie_student_ai, height=200, key="hero_anim")
    with col2:
        st.markdown("<div style='display: flex; flex-direction: column; justify-content: center; height: 200px;'>", unsafe_allow_html=True)
        # APPLIED THE SHIMMER CLASS HERE
        st.markdown("<h1 class='shimmer-text' style='font-size: 48px; margin-bottom: 0;'>CampusMind AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 18px; opacity: 0.8;'>Your 24/7 Smart Campus Assistant</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("") # Spacer

    # --- RECENT UPDATES (GLASS CARDS) ---
    st.markdown("##### 📢 Recent Updates")
    with st.spinner("Syncing..."):
        recent_files = get_recent_circulars()
        
    if recent_files:
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for i, file in enumerate(recent_files):
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="color: #00C853; font-weight: bold; font-size: 18px;">📄 New Circular</div>
                    <div style="margin-top: 5px; font-size: 14px; color: white;">{file['name'][:25]}...</div>
                    <div style="font-size: 12px; color: #aaa; margin-top: 5px;">Tap the mic to ask about this.</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recent circulars uploaded. You can still ask general questions!")

    st.markdown("---")

    # --- CHAT INTERFACE ---
    st.markdown("##### 💬 Ask Anything")
    
    with st.container():
        c_mic, c_input = st.columns([1, 11])
        
        with c_mic:
            audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format="webm", just_once=True)
            
        with c_input:
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
            
            default_val = voice_text if voice_text else ""
            user_question = st.text_input("Search", value=default_val, placeholder="Ex: When are the exams?", label_visibility="collapsed")

    # --- ANSWER SECTION WITH TYPING EFFECT ---
    if user_question:
        with st.spinner("🧠 Analyzing..."):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                if os.path.exists("faiss_index"):
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    
                    full_response = response['output_text']

                    # --- UPDATED DISPLAY LOGIC ---
                    # We create a container that simulates the 'Answer Box' style
                    st.markdown("""
                        <div class="answer-box-container">
                        <h3 style="color: #00C853; margin: 0;">🤖 Answer:</h3>
                        <hr style="border-color: rgba(255,255,255,0.2); margin: 15px 0;">
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # We stream the text OUTSIDE the HTML div but it will look visually connected
                    # This enables the typing effect
                    st.write_stream(stream_text(full_response))
                    
                else:
                    st.warning("⚠️ Knowledge Base Empty. Please upload circulars in Admin Portal.")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# PAGE 2: ADMIN PORTAL
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
    
    # Process Button with Class injection
    if st.button("Process & Upload", key="process_btn", help="Click to process and upload documents"):
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
        else:
            st.warning("Please select at least one PDF file.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Inject custom CSS class for button targeting
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
        <h3>CampusMind AI</h3>
        <p style="color:white;">Built for the 2024 Innovation Hackathon.</p>
        <ul>
            <li><b>Frontend:</b> Streamlit (Custom CSS)</li>
            <li><b>AI Brain:</b> OpenAI GPT-4o-Mini</li>
            <li><b>Vector DB:</b> FAISS</li>
            <li><b>Storage:</b> Google Drive API</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
