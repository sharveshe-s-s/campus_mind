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

# --- 1. CONFIGURATION & SECRETS ---
st.set_page_config(page_title="CampusMind AI", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# LOAD SECRETS
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("🚨 OpenAI API Key missing! Check .streamlit/secrets.toml")
    
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 

except FileNotFoundError:
    st.error("🚨 Secrets file not found!")

# --- 2. HACKATHON WINNING CSS (Glassmorphism & Neon) ---
st.markdown("""
<style>
    /* 1. MAIN BACKGROUND: Deep Modern Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-attachment: fixed;
    }

    /* 2. TEXT VISIBILITY FIX - Force White Text */
    h1, h2, h3, h4, h5, h6, p, li, div, span, label {
        color: #FFFFFF !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* 3. SIDEBAR: Glass Effect */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* 4. INPUT FIELD STYLING */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 10px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00C853;
        box-shadow: 0 0 10px rgba(0, 200, 83, 0.5);
    }

    /* 5. COOL MIC BUTTON (Floating & Glowing) */
    div[data-testid="stButton"] button {
        border-radius: 50%;
        width: 60px;
        height: 60px;
        padding: 0;
        font-size: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, #00C853, #009624);
        border: none;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.6);
        transition: all 0.3s ease;
        color: white !important;
    }
    div[data-testid="stButton"] button:hover {
        transform: scale(1.15) rotate(5deg);
        box-shadow: 0 6px 20px rgba(0, 200, 83, 0.8);
    }

    /* 6. EXPANDER (Recent Circulars) */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px;
        color: white !important;
    }
    
    /* 7. CHAT BUBBLE STYLING */
    .chat-bubble {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border-left: 5px solid #00C853;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
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
        if "gcp_service_account" not in st.secrets:
            return "Error: Secrets missing"
        key_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
        media = MediaFileUpload(file_path, mimetype='application/pdf')
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        return f"Error: {e}"

def get_recent_circulars():
    """Fetches last 5 files"""
    try:
        if "gcp_service_account" not in st.secrets:
            return []
        key_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=creds)
        
        query = f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"
        results = service.files().list(
            q=query, pageSize=5, fields="files(id, name, createdTime)",
            orderBy="createdTime desc", supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        return results.get('files', [])
    except Exception as e:
        return []

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an intelligent campus assistant for CIT. Answer the question based ONLY on the provided Context.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- 4. ASSETS ---
lottie_hello = load_lottieurl("https://lottie.host/5a919f2d-304b-4b15-9c8b-30234157d6b3/2k2k2k2k2k.json") 
lottie_upload = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")

# --- 5. MAIN UI LAYOUT ---

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50) # Small Logo
    selected = option_menu(
        "CampusMind",
        ["Student Chat", "Admin Portal", "About"],
        icons=['chat-dots', 'cloud-upload', 'info-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {"font-size": "16px", "color": "white", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#00C853"},
        }
    )
    st.markdown("---")
    st.caption("Powered by OpenAI & LangChain")

# --- PAGE 1: STUDENT CHAT ---
if selected == "Student Chat":
    
    # --- HEADER SECTION (Animation + Title) ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # THE ANIMATION IS HERE
        if lottie_hello: 
            st_lottie(lottie_hello, height=200, key="hello_anim")
        else:
            st.write("🤖") 

    with col2:
        st.markdown("<h1 style='font-size: 50px; margin-bottom: 0;'>CampusMind AI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='opacity: 0.8;'>Your 24/7 Smart Campus Assistant</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 10px; border-radius: 10px; display: inline-block;'>
        Ask about: 📅 <b>Exams</b> &nbsp;|&nbsp; 🚌 <b>Bus Routes</b> &nbsp;|&nbsp; 📝 <b>Fees</b>
        </div>
        """, unsafe_allow_html=True)

    st.write("") # Spacer

    # --- RECENT UPDATES ---
    with st.expander("📢 Recent Circulars (Live from Office)", expanded=True):
        with st.spinner("Syncing..."):
            recent_files = get_recent_circulars()
            if recent_files:
                cols = st.columns(len(recent_files) if len(recent_files) < 4 else 4)
                for i, file in enumerate(recent_files[:4]):
                    with cols[i]:
                        st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                            <div style='font-size: 20px;'>📄</div>
                            <div style='font-size: 12px; margin-top: 5px; color: #ddd !important;'>{file['name'][:20]}..</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No recent circulars found.")

    st.markdown("---")

    # --- CHAT INPUT AREA (Aligned) ---
    # Col 1: Mic (Small), Col 2: Text Input (Wide)
    mic_col, text_col = st.columns([1, 8], gap="small")
    
    with mic_col:
        # Pushing the button down slightly to align with text box
        st.write("") 
        st.write("")
        audio = mic_recorder(start_prompt="🎙️", stop_prompt="⏹️", key='recorder', format="webm", just_once=True)

    voice_text = ""
    if audio:
        with st.spinner("Transcribing..."):
            try:
                audio_file = io.BytesIO(audio['bytes'])
                audio_file.name = "audio.webm"
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                voice_text = transcript.text
            except:
                st.error("Voice Error")

    # Text Input with Voice auto-fill
    with text_col:
        initial_text = voice_text if voice_text else ""
        user_question = st.text_input("Ask a question:", value=initial_text, placeholder="Type here or use the mic...", label_visibility="visible")

    # --- PROCESSING & ANSWER ---
    if user_question:
        with st.spinner("🧠 Thinking..."):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                if os.path.exists("faiss_index"):
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    
                    # GLASSMORPHISM ANSWER CARD
                    st.markdown(f"""
                    <div class="chat-bubble">
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <span style="font-size: 24px; margin-right: 10px;">🤖</span>
                            <h4 style="margin: 0; color: #00C853 !important;">CampusMind Answer:</h4>
                        </div>
                        <p style="font-size: 17px; line-height: 1.6; color: white !important;">{response['output_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Brain empty! Please upload circulars in Admin Portal.")
            except Exception as e:
                st.error(f"Error: {e}")

# --- PAGE 2: ADMIN PORTAL ---
if selected == "Admin Portal":
    c1, c2 = st.columns([1, 3])
    with c1:
        if lottie_upload: st_lottie(lottie_upload, height=150)
    with c2:
        st.title("Admin Upload")
        st.write("Securely upload circulars to the AI Brain.")

    pdf_docs = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=['pdf'])

    if st.button("Process & Upload"):
        if pdf_docs:
            with st.status("Processing...", expanded=True) as status:
                text = ""
                for pdf in pdf_docs:
                    with pdfplumber.open(pdf) as pdf_file:
                        for page in pdf_file.pages:
                            t = page.extract_text()
                            if t: text += t
                    
                    # Upload to Drive
                    st.write(f"☁️ Uploading {pdf.name}...")
                    with open(pdf.name, "wb") as f: f.write(pdf.getbuffer())
                    upload_to_drive(pdf.name, pdf.name)
                    if os.path.exists(pdf.name): os.remove(pdf.name)

                # Train
                st.write("🧠 Retraining Model...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(text)
                get_vector_store(chunks)
                time.sleep(2)
                status.update(label="System Updated!", state="complete", expanded=False)
                st.success("Success!")
        else:
            st.warning("Select a file first.")

# --- PAGE 3: ABOUT ---
if selected == "About":
    st.title("About CampusMind")
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;'>
    <h3>🚀 Tech Stack</h3>
    <ul>
        <li><b>Frontend:</b> Streamlit (Glassmorphism UI)</li>
        <li><b>AI Brain:</b> OpenAI GPT-4o-Mini + FAISS</li>
        <li><b>Voice:</b> OpenAI Whisper</li>
        <li><b>Storage:</b> Google Drive API</li>
    </ul>
    <br>
    <p>Built for <b>Open Innovation Hackathon 2024</b>.</p>
    </div>
    """, unsafe_allow_html=True)
