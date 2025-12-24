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
st.set_page_config(page_title="CampusMind AI", page_icon="🎓", layout="wide")

# LOAD SECRETS
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("🚨 OpenAI API Key missing! Check .streamlit/secrets.toml")
    
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 

except FileNotFoundError:
    st.error("🚨 Secrets file not found!")

# --- 2. HACKATHON WINNING CSS ---
st.markdown("""
<style>
    /* MAIN GRADIENT BACKGROUND */
    .stApp {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }

    /* TEXT VISIBILITY FIX - GLOBAL */
    h1, h2, h3, h4, h5, p, span, div {
        color: white !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* INPUT BOX - FORCE VISIBILITY (White Box, Black Text) */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #ddd;
    }
    .stTextInput input:focus {
        border-color: #00C853 !important;
        box-shadow: 0 0 10px rgba(0,200,83,0.5);
    }
    .stTextInput label {
        color: white !important;
        font-weight: bold;
    }

    /* CARD STYLING (Glassmorphism) */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: #00C853;
    }

    /* MIC BUTTON STYLING (Targeting the recorder specifically) */
    div[data-testid="stButton"] button {
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #00C853;
        color: white;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        font-size: 24px;
        transition: all 0.3s;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #00e676;
        transform: scale(1.1);
    }

    /* STANDARD BUTTON OVERRIDE (For "Process & Upload") */
    /* This fixes the "Admin Portal" button looking like a circle */
    .stButton button {
        border-radius: 8px !important;
        width: auto !important;
        height: auto !important;
        padding: 0.5rem 1rem !important;
        font-size: 16px !important;
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
    """Fetches last 3 files for the card display"""
    try:
        if "gcp_service_account" not in st.secrets:
            return []
        key_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive'])
        service = build('drive', 'v3', credentials=creds)
        
        # supportsAllDrives=True is crucial for some permissions
        query = f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"
        results = service.files().list(
            q=query, pageSize=3, fields="files(id, name, createdTime)",
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
lottie_ai = load_lottieurl("https://lottie.host/020cc52c-7472-4632-841f-82559b95427d/21H5gH1p7E.json") # Thinking AI

# --- 5. MAIN UI LAYOUT ---

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=60)
    st.markdown("### Menu")
    selected = option_menu(
        menu_title=None,
        options=["Student Chat", "Admin Portal", "About"],
        icons=['chat-dots', 'cloud-upload', 'info-circle'],
        default_index=0,
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {"font-size": "16px", "color": "white", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#00C853"},
        }
    )
    st.markdown("---")
    st.caption("v1.0.0 | Powered by OpenAI")

# ==========================================
# PAGE 1: STUDENT CHAT (THE MAIN EVENT)
# ==========================================
if selected == "Student Chat":
    
    # --- HERO SECTION ---
    # Centered layout for maximum impact
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_hello: 
            st_lottie(lottie_hello, height=180, key="hero_anim")
        st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>CampusMind AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.8;'>Your Personal Campus Assistant • 24/7 Availability</p>", unsafe_allow_html=True)
    
    st.write("") # Spacer
    st.write("") 

    # --- RECENT CIRCULARS (CARDS LAYOUT) ---
    st.markdown("### 📢 Recent Updates")
    
    with st.spinner("Checking Digital Notice Board..."):
        recent_files = get_recent_circulars()
        
    if recent_files:
        # Create 3 columns for 3 cards
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        
        for i, file in enumerate(recent_files):
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="color: #00C853 !important;">📄 Update</h3>
                    <p style="font-size: 14px; margin-bottom: 5px;">{file['name'][:25]}...</p>
                    <small style="color: #bbb !important;">Tap to ask about this</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No recent circulars found on Drive.")

    st.markdown("---")

    # --- INTERACTION ZONE ---
    st.markdown("### 💬 Ask Anything")

    # Layout: Mic Button (Left) + Input Field (Right)
    # We use a container to group them visually
    with st.container():
        c_mic, c_input = st.columns([1, 8])
        
        with c_mic:
            st.write("") # Spacer to push button down
            # The Mic Button
            audio = mic_recorder(
                start_prompt="🎙️", 
                stop_prompt="⏹️", 
                key='recorder', 
                format="webm",
                just_once=True
            )
        
        with c_input:
            # Voice Transcription Logic
            voice_text = ""
            if audio:
                with st.spinner("🎧 Listening..."):
                    try:
                        audio_file = io.BytesIO(audio['bytes'])
                        audio_file.name = "audio.webm"
                        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                        voice_text = transcript.text
                    except:
                        st.error("Could not understand audio.")

            # The Input Box
            default_val = voice_text if voice_text else ""
            user_question = st.text_input(
                "Search", 
                value=default_val, 
                placeholder="Type 'Exam dates' or click the mic...", 
                label_visibility="collapsed"
            )

    # --- AI RESPONSE SECTION ---
    if user_question:
        st.write("")
        st.write("")
        
        # Create a container for the answer
        result_container = st.container()
        
        with result_container:
            # Show a cool "Thinking" animation instead of a boring spinner
            think_col, _ = st.columns([1, 10])
            with think_col:
                if lottie_ai: st_lottie(lottie_ai, height=60, key="thinking")
            
            with st.spinner("Analyzing circulars..."):
                try:
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    if os.path.exists("faiss_index"):
                        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                        docs = new_db.similarity_search(user_question)
                        chain = get_conversational_chain()
                        response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                        
                        # BEAUTIFUL ANSWER CARD
                        st.markdown(f"""
                        <div style="
                            background: rgba(0, 200, 83, 0.1); 
                            border-left: 6px solid #00C853; 
                            padding: 25px; 
                            border-radius: 10px; 
                            margin-top: 10px; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                            <h4 style="color: #00C853 !important; margin-top: 0;">🤖 CampusMind Answer:</h4>
                            <p style="font-size: 18px; line-height: 1.6; color: white !important;">{response['output_text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("⚠️ System Offline: No knowledge found. (Admin must upload circulars)")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# PAGE 2: ADMIN PORTAL
# ==========================================
if selected == "Admin Portal":
    st.markdown("## 🔐 Admin Portal")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("Upload official circulars (PDF) to update the AI's knowledge base.")
    
    pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True, type=['pdf'])
    
    if st.button("🚀 Upload & Train AI"):
        if pdf_docs:
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            
            text = ""
            for i, pdf in enumerate(pdf_docs):
                # Update progress bar
                perc = int(((i+1) / len(pdf_docs)) * 50)
                my_bar.progress(perc, text=f"Reading {pdf.name}...")
                
                with pdfplumber.open(pdf) as pdf_file:
                    for page in pdf_file.pages:
                        t = page.extract_text()
                        if t: text += t
                
                # Backup to Drive
                with open(pdf.name, "wb") as f: f.write(pdf.getbuffer())
                upload_to_drive(pdf.name, pdf.name)
                if os.path.exists(pdf.name): os.remove(pdf.name)

            # Training Phase
            my_bar.progress(70, text="🧠 Training AI Model...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(text)
            get_vector_store(chunks)
            
            my_bar.progress(100, text="✅ Complete!")
            time.sleep(1)
            my_bar.empty()
            
            st.success("System Updated Successfully!")
            st.balloons()
        else:
            st.warning("Please select a file to upload.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: ABOUT
# ==========================================
if selected == "About":
    st.markdown("## ℹ️ About CampusMind")
    st.markdown("""
    <div class="glass-card">
        <h3>🏆 Hackathon Project 2024</h3>
        <p>CampusMind is an AI-powered assistant designed to help students access official campus information instantly.</p>
        <ul>
            <li><b>Instant Answers:</b> No more searching through emails.</li>
            <li><b>Voice Enabled:</b> Just ask, don't type.</li>
            <li><b>Always Updated:</b> Syncs with Admin uploads instantly.</li>
        </ul>
        <br>
        <small>Built with Streamlit, LangChain, and OpenAI.</small>
    </div>
    """, unsafe_allow_html=True)
