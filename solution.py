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
        
    # Drive Folder ID - ENSURE THIS IS CORRECT
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 

except FileNotFoundError:
    st.error("🚨 Secrets file not found!")

# --- 2. CUSTOM CSS (COOL MIC & STYLE) ---
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    
    /* Cool Mic Button Styling */
    div[data-testid="stButton"] button {
        border-radius: 50%;
        width: 60px;
        height: 60px;
        padding: 0;
        font-size: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 14px 0 rgba(0, 200, 83, 0.39);
        transition: transform 0.2s;
    }
    div[data-testid="stButton"] button:hover {
        transform: scale(1.1);
    }

    /* Standard Button Styling (Submit/Upload) */
    .stButton > button {
        border-radius: 12px !important;
        width: auto !important;
        height: auto !important;
        padding: 10px 24px !important;
        font-size: 16px !important;
    }

    /* Card Styling */
    .css-1r6slb0 {
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 20px;
        background: rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---

def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
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
    """Fetches the last 5 uploaded files with Robust Sorting"""
    try:
        if "gcp_service_account" not in st.secrets:
            return []

        key_dict = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(
            key_dict, scopes=['https://www.googleapis.com/auth/drive'])
        
        service = build('drive', 'v3', credentials=creds)
        
        # Query: Inside folder, not trash.
        # supportsAllDrives=True fixes issues with Shared Drives
        query = f"'{DRIVE_FOLDER_ID}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            pageSize=5,
            fields="files(id, name, createdTime)",
            orderBy="createdTime desc",
            supportsAllDrives=True, 
            includeItemsFromAllDrives=True
        ).execute()
        
        return results.get('files', [])
    except Exception as e:
        # st.error(f"Drive Error: {e}") # Uncomment to debug
        return []

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an intelligent campus assistant for CIT. Answer the question based ONLY on the provided Context.
    If the answer is not in the context, say "I couldn't find that in the official circulars."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- 4. ASSETS ---
lottie_hello = load_lottieurl("https://lottie.host/5a919f2d-304b-4b15-9c8b-30234157d6b3/2k2k2k2k2k.json") 
lottie_upload = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")

# --- 5. MAIN UI ---

with st.sidebar:
    selected = option_menu(
        "CampusMind",
        ["Student Chat", "Admin Portal", "About"],
        icons=['chat-dots', 'cloud-upload', 'info-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#262730"},
            "nav-link": {"font-size": "16px", "color": "white"},
            "nav-link-selected": {"background-color": "#00C853"},
        }
    )

# --- PAGE 1: STUDENT CHAT ---
if selected == "Student Chat":
    
    # Header
    c1, c2 = st.columns([1, 4])
    with c1:
        if lottie_hello: st_lottie(lottie_hello, height=150, key="anim")
    with c2:
        st.title("CampusMind AI")
        st.caption("Ask about exams, fees, buses, and more!")

    # --- RECENT UPDATES (New Layout) ---
    with st.expander("📢 Latest Circulars (Live Updates)", expanded=True):
        with st.spinner("Syncing with Office..."):
            recent_files = get_recent_circulars()
            if recent_files:
                # Show in a nice horizontal layout or list
                for i, file in enumerate(recent_files):
                    st.markdown(f"**{i+1}.** 📄 {file['name']}")
            else:
                st.info("No circulars found recently.")

    st.markdown("---")

    # --- CHAT INTERFACE ---
    # We use columns to center the mic or align it
    mic_col, text_col = st.columns([1, 8])
    
    with mic_col:
        st.write(" ") # Spacer
        # The Cool Mic Button
        audio = mic_recorder(
            start_prompt="🎙️",  # Microphone Icon
            stop_prompt="⏹️",   # Stop Icon
            key='recorder',
            format="webm",
            just_once=True      # Prevents infinite rerun loops
        )

    voice_text = ""
    if audio:
        with st.spinner("Transcribing..."):
            try:
                audio_file = io.BytesIO(audio['bytes'])
                audio_file.name = "audio.webm"
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )
                voice_text = transcript.text
            except Exception as e:
                st.error(f"Voice Error: {e}")

    # Search Bar (Auto-filled by voice)
    initial_text = voice_text if voice_text else ""
    user_question = st.text_input("", value=initial_text, placeholder="Type your query or use the mic...", label_visibility="collapsed")

    if user_question:
        with st.spinner("🧠 Analyzing circulars..."):
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
                    
                    st.markdown(f"""
                    <div style="background-color: rgba(0, 200, 83, 0.15); padding: 20px; border-radius: 15px; border-left: 5px solid #00C853; margin-top: 10px;">
                        <h4 style="margin:0; color: #00C853;">🤖 Answer:</h4>
                        <p style="font-size: 18px; margin-top: 10px;">{response['output_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Brain empty! Admin needs to upload data.")
            except Exception as e:
                st.error(f"Error: {e}")

# --- PAGE 2: ADMIN PORTAL ---
if selected == "Admin Portal":
    c1, c2 = st.columns([1, 3])
    with c1:
        if lottie_upload: st_lottie(lottie_upload, height=150)
    with c2:
        st.title("Admin Upload")
        st.write("Upload PDF circulars here.")

    pdf_docs = st.file_uploader("Choose PDFs", accept_multiple_files=True, type=['pdf'])

    if st.button("Process & Upload"):
        if pdf_docs:
            with st.status("Processing...", expanded=True) as status:
                
                text = ""
                for pdf in pdf_docs:
                    # 1. READ
                    with pdfplumber.open(pdf) as pdf_file:
                        for page in pdf_file.pages:
                            t = page.extract_text()
                            if t: text += t
                    
                    # 2. UPLOAD TO DRIVE
                    st.write(f"☁️ Uploading {pdf.name}...")
                    
                    # Create temp file for upload
                    with open(pdf.name, "wb") as f:
                        f.write(pdf.getbuffer())
                    
                    upload_to_drive(pdf.name, pdf.name)
                    
                    # Cleanup
                    if os.path.exists(pdf.name): os.remove(pdf.name)

                # 3. TRAIN AI
                st.write("🧠 Retraining AI Model...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(text)
                get_vector_store(chunks)
                
                # Sleep to allow Drive to update index before User checks it
                time.sleep(2) 
                
                status.update(label="Done! System Updated.", state="complete", expanded=False)
                st.success("Uploaded & Trained Successfully!")
        else:
            st.warning("Please select a file.")

# --- PAGE 3: ABOUT ---
if selected == "About":
    st.title("About CampusMind")
    st.markdown("Built with **Streamlit, LangChain, OpenAI & Google Drive API**.")
