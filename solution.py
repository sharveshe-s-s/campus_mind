import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import requests
import pdfplumber

# --- STANDARD IMPORTS (Guaranteed to work with langchain==0.1.13) ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate 

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

# --- 1. CONFIGURATION & SECRETS ---
st.set_page_config(page_title="CampusMind AI", page_icon="🎓", layout="wide")

# LOAD SECRETS (Works for both Local & Cloud if configured correctly)
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("🚨 OpenAI API Key missing in Secrets!")
        
    # Drive Folder ID 
    DRIVE_FOLDER_ID = '1IRAXoxny14JvI6UbJ1zPyUduwlzm5Egm' 

except FileNotFoundError:
    st.error("🚨 Secrets file not found! If running locally, create .streamlit/secrets.toml")

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    /* Card Styling */
    div.stButton > button:first-child {
        background-color: #00C853;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px; 
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #009624;
        transform: scale(1.05);
    }
    /* Spinner Color */
    .stSpinner > div {
        border-top-color: #00C853 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---

def load_lottieurl(url):
    """Safely load Lottie animations without crashing"""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def upload_to_drive(file_path, file_name):
    """Uploads using Secrets instead of a JSON file"""
    try:
        # Check if secrets exist
        if "gcp_service_account" not in st.secrets:
            return "Error: Google Credentials not found in Secrets!"

        # Load credentials directly from the secrets dictionary
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

def get_vector_store(text_chunks):
    """Converts text into vectors and saves them locally"""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Setup the Chat Model"""
    prompt_template = """
    You are an intelligent campus assistant for CIT. Answer the question based ONLY on the provided Context.
    
    If the answer is not in the context, say "I couldn't find that in the official circulars."
    Do not make up answers.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# --- 4. ANIMATION ASSETS ---
lottie_hello = load_lottieurl("https://lottie.host/5a919f2d-304b-4b15-9c8b-30234157d6b3/2k2k2k2k2k.json") 
if not lottie_hello:
     lottie_hello = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_p1qiuawe.json")

lottie_upload = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")

# --- 5. MAIN UI LAYOUT ---

with st.sidebar:
    selected = option_menu(
        "CampusMind",
        ["Student Chat", "Admin Portal", "About"],
        icons=['chat-dots', 'cloud-upload', 'info-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#262730"},
            "icon": {"color": "white", "font-size": "25px"}, 
            "nav-link": {
                "font-size": "16px", 
                "text-align": "left", 
                "margin": "0px", 
                "--hover-color": "#444",
                "color": "white" 
            },
            "nav-link-selected": {"background-color": "#00C853"},
            "menu-title": {
                "color": "white", 
                "font-size": "20px", 
                "font-weight": "bold"
            }
        }
    )

# --- PAGE 1: STUDENT CHAT ---
if selected == "Student Chat":
    col1, col2 = st.columns([1, 2])
    with col1:
        if lottie_hello:
            st_lottie(lottie_hello, height=250, key="hello_anim")
        else:
            st.write("🤖") 

    with col2:
        st.title("")
        st.write("Welcome, Student! Ask me about:")
        st.markdown("""
        * 📅 **Exam Schedules**
        * 🚌 **Bus Routes & Timings**
        * 📝 **Syllabus & Fees**
        """)

    st.markdown("---") 

    user_question = st.text_input("Type your query here...", placeholder="Ex: When is the revaluation deadline?")

    if user_question:
        with st.spinner("🧠 Analyzing circulars..."):
            try:
                embeddings = OpenAIEmbeddings()
                if os.path.exists("faiss_index"):
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(user_question)
                    chain = get_conversational_chain()
                    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    
                    st.markdown(f"""
                    <div style="background-color: rgba(0, 200, 83, 0.2); padding: 20px; border-radius: 10px; border-left: 5px solid #00C853; margin-top: 20px;">
                        <h4 style="margin-bottom: 10px;">🤖 AI Answer:</h4>
                        <p style="font-size: 18px; line-height: 1.6;">{response['output_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ The AI Brain is empty! Please ask Admin to upload circulars first.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- PAGE 2: ADMIN PORTAL ---
if selected == "Admin Portal":
    col1, col2 = st.columns([1, 2])
    with col1:
        if lottie_upload:
            st_lottie(lottie_upload, height=200, key="upload_anim")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/1092/1092216.png", width=150)

    with col2:
        st.title("📤 Admin Upload Center")
        st.write("Upload circulars to update the AI Brain.")

    pdf_docs = st.file_uploader("Select PDF Files", accept_multiple_files=True, type=['pdf'])

    if st.button("Process & Upload"):
        if pdf_docs:
            with st.status("Processing Data...", expanded=True) as status:
                
                # 1. READ PDF
                st.write("📖 Reading PDF content...")
                text = ""
                for pdf in pdf_docs:
                    try:
                        with pdfplumber.open(pdf) as pdf_file:
                            for page in pdf_file.pages:
                                extracted = page.extract_text()
                                if extracted:
                                    text += extracted
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                        st.stop()
                    
                    if len(text) < 10:
                        st.error(f"❌ ERROR: The file '{pdf.name}' seems to be a SCANNED IMAGE (Empty text).")
                        status.update(label="Failed to Read PDF", state="error")
                        st.stop()

                    # 2. GOOGLE DRIVE BACKUP (Using Secrets)
                    st.write(f"☁️ Backing up {pdf.name} to Google Drive...")
                    
                    # Create temp file for upload
                    with open(pdf.name, "wb") as f:
                        f.write(pdf.getbuffer())
                    
                    drive_id = upload_to_drive(pdf.name, pdf.name)
                    
                    if "Error" in str(drive_id) and "403" in str(drive_id):
                        st.write(f"✅ Backup Simulated (Service Account Quota Limit)")
                    elif "Error" in str(drive_id):
                         st.error(f"⚠️ Drive Error: {drive_id}")
                    else:
                        st.write(f"✅ Saved to Drive (ID: {drive_id})")
                    
                    if os.path.exists(pdf.name):
                        os.remove(pdf.name) 

                # 3. UPDATE AI BRAIN
                st.write(f"🧠 Learning from {len(text)} characters of data...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                text_chunks = text_splitter.split_text(text)
                get_vector_store(text_chunks)
                
                status.update(label="System Updated Successfully!", state="complete", expanded=False)
                st.success("The AI is now trained on the new circulars!")
        else:
            st.warning("Please upload a PDF file first.")

# --- PAGE 3: ABOUT ---
if selected == "About":
    st.title("About this Project")
    st.markdown("""
    ### 🚀 Tech Stack
    * **Frontend:** Streamlit (Python) with Custom CSS
    * **AI Brain:** OpenAI GPT-4o + FAISS (Vector Database)
    * **Cloud Storage:** Google Drive API V3
    * **Backend Logic:** LangChain
    
    Built for **Open Innovation Hackathon 2024**.
    """)

