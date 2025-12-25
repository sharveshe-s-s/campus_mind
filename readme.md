# 🎓 CampusMind AI
### Your 24/7 Smart Campus Copilot | 🏆 Hackathon Edition

**CampusMind AI** is a next-generation Retrieval-Augmented Generation (RAG) application designed to bridge the information gap between campus administration and students. It transforms static, hard-to-find PDF circulars into an interactive, voice-enabled AI conversation that provides instant, accurate answers.

---

## 💡 The Problem
In modern campuses, critical information is often buried in hundreds of emails, PDF attachments, and physical notice boards. Students struggle to find simple answers—like *"When is the revaluation deadline?"* or *"What is the dress code for the internship fair?"*—leading to confusion and administrative bottlenecks.

## 🚀 The Solution
**CampusMind AI** serves as a centralized, intelligent brain for the campus.
1.  **Admins** securely upload circulars via a dedicated portal.
2.  **The AI** instantly indexes, vectorizes, and "learns" the content.
3.  **Students** ask questions via text or voice and get instant, context-aware answers cited directly from official documents.

---

## ✨ Key Features

### 🤖 Intelligent Student Chat
* **RAG Pipeline:** Powered by **LangChain** and **FAISS** to retrieve precise, context-aware answers from uploaded PDFs.
* **Voice-First Interface:** Integrated **OpenAI Whisper** allows students to ask questions verbally for a seamless hands-free experience.
* **Streaming Responses:** Real-time typing effect for a premium, conversational AI feel.
* **Smart Memory:** The AI remembers context from previous turns in the conversation.

### ⚡ Real-Time Global Sync
* **Instant Updates:** Powered by **Google Drive API**, when an admin uploads a file, it becomes instantly available to all students across all devices.
* **Hybrid Caching:** Combines server-side caching with real-time Drive fetching to ensure zero latency for critical updates.

### 🛡️ Admin Portal
* **Secure Uploads:** Drag-and-drop interface for processing multiple PDF circulars simultaneously.
* **Automated Vectorization:** Automatically chunks, embeds, and indexes documents into the vector database upon upload.
* **Persistent Knowledge:** Merges new uploads with the existing database, ensuring the AI "grows" smarter over time without forgetting old circulars.

### 🎨 Premium UI/UX
* **Glassmorphism Design:** Modern, translucent card layout with a futuristic aesthetic.
* **Dark Mode:** Custom-engineered dark theme for reduced eye strain and professional visual appeal.
* **Responsive Layout:** Optimized for projectors (demos) and mobile devices.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | **Streamlit** | Python-based web framework with custom CSS injection for the UI. |
| **LLM** | **OpenAI GPT-4o-mini** | The reasoning engine generating the answers. |
| **Embeddings** | **OpenAI text-embedding-3** | High-performance model converting text into vector representations. |
| **Vector DB** | **FAISS** | Facebook AI Similarity Search for high-speed dense vector retrieval. |
| **Storage** | **Google Drive API** | Cloud storage for raw PDF files and global synchronization. |
| **Orchestration** | **LangChain** | Manages the RAG pipeline, prompt engineering, and chains. |
| **Speech** | **OpenAI Whisper** | High-accuracy speech-to-text transcription for voice queries. |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/sharveshe-s-s/campus_mind.git](https://github.com/sharveshe-s-s/campus_mind.git)
cd campusmind-ai

2. Install Dependencies
Bash
pip install -r requirements.txt

3. Configure Secrets
Create a folder named .streamlit in the root directory and add a file named secrets.toml.You need to add your OpenAI API Key and your Google Service Account JSON..streamlit/secrets.toml Format:Ini, TOMLOPENAI_API_KEY = "sk-proj-..."

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n..."
client_email = "your-email@your-project.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "[https://accounts.google.com/o/oauth2/auth](https://accounts.google.com/o/oauth2/auth)"
token_uri = "[https://oauth2.googleapis.com/token](https://oauth2.googleapis.com/token)"
auth_provider_x509_cert_url = "[https://www.googleapis.com/oauth2/v1/certs](https://www.googleapis.com/oauth2/v1/certs)"
client_x509_cert_url = "..."

4. Run the AppBashstreamlit run app.py
