**KG_CHATBOT - Hospital Knowledge Graph Assistant **
---
KG_CHATBOT is a powerful, context-aware chatbot designed to serve as an intelligent assistant for KG Hospital. It leverages advanced Large Language Models (LLMs) and Vector Databases to provide accurate, real-time answers based on uploaded hospital documents (PDFs and Excel files).

The application is built on Streamlit for the front-end, uses LangChain for RAG (Retrieval-Augmented Generation) pipeline, and is backed by Firebase Storage for secure and scalable document management.
---
**Features**

Intelligent RAG System: Uses a Conversational Retrieval Chain powered by a Groq LLM (llama-3.3-70b-versatile) and a FAISS vector store for efficient, context-specific responses.

Multi-Document Support: Capable of processing and embedding knowledge from both PDFs (using multiple robust loaders) and Excel files (converting structured data into searchable text).

Incremental Loading: Automatically checks and loads new documents from Firebase Storage periodically (every 30 seconds) without interrupting the running application, ensuring the knowledge base is always up-to-date.

Professional Formatting: Implements a custom response formatting layer to ensure LLM output is highly readable, using:

Markdown headers (e.g., ## Symptoms:)

Formatted bullet points and numbered lists

Conditional headers for key topics (e.g., Medication Information:).

Resilient Fallback: If a document-based answer is not found, the chatbot defaults to a helpful, positive, and actionable response, guiding the user to contact the hospital directly.

Clean UI: Features a custom-styled, dark-mode Streamlit interface for a professional and modern user experience.
---
**Technologies Used**

Category	Technology	Purpose
Framework	Python 3.x, Streamlit	Application structure and web interface.

LLM & Hosting	Groq (via langchain-groq)	Fast, high-performance inference for conversation.

Vector DB	FAISS, HuggingFaceEmbeddings (all-MiniLM-L6-v2)	Document embedding, indexing, and retrieval.

RAG Pipeline	LangChain	Orchestration of the conversational retrieval chain.

Data Storage	Firebase Storage	Cloud storage for source documents (PDFs, Excel).

Configuration	python-dotenv, Streamlit Secrets	Environment variables and secure secret management.
---
**Export to Sheets**
**Getting Started**
Follow these steps to set up and run the KG_CHATBOT application locally.
---
Prerequisites
Python 3.8+

A Groq API Key

A Firebase Project with Storage enabled and Admin SDK credentials.
---
**1. Setup Environment**
Clone the repository and install the necessary dependencies:

Bash

git clone https://github.com/ARISH4651/v3-KG_CHATBOT.git
cd v3-KG_CHATBOT
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt 
# Note: requirements.txt should include packages like streamlit, langchain-groq,
# langchain-community, faiss-cpu, python-dotenv, firebase-admin, pandas, openpyxl, etc.
----
**2. Configure Secrets**
The application requires two main secrets: the Groq API Key and Firebase Admin SDK credentials.

A. Groq API Key

The application uses python-dotenv to load environment variables. Create a file named .env in the root directory and add your Groq API Key:

Code snippet

# .env
GROQ_API_KEY="your_groq_api_key_here" 
B. Firebase Credentials (Streamlit Secrets)

The application initializes Firebase Admin SDK using Streamlit secrets. For deployment with Streamlit, the Firebase credentials should be stored in the .streamlit/secrets.toml file (create this file and folder if they don't exist).

Ini, TOML

# .streamlit/secrets.toml

[firebase]
type = "service_account"
project_id = "your-firebase-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n" # Use actual credentials from Firebase Admin SDK JSON file
client_email = "firebase-adminsdk-..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
---
▶️ Running the Application
Execute the Streamlit application from your terminal:

Bash

streamlit run main.py
The app will open in your web browser. It will automatically attempt to connect to Firebase, download any existing documents, build the vector store, and initialize the chat chain.

➕ Document Management
Upload Documents: Upload your hospital knowledge files (PDFs, .xlsx, or .xls) directly to your Firebase Storage bucket.

Automatic Load: The chatbot will automatically detect and process the new files every 30 seconds. Use the "Check for New Files" button in the sidebar for an immediate manual check.

Chat: Once documents are loaded, you can ask questions, and the chatbot will retrieve answers from the provided hospital data.
---







