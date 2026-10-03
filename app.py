import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
import glob

# ================= Configuration =================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(APP_DIR, "chroma_db")
# For deployment, we'll assume notes are in a 'notes' folder inside the app directory
# But for local fallback, we check the parent directory
LOCAL_NOTES_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
DEPLOY_NOTES_DIR = os.path.join(APP_DIR, "notes")

DOCS_DIR = DEPLOY_NOTES_DIR if os.path.exists(DEPLOY_NOTES_DIR) else LOCAL_NOTES_DIR

st.set_page_config(page_title="RHJS Judicial Mastery App", layout="wide", page_icon="⚖️")

# ================= Custom CSS =================
st.markdown("""
<style>
    .main-header { font-size: 40px !important; font-weight: bold; color: #1E3A8A; text-align: center;}
    .sub-header { font-size: 20px !important; color: #4B5563; text-align: center; margin-bottom: 30px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚖️ RHJS Judicial Mastery System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">World-Class AI Coach (Zero Error & 100% Grounded)</p>', unsafe_allow_html=True)

# ================= Sidebar =================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Scale_of_justice_2.svg/512px-Scale_of_justice_2.svg.png", width=100)
    st.header("⚙️ AI Configuration")
    
    ai_model = st.selectbox("Select AI Model", ["Gemini 1.5 Pro", "Llama-3 70B (Groq)"])
    
    if ai_model == "Gemini 1.5 Pro":
        api_key = st.text_input("Enter Gemini API Key", type="password")
    else:
        api_key = st.text_input("Enter Groq API Key", type="password")
        
    st.markdown("---")
    st.subheader("📚 Knowledge Base Setup")
    if st.button("Index My Notes"):
        with st.spinner("Indexing notes... This may take time."):
            try:
                documents = []
                files = glob.glob(os.path.join(DOCS_DIR, "*.pdf")) + glob.glob(os.path.join(DOCS_DIR, "*.docx"))
                for file in files[:15]: # Limit for performance
                    try:
                        if file.endswith('.pdf'): documents.extend(PyPDFLoader(file).load())
                        elif file.endswith('.docx'): documents.extend(Docx2txtLoader(file).load())
                    except: pass
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                
                # Using free HuggingFace embeddings so it works without Google API key
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
                vectorstore.persist()
                st.success("✅ Knowledge Base built successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

# ================= Helper Functions =================
def get_llm():
    if not api_key: return None
    if ai_model == "Gemini 1.5 Pro":
        os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
    else:
        os.environ["GROQ_API_KEY"] = api_key
        return ChatGroq(model_name="llama3-70b-8192", temperature=0.1)

def search_local_and_web(query):
    local_context = "No local database found."
    if os.path.exists(DB_DIR):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        local_docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(query)
        local_context = "\n\n".join([doc.page_content for doc in local_docs])
    try:
        web_context = DuckDuckGoSearchRun().invoke(f"Rajasthan High Court Supreme Court recent judgment on {query}")
    except:
        web_context = "Web search unavailable."
    return local_context, web_context

# ================= App Tabs =================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 AI Research", "📝 Answer Evaluator", "⚖️ Bare Act Nexus", 
    "🎯 Prelims MCQ", "📜 Judgment Studio", "📚 Language Coach"
])

# Define a generic function to run prompts
def run_ai_task(system_prompt, human_prompt_vars, spinner_text):
    if not api_key:
        st.error("Please enter the API Key in the sidebar.")
        return
    with st.spinner(spinner_text):
        llm = get_llm()
        if not llm: return
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{query}")])
        res = (prompt | llm).invoke(human_prompt_vars)
        st.markdown(res.content)

# Tab 1
with tab1:
    query = st.text_input("Ask a legal question:")
    if st.button("Search & Answer"):
        local_ctx, web_ctx = search_local_and_web(query)
        system = f"You are an RHJS Judge. Use Local Notes and Web Search. Local: {local_ctx} Web: {web_ctx}"
        run_ai_task(system, {"query": query}, "Researching...")

# Tab 2
with tab2:
    q = st.text_input("Paste Question:")
    a = st.text_area("Paste Answer:")
    if st.button("Evaluate"):
        sys = "Act as RHJS Examiner. Evaluate answer on Law, Cases, and Intro/Conclusion. Give Topper Blueprint."
        run_ai_task(sys, {"query": f"Q: {q}\nA: {a}"}, "Evaluating...")

# Tab 3
with tab3:
    topic = st.text_input("Topic to compare:")
    if st.button("Compare"):
        run_ai_task("Create table comparing Old (IPC/CrPC/IEA) and New Law (BNS/BNSS/BSA).", {"query": topic}, "Mapping laws...")

# Tab 4
with tab4:
    mcq_topic = st.selectbox("Subject:", ["BNS", "BNSS", "BSA", "CPC", "Hindu Law"])
    if st.button("Generate Question"):
        run_ai_task("Generate 1 tough RHJS MCQ on the topic with detailed explanation.", {"query": mcq_topic}, "Generating MCQ...")

# Tab 5
with tab5:
    st.markdown("Practice drafting bench-grade documents.")
    if st.button("Get Case Scenario"):
        run_ai_task("Provide a brief practical case scenario for a Bail hearing. End by asking user to draft.", {"query": ""}, "Drafting facts...")

# Tab 6
with tab6:
    lang = st.radio("Mode:", ["Hindi Essay Topics", "English Legal Essay", "Grammar Practice"])
    if st.button("Generate Task"):
        run_ai_task("Give a Judicial Service Language Paper task.", {"query": lang}, "Preparing task...")
