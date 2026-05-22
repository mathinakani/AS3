import os
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from langchain_experimental.agents import create_pandas_dataframe_agent

from langchain.agents.agent_types import AgentType

# OPTIONAL GROQ
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False


# ============================================================
# LOAD ENV VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI CSV Data Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Enterprise AI CSV Data Assistant")

st.markdown("""
### Features
✅ Upload CSV Dataset
✅ Natural Language Querying
✅ AI-powered Data Analysis
✅ Pandas Agent Integration
✅ Real-time Answers
✅ Hugging Face / Groq Support
""")


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("⚙ Settings")

llm_option = st.sidebar.selectbox(
    "Choose AI Model",
    [
        "HuggingFace",
        "Groq"
    ]
)


# ============================================================
# LOAD HUGGING FACE MODEL
# ============================================================

@st.cache_resource
def load_huggingface_model():

    model_name = "google/flan-t5-base"

    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
        max_new_tokens=256,
        temperature=0.3
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return llm


# ============================================================
# LOAD GROQ MODEL
# ============================================================

@st.cache_resource
def load_groq_model():

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0
    )

    return llm


# ============================================================
# SELECT LLM
# ============================================================

if llm_option == "Groq":

    if not GROQ_AVAILABLE:
        st.error("langchain-groq not installed.")
        st.stop()

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in .env")
        st.stop()

    llm = load_groq_model()

else:

    llm = load_huggingface_model()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

# ============================================================
# PROCESS CSV
# ============================================================

if uploaded_file is not None:

    try:

        # READ CSV
        df = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully ✅")

        # ====================================================
        # DATASET PREVIEW
        # ====================================================

        st.subheader("📌 Dataset Preview")

        st.dataframe(df.head())

        # ====================================================
        # DATASET INFORMATION
        # ====================================================

        st.subheader("📈 Dataset Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())

        # ====================================================
        # COLUMN NAMES
        # ====================================================

        st.subheader("📋 Column Names")

        st.write(list(df.columns))

        # ====================================================
        # CREATE PANDAS AGENT
        # ====================================================

        with st.spinner("Creating AI Agent..."):

            agent = create_pandas_dataframe_agent(
                llm=llm,
                df=df,
                verbose=False,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
                max_iterations=5
            )

        st.success("AI Agent Ready ✅")

        # ====================================================
        # USER QUERY INPUT
        # ====================================================

        st.subheader("💬 Ask Questions About Dataset")

        user_query = st.text_input(
            "Enter your question",
            placeholder="Example: What is the average salary?"
        )

        # ====================================================
        # PROCESS QUERY
        # ====================================================

        if st.button("Generate Answer"):

            if user_query.strip() == "":
                st.warning("Please enter a question.")

            else:

                with st.spinner("Analyzing dataset..."):

                    try:

                        # NEW LANGCHAIN METHOD
                        response = agent.invoke(user_query)

                        st.subheader("🤖 AI Response")

                        # HANDLE RESPONSE FORMAT
                        if isinstance(response, dict):

                            if "output" in response:
                                st.success(response["output"])

                            else:
                                st.write(response)

                        else:
                            st.success(response)

                    except Exception as e:

                        st.error(f"Error: {str(e)}")

    except Exception as e:

        st.error(f"CSV Error: {str(e)}")

else:

    st.info("Please upload a CSV file to begin.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("### 🚀 Developed Using")
st.markdown("""
- Streamlit
- LangChain
- Pandas
- Hugging Face
- Groq LLM
""")
