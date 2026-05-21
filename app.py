import streamlit as st
import pandas as pd
from transformers import pipeline
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.llms import HuggingFacePipeline

# ------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="AI CSV Data Assistant",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# Title
# ------------------------------------------------
st.title("📊 AI CSV Data Assistant")
st.markdown("Upload a CSV file and ask questions using natural language.")

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.header("⚙️ Configuration")

model_name = st.sidebar.selectbox(
    "Choose Hugging Face Model",
    [
        "google/flan-t5-base",
        "google/flan-t5-large",
        "tiiuae/falcon-rw-1b"
    ]
)

# ------------------------------------------------
# Load Hugging Face Model
# ------------------------------------------------
@st.cache_resource
def load_llm(model_name):

    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        max_new_tokens=256,
        temperature=0.3
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

# ------------------------------------------------
# Upload CSV File
# ------------------------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.success("✅ CSV File Uploaded Successfully")

    # ------------------------------------------------
    # Display Dataset
    # ------------------------------------------------
    st.subheader("📌 Dataset Preview")

    st.dataframe(df.head())

    # Dataset Information
    st.subheader("📈 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # ------------------------------------------------
    # Load LLM
    # ------------------------------------------------
    with st.spinner("Loading Hugging Face Model..."):
        llm = load_llm(model_name)

    # ------------------------------------------------
    # Create Pandas AI Agent
    # ------------------------------------------------
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True
    )

    # ------------------------------------------------
    # Query Section
    # ------------------------------------------------
    st.subheader("💬 Ask Questions About Your Data")

    user_query = st.text_input(
        "Enter your query",
        placeholder="Example: What is the average salary?"
    )

    # ------------------------------------------------
    # Process Query
    # ------------------------------------------------
    if st.button("Generate Answer"):

        if user_query.strip() == "":
            st.warning("Please enter a question.")
        else:

            with st.spinner("Analyzing dataset..."):

                try:
                    response = agent.run(user_query)

                    st.subheader("🤖 AI Response")

                    st.success(response)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # ------------------------------------------------
    # Optional Data Analytics
    # ------------------------------------------------
    st.subheader("📊 Quick Data Analytics")

    if st.checkbox("Show Statistical Summary"):
        st.write(df.describe())

    if st.checkbox("Show Column Names"):
        st.write(df.columns.tolist())

    if st.checkbox("Show Missing Values"):
        st.write(df.isnull().sum())

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.markdown("🚀 Developed using Streamlit + Hugging Face + LangChain")
