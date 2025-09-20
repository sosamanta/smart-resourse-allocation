import streamlit as st
from extraction import extract_text_and_tables
from llm import LLMWrapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
import json     
st.title("📂Smart Resume analyzer and project allocation")

# Sidebar inputs
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Resume(s)", type=["pdf"], accept_multiple_files=True
)

job_requirements = st.sidebar.text_area(
    "Paste Job Requirements",
    height=200,
    placeholder="Paste job description or requirements here..."
)
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Threshold Score to Qualify", min_value=0, max_value=100, value=60, step=1)
analyze_btn = st.sidebar.button("Analyze")
if analyze_btn:
    with st.spinner("Analyzing texts with Ollama — this may take a few seconds..."):
        # Initialize LLM
        llm = LLMWrapper()

        if uploaded_files and job_requirements.strip():
            for file in uploaded_files:
                #st.subheader(f"📄 {file.name}")
                text, tables = extract_text_and_tables(file)
                # Run through your LLM extraction pipeline
                llm_response = llm.generate_response(text)
                try:
              
                    compare_response = llm.compare_texts(llm_response, job_requirements) 
                    compare_response = re.search(r"\{.*\}", compare_response, re.DOTALL).group()
                    print('compare',compare_response)
                    response_json = json.loads(compare_response) 
                    print('response',response_json)
                    name = response_json.get("Name","")
                    print('name',name)
                    score = int(response_json.get("Similarity Score", 0)) 
                    similarities = response_json.get( "Key Similarities", "N/A")
                    differences = response_json.get("Key Differences", "N/A")
                    #summary = response_json.get("Overall summary", "N/A")
                    #print('ddddd')
                    
                    st.markdown("### 📊 Analysis Result")
                    st.write(f"**Name:** {name}")
                    st.write(f"**Similarity Score:** {score:.2f}")
                    st.write(f"**Matching Skills:** {similarities[0]}")
                    st.write(f"**Skills gap:** {differences[0]}")
                    #st.write(f"**Overall Summary:** {summary}")

                    if score > threshold:
                        st.success("✅ Strong match with job requirements!")
                    elif threshold-10 <score <threshold:
                        st.warning("⚠️ Partial match. Candidate meets some requirements.")
                    else:
                        st.error("❌ Weak match. Candidate may not fit the role.")
                except Exception as e:
                    st.error(f"Error calculating similarity: {e}")

        else:
            st.info("Upload at least one Resume PDF and paste Job Requirements, then click Analyze.")
