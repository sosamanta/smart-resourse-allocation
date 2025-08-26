import streamlit as st
from extraction import extract_text_and_tables
from llm import LLMWrapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
st.title("📂 PDF Resume Analyzer")

# Sidebar inputs
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Resume(s)", type=["pdf"], accept_multiple_files=True
)

job_requirements = st.sidebar.text_area(
    "Paste Job Requirements",
    height=200,
    placeholder="Paste job description or requirements here..."
)

analyze_btn = st.sidebar.button("Analyze")
if analyze_btn:
    with st.spinner("Analyzing texts with Ollama — this may take a few seconds..."):
        # Initialize LLM
        llm = LLMWrapper()

        if uploaded_files and job_requirements.strip():
            for file in uploaded_files:
                st.subheader(f"📄 {file.name}")
                text, tables = extract_text_and_tables(file)

                # Run through your LLM extraction pipeline
                llm_response = llm.generate_response(text)

                # Show LLM extracted text
                if llm_response.strip():
                    st.text_area("Extracted Resume Text", llm_response, height=200)

                # Analyze similarity when button clicked
                #if analyze_btn:
                try:
                    # # TF-IDF Cosine similarity
                    # vectorizer = TfidfVectorizer(stop_words="english")
                    # vectors = vectorizer.fit_transform([llm_response, job_requirements])
                    # score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                    # Compare texts using LLM
                    compare_response = llm.compare_texts(llm_response, job_requirements)
                    st.write("### 🧠 LLM Comparison Response")
                    #st.write("Raw LLM response:",compare_response)
                    print('compare_response:', type(compare_response))
                    # Extract similarity score from LLM response
                    import json     
                    response_json = json.loads(compare_response) #json.loads(compare_response) 
                    print('response_json:', type(response_json))
                    score = float(response_json.get("Similarity score", 0)) / 100.0 
                    print('aaaaaa')
                    similarities = response_json.get("Key similarities", "N/A")
                    print('bbbbb')  
                    differences = response_json.get("Key differences", "N/A")
                    print('ccccc')  
                    summary = response_json.get("Overall summary", "N/A")
                    print('ddddd')
                    
                    st.markdown("### 📊 Analysis Result")
                    st.write(f"**Similarity Score:** {score:.2f}")
                    st.write(f"**Key Similarities:** {similarities}")
                    st.write(f"**Key Differences:** {differences}")
                    st.write(f"**Overall Summary:** {summary}")

                    if score > 75:
                        st.success("✅ Strong match with job requirements!")
                    elif 50 <score <75:
                        st.warning("⚠️ Partial match. Candidate meets some requirements.")
                    else:
                        st.error("❌ Weak match. Candidate may not fit the role.")
                except Exception as e:
                    st.error(f"Error calculating similarity: {e}")

        else:
            st.info("Upload at least one Resume PDF and paste Job Requirements, then click Analyze.")
