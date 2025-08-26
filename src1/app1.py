import streamlit as st
from extraction import extract_text_and_tables
from llm import LLMWrapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        if analyze_btn:
            try:
                # TF-IDF Cosine similarity
                vectorizer = TfidfVectorizer(stop_words="english")
                vectors = vectorizer.fit_transform([llm_response, job_requirements])
                score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

                st.markdown("### 📊 Analysis Result")
                st.write(f"**Similarity Score:** {score:.2f}")

                if score > 0.75:
                    st.success("✅ Strong match with job requirements!")
                elif score > 0.5:
                    st.warning("⚠️ Partial match. Candidate meets some requirements.")
                else:
                    st.error("❌ Weak match. Candidate may not fit the role.")
            except Exception as e:
                st.error(f"Error calculating similarity: {e}")

else:
    st.info("Upload at least one Resume PDF and paste Job Requirements, then click Analyze.")
