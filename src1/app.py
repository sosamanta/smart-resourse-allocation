import streamlit as st
from extraction import extract_text_and_tables
from llm import LLMWrapper
st.title("📂 PDF Extractor")

uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
##Job requirements
job_requirements = st.sidebar.text_area(
    "Paste Job Requirements",
    height=200,
    placeholder="Paste job description or requirements here..."
)
print('data',job_requirements)

run_btn = st.sidebar.button("Upload")
llm = LLMWrapper()
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 {file.name}")
        text, tables = extract_text_and_tables(file)

        llm_response = llm.generate_response(text)
        # Show extracted text
        if llm_response.strip():
            st.text_area("Extracted Text", llm_response, height=200)

        # # Show extracted tables
        # if tables:
        #     for tbl in tables:
        #         st.write(f"Table {tbl['table_no']} (Page {tbl['page']})")
        #         st.table(tbl["data"])
else:
    st.info("Upload a PDF to extract text and tables.")
