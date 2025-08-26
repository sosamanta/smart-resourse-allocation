import pdfplumber

def extract_text_and_tables(uploaded_file):
    text_output = ""
    table_output = []

    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract text
            text = page.extract_text()
            if text:
                text_output += f"\n--- Page {i+1} ---\n{text}\n"

            # Extract tables
            tables = page.extract_tables()
            for table_no, table in enumerate(tables):
                table_output.append({
                    "page": i+1,
                    "table_no": table_no+1,
                    "data": table
                })

    return text_output, table_output
