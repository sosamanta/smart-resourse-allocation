from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


class LLMWrapper:
    def __init__(self, model_name="llama3.1"):
        self.model = ChatOllama(model=model_name)
        self.system_message = "You are an expert resume parser. Your task is to read the given resume and extract only the technologies and relevant experiences. Be concise, structured, and avoid unnecessary text.and format the output in a bullet point list."
        
        # Keep prompt template, don't call .format() here
        self.prompt = ChatPromptTemplate.from_template(
            self.system_message + "\n\nQuestion: {question}\n\nAnswer: Let's think step by step."
        )

        # self.prompt1 = ChatPromptTemplate.from_template(self.compare_prompt+ "\n\n{text1}\n\n{text2}")

        # LCEL pipeline: prompt → model → string parser
        self.chain = self.prompt | self.model | StrOutputParser()
        
    def generate_response(self, question: str):
        # Pass inputs here, template will handle it
        return self.chain.invoke({"question": question})
    ##Compare two texts
    def compare_texts(self, text1: str, text2: str):
        self.compare_prompt = f"""
                You are a text comparison assistant. Compare the following two texts:
                Text 1:
                {text1}
                Text 2:
                {text2}
                Return the result in this json format with the following sections:
                1. Similarity score (0–100)
                2. Key similarities
                3. Key differences
                4. Overall summary
                """
        self.prompt1 = ChatPromptTemplate.from_template(self.compare_prompt+ "\n\n{text1}\n\n{text2}")
        self.chain1 = self.prompt1 | self.model | StrOutputParser()
        return self.chain1.invoke({"text1": text1, "text2": text2})



