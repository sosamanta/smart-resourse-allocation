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

        # LCEL pipeline: prompt → model → string parser
        self.chain = self.prompt | self.model | StrOutputParser()

    def generate_response(self, question: str):
        # Pass inputs here, template will handle it
        return self.chain.invoke({"question": question})



