from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


class LLMWrapper:
    def __init__(self, model_name="llama3.1"):
        self.model = ChatOllama(model=model_name)
        # self.system_message = "You are an expert resume parser. Your task is to read the given " \
        # "resume and extract only the technologies and relevant project  experiences. Be concise, structured," \
        # " and avoid unnecessary text.and format the output in a string."
        self.system_message = f"""
        You are an assistant that extracts only the requested fields from a candidate's resume. Do not include anything else.
        Given the resume text, extract the following  section:
            1. Name :  Canditure name
            2. ProjectExperience :  A summary of all project experience the candidate has worked on. Combine all projects into a single descriptive paragraph or list in a concise format. Do not include project names, durations, or unrelated details.
            3. TechnologiesUsed : A concise,comma-separated string of all technologies, tools, or programming languages used across projects.
        """
        # Keep prompt template, don't call .format() here
        self.prompt = ChatPromptTemplate.from_template(
            self.system_message + "\n\nQuestion: {question}\n\n"
        )

        # self.prompt1 = ChatPromptTemplate.from_template(self.compare_prompt+ "\n\n{text1}\n\n{text2}")

        # LCEL pipeline: prompt → model → string parser
        self.chain = self.prompt | self.model | StrOutputParser()
        
    def generate_response(self, question: str):
        # Pass inputs here, template will handle it
        return self.chain.invoke({"question": question})
    ##Compare two texts
    def compare_texts(self, text1: str, text2: str):
        
        # self.compare_prompt = f"""
        #         You are a text comparison assistant. Compare the following two texts:
        #         canditure resume:
        #         {text1}
        #         project requirements:
        #         {text2}
        #         Return the result in this json format with the following sections:
        #         1. Name : mentioned the canditure name in resume.
        #         2. Similarity score (0–100)
        #         3. Key similarities :  mentioned  similarity of the canditure resume with the project requirements .do not include name, 
        #         4. Key differences :   mentioned skills gaps of the canditure resume with the project requirements .if any thing from the canditure resume is present or simlar to any topic of project requirements, do not consider
        #         it as a skills gap .
        #         ** stritly follow do not include first text or second text in output instead use candiatature resume,  project requirements also output shoulbe in json format.

        #         """
        
        self.compare_prompt = f"""
            You are a text comparison assistant. Compare the following two texts:

            Candidature resume:
            {text1}

            Project requirements:
            {text2}

            Return the result strictly in the following JSON format:

            "Name": Extract the candidate's name from the resume,
            "Similarity Score": An integer between 0–100 representing overall similarity,
            "Key Similarities": Concise description of the main overlaps between the candidature resume and the project requirements (do not include name),
            "Key Differences": Concise description of the missing skills, technologies, or experiences in the candidature resume compared to the project requirements. Do not mark anything as a gap if it is already present or similar in the candidature resume.
            

            Guidelines:
            - Do not include the raw text of either the resume or the requirements in the output.
            - Always refer to them as "candidature resume" and "project requirements".
            - The output must be valid JSON only, with no additional text or explanation outside the JSON. 
                """
        self.prompt1 = ChatPromptTemplate.from_template(self.compare_prompt+ "\n\n{text1}\n\n{text2}")
        self.chain1 = self.prompt1 | self.model | StrOutputParser()
        return self.chain1.invoke({"text1": text1, "text2": text2})



