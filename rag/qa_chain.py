from typing import List, Dict
# from openai import OpenAI
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

class QAChain:
    """
    A simple Retrieval-Augmented QA Chain.
    Takes in:
    - query (string)
    - retrieved chunks (list of dicts with text and index) which we are reffering as context
    Sends to LLM:
    "Answer using ONLY the context but be concise."
    """
    def __init__(self, model_name = "gemini-flash-latest"):
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def build_prompt(self, query:str, contexts: List[Dict]) -> str:
        """
        Construct the prompt for the LLM by injecting the reriiteved chunks
        """
        context_text = "\n\n---\n\n".join([c["text"] for c in contexts])
        prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the question.

        Context:
        {context_text}

        Question: {query}

        If answer is not in the context, say "I don't have enough information."
        Keep your answer short and clear.
        """.strip()
        
        return prompt
    

    def answer(self, query : str, contexts: List[Dict]) -> str:
        """
        Calls to Builds the prompt and sends it to the LLM, and returns its response.
        """

        prompt = self.build_prompt(query, contexts)
        response = self.model.generate_content(prompt)

        return response.text
