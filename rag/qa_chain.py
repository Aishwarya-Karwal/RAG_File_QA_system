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
        #context_text = "\n\n---\n\n".join([c["text"] for c in contexts])
        context_blocks = []
        for i, chunk in enumerate(contexts):
            meta = chunk['metadata']
            block = f"""
            [Source {i+1}]
            Text : {chunk['text']}
            Source : {meta['source']}
            Page : {meta['page_number']}"""
            context_blocks.append(block.strip())

        context_text = "\n".join(context_blocks)
        prompt = f"""
        You are a helpful AI assistant answering questions from uploaded documents.

        Use ONLY the provided context to answer the question.

        If the answer is partially available, provide the best possible summarized answer.

        Do NOT say "I don't have enough information" unless the context is completely unrelated.

        Context:
        {context_text}

        Question:
        {query}

        Answer in a clear and concise manner.
        """
        
        return prompt
    

    def answer(self, query : str, contexts: List[Dict]) -> str:
        """
        Calls to Builds the prompt and sends it to the LLM, and returns its response.
        """

        prompt = self.build_prompt(query, contexts)
        response = self.model.generate_content(prompt)

        return response.text
