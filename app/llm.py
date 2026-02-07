import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3-flash-preview")

def generate_answer(question, chunks):
    context = "\n\n".join(chunks)

    prompt = f"""
You are a legal assistant.

Answer the question ONLY using the context below.
If the answer is not present, say "Not found in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)
    return response.text.strip()
