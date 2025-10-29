from google import genai
from pydantic import BaseModel
import json
import PyPDF2
from PyPDF2 import PdfReader
import os
from groq import Groq
from dotenv import load_dotenv
from google.genai import types
import notification
import time

load_dotenv()


def Clause_extraction(file):
    print("inside clause extraction")

    class ClauseExtraction(BaseModel):
        clause_id: str
        heading: str
        text: str

    text = ""
    try:
        with open(file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as pdf_error:
        notification.notify_all("PDF Parsing Error", f"File: {file}\nError: {pdf_error}")
        return None

    prompt = f"""
    You are an expert in legal contract analysis.
    Extract clauses precisely from the contract text and return as valid JSON.
    Input: {text}
    """

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=list[ClauseExtraction],
            ),
        )
        return response.text
    except Exception as e:
        notification.notify_all("Gemini API (Clause_extraction) failed", f"Switching to Groq. Error: {e}")
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY_1"))
            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a legal clause extractor."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e2:
            notification.notify_all("Groq Fallback Failed", f"Error: {e2}")
            return None


def Clause_extraction_with_summarization(file):
    print("inside clause extraction with summarization")

    class ClauseExtraction(BaseModel):
        clause_id: str
        heading: str
        summarised_text: str

    text = ""
    try:
        with open(file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as pdf_error:
        notification.notify_all("PDF Parsing Error", f"File: {file}\nError: {pdf_error}")
        return None

    prompt = f"""
    Extract all clauses and provide concise summaries.
    Input: {text}
    """

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_2"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=list[ClauseExtraction],
            ),
        )
        return response.text
    except Exception as e:
        notification.notify_all("Gemini API (Clause_extraction_with_summarization) failed", f"Switching to Groq. Error: {e}")
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY_2"))
            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a contract summarizer."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e2:
            notification.notify_all("Groq Fallback Failed", f"Error: {e2}")
            return None


def summarize_clause_text(clause_text):
    prompt = f"""
    Summarize this legal clause text briefly but accurately.
    Input: {clause_text}
    """

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_3"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.3
            ),
        )
        return response.text
    except Exception as e:
        notification.notify_all("Gemini API (summarize_clause_text) failed", f"Switching to Groq. Error: {e}")
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY_3"))
            response = groq_client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a legal clause summarizer."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e2:
            notification.notify_all("Groq Fallback Failed", f"Error: {e2}")
            return None
