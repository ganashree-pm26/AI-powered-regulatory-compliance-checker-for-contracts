from google import genai
from google.genai import types
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from enum import Enum
import PyPDF2
from PyPDF2 import PdfReader
import json
from groq import Groq
import notification

load_dotenv()


# ********   Phase 2    ******** #
def document_type(file):
    class DocumentType(str, Enum):
        DPA = "Data Processing Agreement"
        JCA = "Joint Controller Agreement"
        C2C = "Controller-to-Controller Agreement"
        subprocessor = "Processor-to-Subprocessor Agreement"
        SCC = "Standard Contractual Clauses"
        NoOne = "NoOne"

    class FindDocumentType(BaseModel):
        document_type: DocumentType

    text = ""
    with open(file, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()

    prompt = f"""
        Tell me what type of document is this

        document should be type of between 
        1. Data Processing Agreement
        2. Joint Controller Agreement
        3. Controller-to-Controller Agreement
        4. Processor-to-Subprocessor Agreement
        5. Standard Contractual Clauses
        
        Input: {text}
        
        Response in this JSON Structure:
        [{{
            "document_type": "<type_of_document>"
        }}]
    """

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_1"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=list[FindDocumentType],
            ),
        )
        json_object = json.loads(response.text)
        return json_object[0]["document_type"]
    except Exception as e:
        notification.notify_all("Gemini API (doc type) failed", f"Switching to Groq. Error: {e}")
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY_1"))
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are a legal document classifier."},
                    {"role": "user", "content": prompt},
                ],
            )
            text_response = response.choices[0].message.content
            json_object = json.loads(text_response)
            return json_object[0]["document_type"]
        except Exception as e2:
            notification.notify_all("Groq Fallback Failed", f"Error: {e2}")
            return "NoOne"


def compare_agreements(unseen_data, template_data):
    prompt = f"""
    You are an AI legal assistant specialized in contract review and compliance.

    Compare the two documents below:

    Template document (regulatory standard reference): 
    {template_data}

    New contract document to review:
    {unseen_data}

    Tasks:
    1. Identify any missing or altered clauses in the new contract compared to the template.
    2. Flag potential compliance risks based on GDPR regulations.
    3. Assign a risk score between 0 and 100.
    4. Provide reasoning and recommendations in concise format.
    """

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_2"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.3
            ),
        )
        print(response.text)
        return response.text
    except Exception as e:
        notification.notify_all("Gemini API (compare_agreements) failed", f"Switching to Groq. Error: {e}")
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY_2"))
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are an AI legal assistant specialized in contract comparison."},
                    {"role": "user", "content": prompt},
                ],
            )
            text_response = response.choices[0].message.content
            print(text_response)
            return text_response
        except Exception as e2:
            notification.notify_all("Groq Fallback Failed", f"Error: {e2}")
            return "Error during comparison"


def risk_score_analysis(result_text):
    prompt = f"""
    Analyze the following compliance comparison result and calculate a refined risk score (0–100),
    highlighting high-risk areas briefly.

    Input:
    {result_text}

    Response format:
    - Final Risk Score: <score>
    - Key High-Risk Clauses: [...]
    - Justification: [...]
    """

    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY_3"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.2
            ),
        )
        return response.text
    except Exception as e:
        notification.notify_all("Gemini API (risk_score_analysis) failed", f"Switching to Groq. Error: {e}")
        try:
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY_3"))
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are an AI compliance risk assessor."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e2:
            notification.notify_all("Groq Fallback Failed", f"Error: {e2}")
            return "Error during risk score analysis"
