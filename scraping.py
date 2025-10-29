import requests, json
import data_extration
import time
import os
import notification
from groq import Groq
from google import genai
from dotenv import load_dotenv

load_dotenv()

# scrape data from different link using get api 
def scrape_data(url, name):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download Successful", time.ctime())
        notification.notify_all("Template Downloaded", f"Downloaded successfully from {url}")
    else:
        print("Failed to download", response.status_code)
        notification.notify_all("Scraping Failed", f"Failed to download file from {url}. "
                                f"Status code: {response.status_code}, URL: {url}\nError: {response.text}")

def call_scrape_funtion():
    try:
        DOCUMENT_MAP = {
            "DPA": {
                "json_file": "json_files/dpa.json",
                "link": r"https://www.benchmarkone.com/wp-content/uploads/2018/05/GDPR-Sample-Agreement.pdf"
            },
            "JCA": {
                "json_file": "json_files/jca.json",
                "link": r"https://www.surf.nl/files/2019-11/model-joint-controllership-agreement.pdf"
            },
            "C2C": {
                "json_file": "json_files/c2c.json",
                "link": r"https://www.fcmtravel.com/sites/default/files/2020-03/2-Controller-to-controller-data-privacy-addendum.pdf"
            },
            "SCC": {
                "json_file": "json_files/scc.json",
                "link": r"https://www.thomsonreuters.com/content/dam/ewp-m/documents/thomsonreuters/en/pdf/global-sourcing-procurement/eu-eea-standard-contractual-clauses-v09-2021.pdf"
            },
            "subprocessing": {
                "json_file": "json_files/subprocessing.json",
                "link": r"https://greaterthan.eu/wp-content/uploads/Personal-Data-Sub-Processor-Agreement-2024-01-24.pdf"
            }
        }

        temp_agreement = "temp_agreement.pdf"

        for key, value in DOCUMENT_MAP.items():
            # Step 1: Download
            scrape_data(DOCUMENT_MAP[key]["link"], temp_agreement)

            try:
                # Step 2: Extract clauses (Gemini first)
                clauses = data_extration.Clause_extraction(temp_agreement)
            except Exception as e:
                print(f"Gemini error in scraping for {key}: {e}")
                if "RESOURCE_EXHAUSTED" in str(e) or "overloaded" in str(e):
                    try:
                        print("Fallback to Groq for clause extraction...")
                        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY1"))
                        prompt = f"Extract GDPR clauses clearly from the text:\n\n{temp_agreement}"
                        chat_completion = groq_client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            model="llama-3.3-70b-versatile",
                        )
                        clauses = chat_completion.choices[0].message.content
                    except Exception as groq_error:
                        print(f"Groq fallback failed: {groq_error}")
                        notification.notify_all("Scraping Fallback Failed", f"{key} extraction failed after Groq fallback.")
                        continue
                else:
                    notification.notify_all("Clause Extraction Error", f"{key} clause extraction failed: {e}")
                    continue

            # Step 3: Update JSON file
            with open(DOCUMENT_MAP[key]["json_file"], "w", encoding="utf-8") as f:
                json.dump(clauses, f, indent=2, ensure_ascii=False)
            
            notification.notify_all("Template Updated", f"{key} template refreshed successfully.")
            print(f"{key} template refreshed successfully.")

    except Exception as e:
        print("Error Occurred", e)
        notification.notify_all("Error occurred in scraping function", f"Error is {e}")

# call_scrape_funtion()
