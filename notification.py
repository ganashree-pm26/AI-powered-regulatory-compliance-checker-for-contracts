import smtplib
from email.mime.text import MIMEText
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def send_notification(subject, notification):
    try:
        sender = os.getenv("sender_email")
        password = os.getenv("EMAIL_PASSWORD")
        receiver = os.getenv("reciever_email")

        msg = MIMEText(f"{notification}")
        msg["Subject"] = subject
        msg["From"] = f"Ganashree <{sender}>"
        msg["To"] = receiver

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)

        print("Email sent Successfully!")
    except Exception as e:
        print("Error Occurred", e)


#  Added Slack Integration
def send_slack_notification(message):
    try:
        webhook_url = os.getenv("SLACK_URL")
        payload = {
            "text": message,
            "username": "ComplianceBot 🕵️‍♀️",
            "icon_emoji": ":shield:"
        }
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            print("Slack notification sent successfully!")
        else:
            print("Slack notification failed:", response.text)
    except Exception as e:
        print("Slack notification error:", e)


#  Combined function to send both Email + Slack
def notify_all(subject, message):
    send_notification(subject, message)
    send_slack_notification(f"*{subject}*\n{message}")
