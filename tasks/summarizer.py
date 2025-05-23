import os
import sys
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

# Load .env and API key
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('MODEL')

MODEL = None
if model == "openai":
    MODEL="gpt-4o-mini"
elif model =="llama":
    MODEL="llama3.2"
    api_key="ollama"
else:
    print("❌ No model matched, check .env")
    sys.exit(1)

# Validate key
if not api_key:
    print("❌ No API key found. Set OPENAI_API_KEY in your .env file.")
    sys.exit(1)
elif not api_key.startswith("sk-proj-"):
    print("⚠️ API key doesn't start with sk-proj-. Please double-check.")
elif api_key.strip() != api_key:
    print("⚠️ API key has whitespace. Please clean it up.")

# Check for website input
if len(sys.argv) < 2:
    print("❌ No website URL provided.\nUsage: python website_summary.py https://example.com")
    sys.exit(1)

url = sys.argv[1]

# Proper headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

# Website parser class
class Website:
    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string.strip() if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

# Prompt templates
system_prompt = (
    "You are an assistant that analyzes the contents of a website "
    "and provides a short summary, ignoring text that might be navigation related. "
    "Respond with text and provide information neatly in appropriate sections that should be highlighted."
)

def user_prompt_for(website):
    return (
        f"You are looking at a website titled {website.title}.\n"
        "The contents of this website is as follows. Please provide a short summary in markdown. "
        "If it includes news or announcements, summarize these too.\n\n"
        f"{website.text}"
    )

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize_openai(url):
    client = OpenAI(api_key=api_key)
    website = Website(url)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages_for(website)
    )
    return response.choices[0].message.content

def summarize_llama(url):
    website = Website(url)
    client = OpenAI(base_url='http://localhost:11434/v1', api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages_for(website)
    )
    return response.choices[0].message.content

# Run and print summary
summary = None
if model == "openai":
    summary = summarize_openai(url)
elif model == "llama":
    summary = summarize_llama(url)
else:
    print("❌ No model matched, check .env")
    sys.exit(1)

print("\n📝 Website Summary:\n")
print(summary)