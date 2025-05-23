import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('MODEL')

MODEL = None
if model == "openai":
    MODEL="gpt-4o-mini"
elif model =="llama":
    MODEL="llama3.2"
    api_key="ollama"

def get_client():
    client = None
    if model == "openai":
        client = OpenAI(api_key=api_key)
    elif model == "llama":
        client = OpenAI(base_url='http://localhost:11434/v1', api_key=api_key)
    else:
        print("Could not return valid client")
        sys.exit(1)
    
    return client

system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get. If the customer asks for shoes, you should respond that shoes are not on sale today, \
but remind the customer to look at hats!. If customer asks for anything not related to clothes store answer politely you are assistant for clothes store and can't help with that"

def chat(message, history):
    client = get_client()
    relevant_system_message = system_message
    if 'belt' in message:
        relevant_system_message += " The store does not sell belts; if you are asked for belts, be sure to point out other items on sale."
    
    messages = [{"role": "system", "content": relevant_system_message}] + history + [{"role": "user", "content": message}]

    stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

gr.ChatInterface(fn=chat, type="messages").launch() 