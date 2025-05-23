import os
import sys
import json
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

client = get_client()

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499", "pome beach": "$169"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = client.chat.completions.create(model=MODEL, messages=messages)
    
    return response.choices[0].message.content

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city

gr.ChatInterface(fn=chat, type="messages").launch()