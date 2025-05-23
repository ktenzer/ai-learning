import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from IPython.display import Markdown, display, update_display

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
model = os.getenv('MODEL')

# Connect to OpenAI, Anthropic
openai = OpenAI(api_key=openai_api_key)
claude = anthropic.Anthropic(api_key=anthropic_api_key)

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

# GPT 4.0
print("GPT 4.0 Says...")
completion = openai.chat.completions.create(model='gpt-4o-mini', messages=prompts)
print(completion.choices[0].message.content)

print("GPT 4.1 Says...")
# GPT 4.1
completion = openai.chat.completions.create(
    model='gpt-4.1-mini',
    messages=prompts,
    temperature=0.7
)
print(completion.choices[0].message.content)

print("GPT 4.1 Nano Says...")
# GPT Fast Nana
completion = openai.chat.completions.create(
    model='gpt-4.1-nano',
    messages=prompts
)
print(completion.choices[0].message.content)

# Claude
print("Claude Sonnet 3.7 Says...")
message = claude.messages.create(
    model="claude-3-7-sonnet-latest",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ],
)

print(message.content[0].text)

# Claude Streaming
# result = claude.messages.stream(
#     model="claude-3-7-sonnet-latest",
#    max_tokens=200,
#    temperature=0.7,
#    system=system_message,
#    messages=[
#        {"role": "user", "content": user_prompt},
#    ],
#)

#with result as stream:
#    for text in stream.text_stream:
#            print(text, end="", flush=True)