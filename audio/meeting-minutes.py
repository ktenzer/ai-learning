import os
import platform
import torch
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load API keys
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Authenticate
login(token=hf_token)
openai = OpenAI(api_key=openai_api_key)

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device.type == "mps" else torch.float32

# Load LLaMA model + tokenizer once
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map=None,
    torch_dtype=dtype
).to(device)

def process_meeting(audio_file):
    # Transcribe using OpenAI Whisper
    with open(audio_file, "rb") as f:
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )

    # Prompt setup
    system_message = (
        "You are an assistant that produces minutes of meetings from transcripts, "
        "including summary, attendees, location, date, discussion points, and action items."
    )
    user_prompt = (
        f"Below is an extract transcript of a council meeting. Please write minutes in markdown, "
        f"including a summary with attendees, location and date; discussion points; takeaways; "
        f"and action items with owners.\n{transcription}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    # Tokenize
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Stream token-by-token and yield text to Gradio
    generated_text = ""
    past_key_values = None
    stop_token_id = tokenizer.eos_token_id

    model.eval()
    with torch.no_grad():
        for _ in range(2000):
            if past_key_values is None:
                outputs = model(
                    input_ids=inputs,
                    use_cache=True
                )
            else:
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if next_token.item() == stop_token_id:
                break

            decoded = tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated_text += decoded
            yield generated_text

# Gradio Interface with streaming output
iface = gr.Interface(
    fn=process_meeting,
    inputs=gr.Audio(type="filepath", label="Upload Meeting Audio"),
    outputs=gr.Textbox(label="Meeting Minutes (Markdown)"),
    title="Meeting Minutes Generator",
    description="Upload an audio recording of a meeting to transcribe and summarize using Whisper + LLaMA.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()