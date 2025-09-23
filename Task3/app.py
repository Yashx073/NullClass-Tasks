# -----------------------------
# Imports
# -----------------------------
import os
from dotenv import load_dotenv
import gradio as gr
from PIL import Image
import google.generativeai as genai

# -----------------------------
# Load API key from .env
# -----------------------------
load_dotenv()  # Load .env file in project folder
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not set in .env file!")

# -----------------------------
# Configure Gemini 1.5 Flash
# -----------------------------
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Chatbot function
# -----------------------------
def chatbot_interface(text_input, image_input=None):
    try:
        if image_input:
            # If Gradio returns a path string, open as PIL image
            if isinstance(image_input, str):
                image_input = Image.open(image_input)
            response = model.generate_content([text_input, image_input])
        else:
            response = model.generate_content(text_input)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Multi-Modal Chatbot (Gemini 1.5 Flash)")
    
    with gr.Row():
        text_in = gr.Textbox(label="Enter your query")
        image_in = gr.Image(type="filepath", label="Upload an image (optional)")
    
    out = gr.Textbox(label="Bot Response")
    
    btn = gr.Button("Ask")
    btn.click(chatbot_interface, inputs=[text_in, image_in], outputs=out)

# Launch the app
demo.launch()
