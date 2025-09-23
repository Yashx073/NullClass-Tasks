import os
from dotenv import load_dotenv
import gradio as gr
from PIL import Image
import google.generativeai as genai
from utils.embedder import Embedder
from langdetect import detect
from googletrans import Translator

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

translator = Translator()

# Load knowledge base
embedder = Embedder()
vectors, documents = embedder.load_vectors()

def retrieve_context(query, top_k=3):
    # ... same as Task4
    pass

def chatbot_interface(text_input, image_input=None):
    try:
        # Detect user language
        user_lang = detect(text_input)
        
        # Translate to English if needed
        if user_lang != 'en':
            query_en = translator.translate(text_input, src=user_lang, dest='en').text
        else:
            query_en = text_input
        
        # Retrieve context
        context = retrieve_context(query_en)
        prompt = f"{context}\n\nUser: {query_en}\nBot:"
        
        # Image input support
        if image_input:
            if isinstance(image_input, str):
                image_input = Image.open(image_input)
            response = model.generate_content([prompt, image_input])
        else:
            response = model.generate_content(prompt)
        
        bot_response = response.text
        
        # Translate response back to user language
        if user_lang != 'en':
            bot_response = translator.translate(bot_response, src='en', dest=user_lang).text
        
        return bot_response
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI (same as Task4, maybe increase textbox lines)
with gr.Blocks() as demo:
    gr.Markdown("# 🌐 Multilingual Dynamic Chatbot (Gemini 1.5 Flash)")
    with gr.Row():
        text_in = gr.Textbox(label="Enter your query", lines=3, placeholder="Type your question here...")
        image_in = gr.Image(type="filepath", label="Upload an image (optional)")
    out = gr.Textbox(label="Bot Response", lines=12, interactive=False)
    btn = gr.Button("Ask")
    btn.click(chatbot_interface, inputs=[text_in, image_in], outputs=out)

demo.launch()
