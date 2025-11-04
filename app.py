import gradio as gr
from main.translator import translate
import os

def translate_for_gradio(text):
    translation, _ = translate(text)
    return translation

demo = gr.Interface(
    fn=translate_for_gradio, 
    inputs=gr.Textbox(lines=5, label="English Text"), 
    outputs=gr.Textbox(label="Tamil Translation"),
    title="English to Tamil Translation",
    description="Enter an English sentence to translate it into Tamil."
)

if __name__ == "__main__":
    demo.launch()