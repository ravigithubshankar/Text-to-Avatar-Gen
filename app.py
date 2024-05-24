import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Set up the Streamlit app layout
st.title("Image Generation with Stable Diffusion")
st.write("Enter a text prompt and generate an image using Stable Diffusion")

# Input text prompt from the user
prompt = st.text_input("Text Prompt", "A futuristic cityscape with towering skyscrapers and flying cars")

# Button to generate the image
if st.button("Generate Image"):
    # Load the Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    # Generate the image
    with torch.no_grad():
        image = pipe(prompt, guidance_scale=7.5).images[0]

    # Convert the image to a format Streamlit can display
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = buffered.getvalue()

    # Display the image
    st.image(img_str, caption="Generated Image", use_column_width=True)

# Display footer
st.write("Powered by Streamlit and Stable Diffusion")
