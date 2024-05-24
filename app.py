import streamlit as st
import requests
import io
from PIL import Image

def query_stabilitydiff(payload, headers):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

with st.sidebar:
    st.title("💬 Chatbot - Text to Image Gen")
    st.caption("🚀 A Streamlit chatbot powered by Strom")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "What kind of image that I need to draw? (example: running cat)"}]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
    if "image" in message:
        st.chat_message("assistant").image(message["image"], caption=message["prompt"], use_column_width=True)

if prompt := st.chat_input():

    if not isinstance(st.secrets.hugging_face_token, dict):
        st.info("Your Hugging Face Token configuration seems incorrect. Please check your secrets setup.")
        st.stop()
    elif "api_key" not in st.secrets.hugging_face_token:
        st.info("Please add your Hugging Face Token API key to continue.")
        st.stop()


    # Input prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Query Stable Diffusion
    headers = {"Authorization": f"Bearer {st.secrets.hugging_face_token.api_key}"}
    image_bytes = query_stabilitydiff({
        "inputs": prompt,
    }, headers)

    # Return Image
    image = Image.open(io.BytesIO(image_bytes))
    msg = f'here is your image related to "{prompt}"'

    # Show Result
    st.session_state.messages.append({"role": "assistant", "content": msg, "prompt": prompt, "image": image})
    st.chat_message("assistant").write(msg)
    st.chat_message("assistant").image(image, caption=prompt, use_column_width=True)




