import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor,AutoModel
import torch
import urllib
import PIL.Image
import io

st.title("Image Generation with OpenGPT 4o")

model3 = AutoModel.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-dpo", trust_remote_code=True)

# Function to generate images
def generate_image(prompt, image_url):
    inputs = processor(text=[prompt], images=[image_url], return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=256,
            eos_token_id=151645,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
    if decoded_text.endswith(""):
        decoded_text = decoded_text[:-10]
    
    return decoded_text

# UI components
prompt = st.text_input("Enter prompt:", "Describe the image you want to generate")
image_url = st.text_input("Enter image URL:", "")

if st.button("Generate Image"):
    if not prompt or not image_url:
        st.error("Please provide both a prompt and an image URL.")
    else:
        try:
            # Download and display the image
            image_data = urllib.request.urlopen(image_url).read()
            image = PIL.Image.open(io.BytesIO(image_data))
            st.image(image, caption='Input Image', use_column_width=True)

            # Generate and display the image description
            generated_text = generate_image(prompt, image_url)
            st.write("Generated Image Description:")
            st.write(generated_text)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
