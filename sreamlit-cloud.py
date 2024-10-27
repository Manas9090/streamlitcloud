import streamlit as st
from transformers import pipeline

# Load the model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2") 

model = load_model() 

# Streamlit app title
st.title("LLM Text Generation with Streamlit") 

# User input
user_input = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if user_input:
        # Generate text using the LLM
        with st.spinner("Generating..."):
            output = model(user_input, max_length=50, num_return_sequences=1)
            st.subheader("Generated Text:")
            st.write(output[0]['generated_text'])
    else:
        st.error("Please enter a prompt to generate text.")
