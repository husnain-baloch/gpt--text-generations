import streamlit as st
from transformers import pipeline

st.title("GPT-2 Text Generation")
st.write("Decoder-only model: continues the beginning of a sentence by predicting the next words.")

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="lvwerra/gpt2-imdb")

gen = load_model()

text = st.text_input("Start of sentence", "The movie was")
length = st.slider("Number of new words (tokens)", 10, 80, 40)

if st.button("Run GPT-2"):
    out = gen(text, max_new_tokens=length, num_return_sequences=1, do_sample=True)
    st.write(out[0]["generated_text"])