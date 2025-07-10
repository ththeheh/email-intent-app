import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st

@st.cache_resource
def load_model():
    # 🔥 Correct and OS-safe absolute path (DO NOT pass it as a Hugging Face repo id)
    model_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "models", "snips-intent-classifier"))

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found at: {model_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        local_files_only=True
    )
    return model, tokenizer

model, tokenizer = load_model()

st.title("📧 Email Intent Classifier")
st.write("Paste your email text below to detect the intent:")

user_input = st.text_area("Email Text", height=150)

if st.button("Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = model(**inputs)
            pred = torch.argmax(output.logits, dim=1).item()
            label = model.config.id2label[pred]
            st.success(f"Predicted Intent: **{label}**")
