import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = ["Non-Hate", "Offensive", "Hate"]

# Load model and tokenizer
@st.cache_resource  # Cache to load model only once
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained("odd_layer_best").to(device)
    tokenizer = BertTokenizer.from_pretrained("odd_layer_best")
    model.eval()
    return model, tokenizer

# Classification function
def classify_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = logits.argmax(-1).item()
    return labels[pred]

# Streamlit app
def main():
    st.title("Hate Speech Detector")
    st.write("Enter text below to classify it as Non-Hate, Offensive, or Hate.")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Text input
    user_input = st.text_area("Your Text", placeholder="Type here...")

    if st.button("Classify"):
        if user_input:
            result = classify_text(model, tokenizer, user_input)
            st.success(f"Prediction: **{result}**")
            st.write(f"Input: \"{user_input}\"")
        else:
            st.warning("Please enter some text to classify.")

    # Optional: Display a pre-saved performance plot (if available)
    try:
        st.subheader("Model Performance")
        st.image("performance_plot.png", caption="Training Loss and Eval Accuracy")
    except FileNotFoundError:
        st.write("Performance plot not available.")

if __name__ == "__main__":
    main()
