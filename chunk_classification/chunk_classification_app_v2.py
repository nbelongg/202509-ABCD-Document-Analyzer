import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load the pre-trained models and tokenizers
# download_link_for customModel1 - https://drive.google.com/file/d/1EOH57oWkjkfvC7i8el8K6ZgKm6CAoewu/view?usp=sharing
#old model with less accuracy
model1_path = "./CustomModel"
# better model with high accuracy
# download_link_for customModel2- https://drive.google.com/file/d/1VcsAYunt4BKARRjo3tf9wVE_cpCHTUps/view?usp=sharing
model2_path = "./CustomModel_2"
tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased')
model1 = BertForSequenceClassification.from_pretrained(model1_path)
model2 = BertForSequenceClassification.from_pretrained(model2_path)

# Refactor predict functions for individual models
def predict_relevance(model, sample_data, tokenizer):
    sample_tokenized = tokenizer(sample_data, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**sample_tokenized)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.detach().numpy()
    return predictions

def predict_chunk_with_majority(model, text, tokenizer):
    num_words = len(text.split())
    all_predictions = {"Meaningless": 0, "Meaningful": 0}
    for i in range(0, num_words, 512):
        chunk_words = text.split()[i:i + 512]
        chunk_text = ' '.join(chunk_words)
        tokens = tokenizer.encode(chunk_text, add_special_tokens=True, truncation=True)
        inputs = torch.tensor(tokens).unsqueeze(0)
        outputs = model(inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = predictions.cpu().detach().numpy()
        if predictions[0][0] >= 0.5:
            all_predictions['Meaningless'] += 1
        else:
            all_predictions['Meaningful'] += 1

    majority_label = 1 if all_predictions['Meaningful'] > all_predictions['Meaningless'] else 0
    return majority_label, num_words, all_predictions

def main():
    st.title("Chunk Classification App")
    st.write("Enter your text below to predict its meaningfulness.")

    sample_data = st.text_area("Text")

    # Add a multi-select option for model selection
    selected_models = st.multiselect("Choose models for prediction:", ["Model 1", "Model 2", "Both"])

    if st.button("Predict"):
        if sample_data:
            with st.spinner("Predicting..."):
                predictions1, predictions2 = None, None

                # Perform predictions based on selected models
                if "Model 1" in selected_models or "Both" in selected_models:
                    predictions1 = predict_relevance(model1, sample_data, tokenizer1)

                if "Model 2" in selected_models or "Both" in selected_models:
                    predictions2 = predict_chunk_with_majority(model2, sample_data, tokenizer2)

            # Display results for each selected model
            if predictions1 is not None:
                relevance_label1 = "Meaningful" if predictions1[0][1] > predictions1[0][0] else "Not Meaningful"
                confidence1 = predictions1[0][1] * 100
                if relevance_label1 == "Meaningful":
                    st.success(f"Model 1 Prediction: The text is {relevance_label1} with a confidence of {confidence1:.2f}%.")
                else:
                    st.markdown(f"<font color='red'>Model 1 Prediction: The text is {relevance_label1} with a confidence of {confidence1:.2f}%.</font>", unsafe_allow_html=True)

            if predictions2 is not None:
                relevance_label2 = "Meaningful" if predictions2[0] == 1 else "Not Meaningful"
                with st.expander("Entered Text"):
                    st.write(sample_data)
                st.write(f"Total number of words in the input text: {predictions2[1]}")
                st.write("The all Predictions are as follows", predictions2[2])

                if relevance_label2 == "Meaningful":
                    st.success(f"Model 2 Prediction: The text is {relevance_label2}.")
                else:
                    st.markdown(f"<font color='red'>Model 2 Prediction: The text is {relevance_label2}.</font>", unsafe_allow_html=True)

        else:
            st.warning("Please enter text.")

if __name__ == "__main__":
    main()
