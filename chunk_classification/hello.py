import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load the pre-trained model and tokenizer
model_path = "./CustomModel"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_path)

def predict_chunk_with_majority(text):
    # Calculate the number of words in the input text
    num_words = len(text.split())

    # Tokenize the input text in chunks of 512 words
    all_predictions = {"Meaningless":0,"Meaningful":0}
    for i in range(0, num_words, 512):
        chunk_words = text.split()[i:i + 512]
        chunk_text = ' '.join(chunk_words)
        tokens = tokenizer.encode(chunk_text, add_special_tokens=True, truncation=True)
        inputs = torch.tensor(tokens).unsqueeze(0)

        # Make predictions for the chunk
        outputs = model(inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions=predictions.cpu().detach().numpy()
        if predictions[0][0]>=0.5:
            all_predictions['Meaningless']+=1
        else:
            all_predictions['Meaningful']+=1

    # Calculate the majority prediction
        if all_predictions['Meaningful']>all_predictions['Meaningless']:
            majority_label=1
        else:
            majority_label=0
    return majority_label, num_words,all_predictions

def main():
    st.title("Chunk Classification App")
    st.write("Enter your chunk text below to predict whether it's meaningful or not.")

    sample_data = st.text_area("Chunk Text")

    if st.button("Predict"):
        if sample_data:
            with st.spinner("Predicting..."):
                majority_label, num_words,all_predictions = predict_chunk_with_majority(sample_data)
        
            relevance_label = "Meaningful" if majority_label == 1 else "Not Meaningful"
            with st.expander("Entered Chunk Text"):
                st.write(sample_data)
            st.write(f"Total number of words in the input text: {num_words}")

            st.write(" The all Prediction are as follows",all_predictions)

            if relevance_label == "Meaningful":
                st.success(f"The sample data is {relevance_label} ")
            else:
                st.markdown(f"<font color='red'>The sample data is {relevance_label} </font>", unsafe_allow_html=True)
        else:
            st.warning("Please enter sample data.")

if __name__ == "__main__":
    main()

