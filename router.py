from datetime import datetime
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5') 
model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')


with open('rf_classifier.bin', 'rb') as f:
    rf_model = pickle.load(f)

with open('label_encoder.bin', 'rb') as f:
    encoder = pickle.load(f)



def classify_query(query):
    start_time = datetime.now()  # to calculate time taken in classification
    # embedding the query
    encoded_query = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_query)
        test_embeddings = model_output[0][:, 0]
    test_embeddings = torch.nn.functional.normalize(test_embeddings, p=2, dim=1)
    pred=rf_model.predict(test_embeddings)
    class_pred=encoder.inverse_transform(pred)
    end_time = datetime.now()
    return class_pred[0], (end_time - start_time)



################ USER INTERFACE ################



import gradio as gr


# Create the Gradio interface
iface = gr.Interface(
    fn=classify_query,                   # Function to process the input
    inputs=gr.Textbox(label="Input Query"),  
    outputs=[                           
        gr.Textbox(label="Query Class"),
        gr.Textbox(label="Time Taken to classify query")
    ],
    title="Accelerated LLM",
    description="Classify Queries."
)


if __name__ == "__main__":
    iface.launch()
