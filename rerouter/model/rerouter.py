import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class QueryRerouter(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_classes=3):
        super(QueryRerouter, self).__init__()
        # feedforward neural network 2 layers 
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return self.softmax(out)

# pre-trained BERT embeddings
class QueryEmbeddingExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def get_embeddings(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", max_length=128, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

def classify_query(query, rerouter, embedding_extractor):
    embeddings = embedding_extractor.get_embeddings(query)
    output = rerouter(embeddings)
    _, predicted_class = torch.max(output.data, 1)
    return predicted_class.item()

# domain classes: 0 - Social Science, 1 - STEM, 2 - Coding
DOMAIN_CLASSES = {0: 'social_science', 1: 'stem', 2: 'coding'}
