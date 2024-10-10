import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class QueryRerouter(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, num_classes=4):
        super(QueryRerouter, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden_size // 4, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.layer3(out)
        out = self.relu3(out)
        out = self.layer4(out)
        return self.softmax(out)

# BERT embeddings
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

DOMAIN_CLASSES = {0: 'question_answering', 1: 'translation', 2: 'summarization', 3: 'text2text_generation'}
