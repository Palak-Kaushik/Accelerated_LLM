import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from model.rerouter import QueryRerouter, QueryEmbeddingExtractor

def load_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    queries = df['query'].tolist()
    labels = df['label'].tolist()
    return queries, labels

def train_rerouter(queries, labels, num_epochs=10, learning_rate=0.001):
    embedding_extractor = QueryEmbeddingExtractor()
    rerouter = QueryRerouter(input_size=768, hidden_size=256, num_classes=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rerouter.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for i, query in enumerate(queries):
            embeddings = embedding_extractor.get_embeddings(query)
            
            optimizer.zero_grad()
            
            outputs = rerouter(embeddings)
            label = torch.tensor([labels[i]])
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(queries):.4f}')

    return rerouter

if __name__ == "__main__":
    queries, labels = load_data_from_csv("query_training_data.csv")
    trained_rerouter = train_rerouter(queries, labels, num_epochs=15, learning_rate=0.001)
    
    torch.save(trained_rerouter.state_dict(), "trained_rerouter.pth")
    print("Model has been trained and saved as 'trained_rerouter.pth'.")
