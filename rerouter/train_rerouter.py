import torch
import torch.optim as optim
import torch.nn as nn
from model.rerouter import QueryRerouter, QueryEmbeddingExtractor

# Example queries and their corresponding labels
queries = [
    "Who was the president of the US during the Civil War?", 
    "What is the integral of x^2?", 
    "Write a Python program to reverse a string."
]
labels = [0, 1, 2]  # Labels: 0 - Social Science, 1 - STEM, 2 - Coding

def train_rerouter(queries, labels, num_epochs=10, learning_rate=0.001):
    # Initialize the embedding extractor and the rerouter model
    embedding_extractor = QueryEmbeddingExtractor()
    rerouter = QueryRerouter(input_size=768, hidden_size=128, num_classes=3)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(rerouter.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for i, query in enumerate(queries):
            # Get embeddings for each query
            embeddings = embedding_extractor.get_embeddings(query)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass through rerouter
            outputs = rerouter(embeddings)
            label = torch.tensor([labels[i]])
            
            # Compute the loss
            loss = criterion(outputs, label)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(queries):.4f}')

    return rerouter

# Train the rerouter
if __name__ == "__main__":
    train_rerouter = train_rerouter(queries, labels, num_epochs=10, learning_rate=0.001)
