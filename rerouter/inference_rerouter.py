from model.rerouter import classify_query, QueryRerouter, QueryEmbeddingExtractor, DOMAIN_CLASSES
from train_rerouter import train_rerouter

# Example queries and their corresponding labels
queries = [
    "Who was the president of the US during the Civil War?", 
    "What is the integral of x^2?", 
    "Write a Python program to reverse a string."
]
labels = [0, 1, 2]  # Labels: 0 - Social Science, 1 - STEM, 2 - Coding

# Train the rerouter model first
trained_rerouter = train_rerouter(queries, labels, num_epochs=10, learning_rate=0.001)

# Test the rerouter after training
def test_rerouter(trained_rerouter):
    embedding_extractor = QueryEmbeddingExtractor()

    # Example queries for testing
    test_queries = [
        "Who was the president of the US during the Civil War?",  # Expected: Social Science
        "What is the integral of x^2?",  # Expected: STEM
        "Write a Python program to reverse a string.",  # Expected: Coding
        "What is the capital of India?",
        "How is India geography mad ethe british rule?",
        "How many years did British & East India Company ruled over India?",
        "How do i Connect the backend to the frontend?"
    ]
    
    for query in test_queries:
        predicted_class = classify_query(query, trained_rerouter, embedding_extractor)
        print(f"Query: {query}")
        print(f"Predicted Domain: {DOMAIN_CLASSES[predicted_class]}\n")

# Test the model
if __name__ == "__main__":
    test_rerouter(trained_rerouter)
