import torch
from model.rerouter import classify_query, QueryRerouter, QueryEmbeddingExtractor, DOMAIN_CLASSES

# Load the trained model
def load_trained_rerouter(model_path="trained_rerouter.pth"):
    embedding_extractor = QueryEmbeddingExtractor()
    rerouter = QueryRerouter(input_size=768, hidden_size=256, num_classes=4)
    rerouter.load_state_dict(torch.load(model_path))
    rerouter.eval()  
    return rerouter, embedding_extractor

def classify_user_query(trained_rerouter, embedding_extractor):
    while True:
        # Get user input
        user_query = input("Enter your query (or type 'exit' to quit): ").strip()
        
        if user_query.lower() == "exit":
            print("Exiting the query classification.")
            break

        predicted_class = classify_query(user_query, trained_rerouter, embedding_extractor)
        predicted_domain = DOMAIN_CLASSES[predicted_class]

        print(f"Query: {user_query}")
        print(f"Predicted Domain: {predicted_domain}\n")

if __name__ == "__main__":
    trained_rerouter, embedding_extractor = load_trained_rerouter("trained_rerouter.pth")
    classify_user_query(trained_rerouter, embedding_extractor)
