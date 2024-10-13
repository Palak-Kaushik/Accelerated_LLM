import torch
from model.rerouter import classify_query, QueryRerouter, QueryEmbeddingExtractor, DOMAIN_CLASSES
from model.sub_llms import SubLLMs

# Load the trained rerouter model
def load_trained_rerouter(model_path="trained_rerouter.pth"):
    embedding_extractor = QueryEmbeddingExtractor()
    rerouter = QueryRerouter(input_size=768, hidden_size=256, num_classes=4)
    rerouter.load_state_dict(torch.load(model_path))
    rerouter.eval()  # Set the model to evaluation mode
    return rerouter, embedding_extractor

# Load Sub-LLM on classified domain
def process_query_with_sub_llm(query, classified_domain, sub_llms):
    if classified_domain == "question_answering":
        context = "Germany is a country in Europe. The capital of Germany is Berlin."
        return sub_llms.question_answering(query, context)
    elif classified_domain == "translation":
        return sub_llms.translation(query)
    elif classified_domain == "summarization":
        return sub_llms.summarization(query)
    elif classified_domain == "text2text_generation":
        return sub_llms.text2text_generation(query)
    else:
        return "Unknown domain"

# Function to classify a user-inputted query and process it with the corresponding Sub-LLM
def classify_and_process_user_query(trained_rerouter, embedding_extractor, sub_llms):
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ").strip()
        
        if user_query.lower() == "exit":
            print("Exiting the query classification.")
            break

        predicted_class = classify_query(user_query, trained_rerouter, embedding_extractor)
        predicted_domain = DOMAIN_CLASSES[predicted_class]

        result = process_query_with_sub_llm(user_query, predicted_domain, sub_llms)
        
        print(f"Query: {user_query}")
        print(f"Predicted Domain: {predicted_domain}")
        print(f"Result: {result}\n")

if __name__ == "__main__":
    trained_rerouter, embedding_extractor = load_trained_rerouter("trained_rerouter.pth")
    
    sub_llms = SubLLMs()

    classify_and_process_user_query(trained_rerouter, embedding_extractor, sub_llms)
