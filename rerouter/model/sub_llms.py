from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Sub-LLMs for healthcare-specific tasks using fine-tuned models
class HealthcareSubLLMs:
    def __init__(self):
        # Load the fine-tuned models and tokenizers for each domain
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("fine_tuned_medical_qa_model")
        self.qa_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_medical_qa_model")
        
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_medical_translation_model")
        self.translation_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_medical_translation_model")
        
        self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_medical_summarization_model")
        self.summarization_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_medical_summarization_model")
        
        self.text2text_model = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_medical_text2text_model")
        self.text2text_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_medical_text2text_model")
    
    def medical_question_answering(self, query, context):
        qa_pipeline = pipeline("question-answering", model=self.qa_model, tokenizer=self.qa_tokenizer)
        result = qa_pipeline(question=query, context=context)
        return result['answer']
    
    def medical_translation(self, query):
        inputs = self.translation_tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.translation_model.generate(inputs["input_ids"], max_length=512, num_beams=5, early_stopping=True)
        return self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def medical_summarization(self, query):
        inputs = self.summarization_tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.summarization_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def medical_text2text_generation(self, query):
        inputs = self.text2text_tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.text2text_model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
        return self.text2text_tokenizer.decode(outputs[0], skip_special_tokens=True)
