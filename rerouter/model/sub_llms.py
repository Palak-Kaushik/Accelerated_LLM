from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

# Sub-LLMs for different tasks
class SubLLMs:
    def __init__(self):
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.qa_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        self.translation_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        
        self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        self.summarization_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        self.text2text_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.text2text_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    def question_answering(self, query, context):
        qa_pipeline = pipeline("question-answering", model=self.qa_model, tokenizer=self.qa_tokenizer)
        result = qa_pipeline(question=query, context=context)
        return result['answer']
    
    def translation(self, query):
        inputs = self.translation_tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.translation_model.generate(inputs["input_ids"], max_length=512, num_beams=5, early_stopping=True)
        return self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def summarization(self, query):
        inputs = self.summarization_tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.summarization_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def text2text_generation(self, query):
        inputs = self.text2text_tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.text2text_model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
        return self.text2text_tokenizer.decode(outputs[0], skip_special_tokens=True)
