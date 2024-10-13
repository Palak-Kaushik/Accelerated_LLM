from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load Medical QA Dataset
dataset = load_dataset("csv", data_files={"train": "../datasets/medical_qa.csv"})

# Initialize the BERT model and tokenizer
model_name = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['question'], examples['context'], truncation=True, padding=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Fine-tune the QA model using Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()
trainer.save_model("../models/fine_tuned_medical_qa_model")
