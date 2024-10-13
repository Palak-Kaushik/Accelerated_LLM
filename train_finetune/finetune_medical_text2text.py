from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load Medical Text2Text Dataset (e.g., Medical Dialogues or i2b2)
dataset = load_dataset("csv", data_files={"train": "../datasets/medical_text2text.csv"})

# Initialize the T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['input_text'], truncation=True, padding=True)

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

# Fine-tune the Text2Text generation model using Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()

# Save the fine-tuned model
trainer.save_model("../models/fine_tuned_medical_text2text_model")
