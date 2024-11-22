from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

# Load the GPT-2 model and tokenizer 
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding

dataset_path = "recipe_generation_dataset.txt"
dataset = load_dataset('text', data_files={'train': dataset_path})
dataset['train'] = dataset['train'].shuffle(seed=42).select(range(30000))  # Adjust the range as needed

# Tokenize the dataset with reduced max_length
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=64,  # Lower max length to reduce load time and memory
        return_tensors='pt'
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling (GPT-style)
)

# Define training arguments with optimized settings
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-recipes",
    overwrite_output_dir=True,
    num_train_epochs=1,  # Fewer epochs for faster training
    per_device_train_batch_size=2,  # Reduce batch size
    gradient_accumulation_steps=8,  # Simulate a larger batch size
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    fp16=True  # Enable mixed precision
)

# Set up Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train']
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("fine-tuned-gpt2-recipe")
tokenizer.save_pretrained("fine-tuned-gpt2-recipe")
