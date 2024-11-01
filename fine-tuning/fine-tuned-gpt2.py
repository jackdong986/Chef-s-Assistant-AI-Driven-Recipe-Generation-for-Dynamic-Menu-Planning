from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the fine-tuning dataset for recipe generation
dataset_path = "recipe_generation_dataset.txt"

# Load the GPT-2 model and tokenizer (openai-community/gpt2-large)
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large")
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2-large")

# Function to load and prepare the dataset
def load_dataset(file_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

# Load dataset
train_dataset = load_dataset(dataset_path)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-large-finetuned-recipes",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
)

# Set up Trainer for GPT-2
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine-tuned-gpt2-large-recipe")
tokenizer.save_pretrained("fine-tuned-gpt2-large-recipe")
