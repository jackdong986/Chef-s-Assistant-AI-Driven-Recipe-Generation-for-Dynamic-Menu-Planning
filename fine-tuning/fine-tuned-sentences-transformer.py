from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd
import os

# Load the model
model = SentenceTransformer("paraphrase-MiniLM-L12-v2")

# Load dataset for fine-tuning
similarity_df = pd.read_csv("recipe_similarity_pairs.csv")

# Ensure required columns exist
required_columns = ["text_a", "text_b", "score"]
if not all(col in similarity_df.columns for col in required_columns):
    raise ValueError(f"Missing required columns: {set(required_columns) - set(similarity_df.columns)}")

# Drop rows with missing values in the required columns
similarity_df = similarity_df.dropna(subset=required_columns)

# Adjust sample size
sample_size = min(20000, len(similarity_df))  # Use 20,000 pairs or fewer for quicker training
sampled_df = similarity_df.sample(sample_size, random_state=42)

# Split dataset into training and validation
train_size = int(0.8 * len(sampled_df))  # 80% for training
train_df = sampled_df.iloc[:train_size]
val_df = sampled_df.iloc[train_size:]

# Prepare training examples
train_examples = [
    InputExample(texts=[row['text_a'], row['text_b']], label=min(max(float(row['score']), 0), 1))
    for _, row in train_df.iterrows()
]

# Prepare validation examples
val_examples = [
    InputExample(texts=[row['text_a'], row['text_b']], label=min(max(float(row['score']), 0), 1))
    for _, row in val_df.iterrows()
]

# Create DataLoader for training and validation
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=32)

# Use Cosine Similarity Loss for training
train_loss = losses.CosineSimilarityLoss(model)

# Add evaluator for validation
val_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name="val-eval")

# Fine-tune the model
output_dir = "fine-tuned-minilm-similarity-checkpoints"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    epochs=5,  # Train for 5 epochs
    evaluation_steps=500,  # Evaluate every 500 steps
    warmup_steps=int(len(train_dataloader) * 0.1),  # 10% of training steps for warm-up
    output_path=output_dir,  # Save checkpoints during training
    show_progress_bar=True
)

# Save the final fine-tuned model
final_model_path = "fine-tuned-minilm-similarity"
model.save(final_model_path)
print(f"Model fine-tuned and saved to {final_model_path}")
