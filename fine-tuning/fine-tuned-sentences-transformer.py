from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

similarity_df = pd.read_csv("recipe_similarity_pairs.csv")

# Sample a subset of the data for quicker training
sample_size = min(30000, len(similarity_df))  # Adjust sample size as needed
sampled_df = similarity_df.sample(sample_size, random_state=42)

train_examples = [
    InputExample(texts=[row['text_a'], row['text_b']], label=float(row['score']))
    for _, row in sampled_df.iterrows()
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    show_progress_bar=True
)

model.save("fine-tuned-minilm-similarity")
