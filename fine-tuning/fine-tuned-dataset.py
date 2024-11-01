import pandas as pd
import random

# Define the number of pairs to generate
num_pairs = 10000  # Adjust this as needed

# Load only necessary columns and reduce memory usage
columns_needed = ['name', 'description', 'tags', 'ingredients', 'steps']
dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'
df = pd.read_csv(dataset_path, encoding='ISO-8859-1', usecols=columns_needed)

# Ensure string types and handle NaNs
df['name'] = df['name'].fillna("").astype(str)
df['description'] = df['description'].fillna("").astype(str)
df['tags'] = df['tags'].fillna("").astype(str)

# Part 1a: Prepare the Text Dataset for GPT-2 Fine-Tuning
with open("recipe_generation_dataset.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        prompt = f"Generate a recipe for {row['name']}."
        recipe_text = f"Ingredients: {row['ingredients']}. Steps: {row['steps']}"
        f.write(f"{prompt}\n{recipe_text}\n<|endoftext|>\n")

# Part 1b: Prepare Similarity Pairs for MiniLM Fine-Tuning
# Sample without replacement to prevent duplicates and self-pairing
sampled_indices = random.sample(range(len(df)), num_pairs * 2)  # Oversample for more unique pairs
sampled_df = df.iloc[sampled_indices]

# Generate unique pairs only and compute similarity
similarity_pairs = []

def compute_similarity(row1, row2):
    tags_a = set(row1['tags'].split(","))
    tags_b = set(row2['tags'].split(","))
    common_tags = tags_a & tags_b
    if common_tags:  # If tags are common, compute similarity
        return len(common_tags) / len(tags_a | tags_b)
    return 0

# Generate pairs
for i in range(0, len(sampled_df), 2):
    if i + 1 >= len(sampled_df):
        break

    recipe_a, recipe_b = sampled_df.iloc[i], sampled_df.iloc[i + 1]
    similarity_score = compute_similarity(recipe_a, recipe_b)
    
    if similarity_score > 0:  # Only include pairs with similarity
        text_a = recipe_a['name'] + " " + recipe_a['description']
        text_b = recipe_b['name'] + " " + recipe_b['description']
        similarity_pairs.append({'text_a': text_a, 'text_b': text_b, 'score': similarity_score})

# Save similarity pairs
similarity_df = pd.DataFrame(similarity_pairs)
similarity_df.to_csv("recipe_similarity_pairs.csv", index=False)
