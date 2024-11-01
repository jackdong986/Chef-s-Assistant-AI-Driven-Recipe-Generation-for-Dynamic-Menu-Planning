import pandas as pd
import random

# Load your dataset
dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Part 1a: Prepare the Text Dataset for GPT-2 Fine-Tuning
with open("recipe_generation_dataset.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        prompt = f"Generate a recipe for {row['name']}."
        recipe_text = f"Ingredients: {row['ingredients']}. Steps: {row['steps']}"
        # Separate each prompt-recipe pair with <|endoftext|>
        f.write(f"{prompt}\n{recipe_text}\n<|endoftext|>\n")

# Part 1b: Prepare Similarity Pairs for MiniLM Fine-Tuning
similarity_pairs = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        recipe_a = df.iloc[i]
        recipe_b = df.iloc[j]

        # Calculate a basic similarity score based on overlapping tags
        tags_a = set(str(recipe_a['tags']).split(","))
        tags_b = set(str(recipe_b['tags']).split(","))
        similarity_score = len(tags_a & tags_b) / len(tags_a | tags_b)

        similarity_pairs.append({
            'text_a': recipe_a['name'] + " " + str(recipe_a['description']),
            'text_b': recipe_b['name'] + " " + str(recipe_b['description']),
            'score': similarity_score
        })

# Convert to a DataFrame and save
similarity_df = pd.DataFrame(similarity_pairs)
similarity_df.to_csv("recipe_similarity_pairs.csv", index=False)
