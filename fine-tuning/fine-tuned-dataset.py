import pandas as pd
import random
import re

# Define the number of pairs to generate
num_pairs = 100000  # Adjust this as needed

# Load only necessary columns and reduce memory usage
columns_needed = ['name', 'description', 'tags', 'ingredients', 'steps']
dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'
df = pd.read_csv(dataset_path, encoding='ISO-8859-1', usecols=columns_needed)

# Data Preprocessing and Cleaning
# Fill NaNs with empty strings and ensure all columns are strings
df['name'] = df['name'].fillna("").astype(str).str.lower().str.strip()
df['description'] = df['description'].fillna("").astype(str).str.lower().str.strip()
df['tags'] = df['tags'].fillna("").astype(str).str.lower().str.strip()
df['ingredients'] = df['ingredients'].fillna("").astype(str).str.lower().str.strip()
df['steps'] = df['steps'].fillna("").astype(str).str.lower().str.strip()

# Remove special characters from the 'description'
df['description'] = df['description'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Convert 'tags' into a list and remove extra formatting
df['tags'] = df['tags'].apply(lambda x: x.strip('[]').replace("'", "").split(', ') if pd.notna(x) else [])

# Remove duplicate recipes based on 'name' and 'description'
df = df.drop_duplicates(subset=['name', 'description'])

# Filter out recipes with too many steps or ingredients (outlier removal)
df = df[df['steps'].apply(lambda x: len(x.split('.')) <= 20)]
df = df[df['ingredients'].apply(lambda x: len(x.split(',')) <= 15)]

# Reformat 'ingredients' and 'steps' for consistency
df['ingredients'] = df['ingredients'].apply(lambda x: ', '.join(x.split(',')))
df['steps'] = df['steps'].apply(lambda x: '. '.join(x.split('.')))

# Prepare the dataset for text generation
with open("recipe_generation_dataset.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        prompt = f"Generate a recipe for {row['name']}."
        recipe_text = f"Ingredients: {row['ingredients']}. Steps: {row['steps']}"
        f.write(f"{prompt}\n{recipe_text}\n<|endoftext|>\n")

# Sample data without replacement to create pairs for similarity calculation
sampled_indices = random.sample(range(len(df)), num_pairs * 2)  # Oversample to create more unique pairs
sampled_df = df.iloc[sampled_indices]

# Generate unique pairs only and compute similarity
similarity_pairs = []

def compute_similarity(row1, row2):
    tags_a = set(row1['tags'])
    tags_b = set(row2['tags'])
    common_tags = tags_a & tags_b
    if common_tags:
        return len(common_tags) / len(tags_a | tags_b)
    return 0

for i in range(0, len(sampled_df), 2):
    if i + 1 >= len(sampled_df):
        break
    
    recipe_a, recipe_b = sampled_df.iloc[i], sampled_df.iloc[i + 1]
    similarity_score = compute_similarity(recipe_a, recipe_b)
    
    if similarity_score > 0:  # Only include pairs with some similarity
        text_a = recipe_a['name'] + " " + recipe_a['description']
        text_b = recipe_b['name'] + " " + recipe_b['description']
        similarity_pairs.append({'text_a': text_a, 'text_b': text_b, 'score': similarity_score})

# Save the similarity pairs to a CSV file
similarity_df = pd.DataFrame(similarity_pairs)
similarity_df.to_csv("recipe_similarity_pairs.csv", index=False)
