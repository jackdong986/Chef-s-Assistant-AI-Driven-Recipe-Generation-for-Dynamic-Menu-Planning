import pandas as pd
import random
import re
from tqdm import tqdm
from multiprocessing import Pool
import os

tqdm.pandas()

def preprocess_and_save_data():
    # Define the number of pairs to generate
    num_pairs = 10000  # Adjust this as needed

    # Load only necessary columns and reduce memory usage
    columns_needed = ['name', 'description', 'tags', 'ingredients', 'steps']
    dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'

    print("Loading dataset...")
    df = pd.read_csv(dataset_path, encoding='ISO-8859-1', usecols=columns_needed)
    print("Dataset loaded successfully!")

    # Data Preprocessing and Cleaning
    print("Preprocessing data...")
    df['name'] = df['name'].fillna("").astype(str).str.lower().str.strip()
    df['description'] = df['description'].fillna("").astype(str).str.lower().str.strip()
    df['tags'] = df['tags'].fillna("").astype(str).str.lower().str.strip()
    df['ingredients'] = df['ingredients'].fillna("").astype(str).str.lower().str.strip()
    df['steps'] = df['steps'].fillna("").astype(str).str.lower().str.strip()

    df['description'] = df['description'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    def clean_text(text):
        text = text.replace("[", "").replace("]", "").replace('"', "").replace("'", "").strip()
        return text

    for column in ['name', 'description', 'tags', 'ingredients', 'steps']:
        print(f"Cleaning column: {column}")
        df[column] = df[column].progress_apply(clean_text)

    print("Removing duplicates...")
    df = df.drop_duplicates(subset=['name', 'description'])

    print("Filtering out recipes with too many steps...")
    df = df[df['steps'].progress_apply(lambda x: len(x.split('.')) <= 20)]

    print("Filtering out recipes with too many ingredients...")
    df = df[df['ingredients'].progress_apply(lambda x: len(x.split(',')) <= 15)]

    df['ingredients'] = df['ingredients'].progress_apply(lambda x: ', '.join(x.split(',')))
    df['steps'] = df['steps'].progress_apply(lambda x: '. '.join(x.split('.')))

    return df, num_pairs

def compute_similarity(row1, row2):
    row1_name, row2_name = set(row1['name'].split()), set(row2['name'].split())
    row1_desc, row2_desc = set(row1['description'].split()), set(row2['description'].split())
    row1_tags, row2_tags = set(row1['tags']), set(row2['tags'])
    row1_ing, row2_ing = set(row1['ingredients'].split(',')), set(row2['ingredients'].split(','))

    name_sim = len(row1_name & row2_name)
    desc_sim = len(row1_desc & row2_desc)
    tags_sim = len(row1_tags & row2_tags)
    ingredients_sim = len(row1_ing & row2_ing)

    weight_name, weight_description, weight_tags, weight_ingredients = 3, 2, 2, 1
    score = weight_name * name_sim + weight_description * desc_sim + weight_tags * tags_sim + weight_ingredients * ingredients_sim
    return score / (weight_name + weight_description + weight_tags + weight_ingredients)

def generate_pairs_for_chunk(args):
    chunk_a, chunk_b = args
    pairs = []
    for _, recipe_a in chunk_a.iterrows():
        for _, recipe_b in chunk_b.iterrows():
            if recipe_a.name >= recipe_b.name:
                continue
            similarity_score = compute_similarity(recipe_a, recipe_b)
            if similarity_score > 0:
                text_a = f"{recipe_a['name']} {recipe_a['description']} Ingredients: {recipe_a['ingredients']}"
                text_b = f"{recipe_b['name']} {recipe_b['description']} Ingredients: {recipe_b['ingredients']}"
                pairs.append({'text_a': text_a, 'text_b': text_b, 'score': similarity_score})
    return pairs

def save_for_gpt2_format(df, output_file):
    """Save recipes to a .txt file for GPT-2 text generation."""
    print(f"Saving recipes for GPT-2 text generation to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(f"Recipe Name: {row['name']}\n")
            f.write(f"Description: {row['description']}\n")
            f.write(f"Ingredients: {row['ingredients']}\n")
            f.write(f"Steps: {row['steps']}\n\n")
    print("Recipes saved successfully!")

if __name__ == "__main__":
    # Preprocess data
    df, num_pairs = preprocess_and_save_data()

    # Save data for GPT-2 text generation
    gpt2_file = "recipe_generation_dataset.txt"
    save_for_gpt2_format(df, gpt2_file)

    print("Generating similarity pairs...")
    sampled_df = df.sample(min(len(df), num_pairs * 2), random_state=42)

    # Split data into chunks
    batch_size = 1000
    chunks = [sampled_df[i:i + batch_size] for i in range(0, len(sampled_df), batch_size)]
    chunk_pairs = [(chunks[i], chunks[j]) for i in range(len(chunks)) for j in range(i, len(chunks))]

    # Use multiprocessing
    output_file = "recipe_similarity_pairs.csv"

    if not os.path.exists(output_file):
        pd.DataFrame(columns=['text_a', 'text_b', 'score']).to_csv(output_file, index=False)

    print("Processing pairs with multiple cores...")
    with Pool(processes=4) as pool:  # Utilize 4 cores
        for i, result in enumerate(tqdm(pool.imap(generate_pairs_for_chunk, chunk_pairs), total=len(chunk_pairs))):
            # Save the results every 10% of processing
            pd.DataFrame(result).to_csv(output_file, mode='a', index=False, header=False)
            print(f"Saved {i+1}/{len(chunk_pairs)} chunks to {output_file}")
