from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
import random

app = Flask(__name__, template_folder='../ui')

#fine-tuned model
pipe = pipeline("text-generation", model="fine-tuned-gpt2-recipe")
sentence_model = SentenceTransformer("fine-tuned-minilm-similarity")

# Path to the dataset
dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Clean the dataset to replace missing values
df['name'] = df['name'].fillna("").astype(str)
df['description'] = df['description'].fillna("").astype(str)

def find_similar_recipes(prompt, top_n=5):
    """Finds the top N similar recipes based on the given prompt using fine-tuned MiniLM"""
    prompt_embedding = sentence_model.encode(prompt)
    recipes = []

    for index, row in df.iterrows():
        name = row['name']
        description = row['description']
        combined_text = f"{name} {description}"
        recipe_embedding = sentence_model.encode(combined_text)
        similarity_score = sentence_model.similarity(prompt_embedding, recipe_embedding)

        recipes.append((name, similarity_score))

    recipes = sorted(recipes, key=lambda x: x[1], reverse=True)[:top_n]
    return recipes

@app.route('/generate', methods=['POST'])
def generate_recipe():
    prompt = request.json.get("prompt")
    
    # Generate steps using GPT-2
    step_output = pipe(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
    
    # Find similar recipes
    similar_recipes = find_similar_recipes(prompt)
    
    response = {
        "generated_steps": step_output,
        "similar_recipes": similar_recipes
    }
    return jsonify(response)

@app.route('/random', methods=['GET'])
def random_recipe():
    # Select a random recipe from the dataset
    random_recipe = df.sample(n=1).iloc[0]
    return jsonify({
        'name': random_recipe['name'],
        'description': random_recipe['description'],
        'steps': random_recipe['steps']
    })

if __name__ == "__main__":
    app.run(debug=True)
