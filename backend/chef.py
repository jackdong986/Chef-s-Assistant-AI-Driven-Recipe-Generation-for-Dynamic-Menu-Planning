from flask import Flask, render_template, request, jsonify, redirect, url_for
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
import random
import re

app = Flask(__name__, template_folder='../ui')

pipe = pipeline("text-generation", model="fine-tuned-gpt2-recipe")
sentence_model = SentenceTransformer("fine-tuned-minilm-similarity")

dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Cleaning Function
def clean_text(text):
    """Remove unwanted characters and ensure uniform formatting."""
    text = text.replace("[", "").replace("]", "")  
    text = text.replace('"', "").replace("'", "")  
    text = text.strip()  
    return text

df['name'] = df['name'].fillna("").astype(str).str.lower().apply(clean_text)
df['description'] = df['description'].fillna("").astype(str).str.lower().apply(clean_text)
df['tags'] = df['tags'].fillna("").astype(str).str.lower().apply(lambda x: [clean_text(tag) for tag in x.strip('[]').split(', ') if tag])
df['ingredients'] = df['ingredients'].fillna("").astype(str).str.lower().apply(clean_text)
df['steps'] = df['steps'].fillna("").astype(str).str.lower().apply(clean_text)

df = df[df['steps'].apply(lambda x: len(x.split('.')) <= 20)]
df = df[df['ingredients'].apply(lambda x: len(x.split(',')) <= 15)]

df['ingredients'] = df['ingredients'].apply(lambda x: ', '.join(x.split(',')))
df['steps'] = df['steps'].apply(lambda x: '. '.join(x.split('.')))

def find_similar_recipes(prompt, top_n=10):
    """Find top N recipes similar to the prompt."""
    prompt_embedding = sentence_model.encode(prompt)
    recipes = []

    for _, row in df.iterrows():
        combined_text = f"{row['name']} {row['description']}"
        recipe_embedding = sentence_model.encode(combined_text)
        similarity_score = sentence_model.similarity(prompt_embedding, recipe_embedding)
        recipes.append((row['name'], similarity_score))

    recipes = sorted(recipes, key=lambda x: x[1], reverse=True)[:top_n]
    return [{"name": r[0]} for r in recipes]

@app.route('/')
def home():
    """Render the main page with the search and random recipe buttons."""
    return render_template('chefRecipeGenerator.html')

@app.route('/search', methods=['POST'])
def search_recipe():
    """Search for recipes by name in the dataset."""
    prompt = request.json.get("prompt", "").lower().strip()

    matching_recipes = df[df['name'].str.contains(prompt, na=False, case=False)]

    recipes = matching_recipes.head(10)[['name', 'ingredients', 'steps', 'description']].to_dict(orient='records')

    return jsonify({"recipes": recipes})


@app.route('/random', methods=['GET'])
def random_recipe():
    """Return 10 random recipes."""
    random_recipes = df.sample(n=10)[['name', 'ingredients', 'steps', 'description']].to_dict(orient='records')
    return jsonify(random_recipes)

@app.route('/recipe_details', methods=['GET'])
def recipe_details():
    """Return details of a specific recipe by name."""
    name = request.args.get("name", "").lower()
    recipe = df[df['name'] == name].iloc[0]
    ingredients = recipe['ingredients'].split(', ')
    steps = recipe['steps'].split('. ')
    
    ingredients = [ingredient.strip() for ingredient in ingredients if ingredient.strip()]
    steps = [step.strip() for step in steps if step.strip()]

    steps = [f"{i+1}. {step}" for i, step in enumerate(steps)]

    return jsonify({
        "name": recipe['name'].title(),
        "ingredients": ingredients,
        "steps": steps,
        "description": recipe['description']
    })

if __name__ == "__main__":
    app.run(debug=True)
