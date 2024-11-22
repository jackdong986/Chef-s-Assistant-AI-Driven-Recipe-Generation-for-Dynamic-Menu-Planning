from flask import Flask, render_template, request, jsonify, redirect, url_for
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
import random

app = Flask(__name__, template_folder='../ui')

# Fine-tuned models
pipe = pipeline("text-generation", model="fine-tuned-gpt2-recipe")
sentence_model = SentenceTransformer("fine-tuned-minilm-similarity")

# Load and preprocess dataset
dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
df['name'] = df['name'].fillna("").astype(str).str.lower()
df['description'] = df['description'].fillna("").astype(str).str.lower()
df['steps'] = df['steps'].fillna("").astype(str).str.lower()
df['ingredients'] = df['ingredients'].fillna("").astype(str).str.lower()

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

@app.route('/generate', methods=['POST'])
def generate_recipe():
    prompt = request.json.get("prompt")
    # Generate steps for the recipe
    steps = pipe(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
    # Find similar recipes
    similar_recipes = find_similar_recipes(prompt)
    return jsonify({"generated_steps": steps, "similar_recipes": similar_recipes})

@app.route('/random', methods=['GET'])
def random_recipe():
    """Return 10 random recipes."""
    random_recipes = df.sample(n=10)
    recipes = random_recipes[['name', 'ingredients', 'steps']].to_dict(orient='records')
    return jsonify(recipes)

@app.route('/recipe_details', methods=['GET'])
def recipe_details():
    """Return the details of a specific recipe."""
    index = int(request.args.get("index"))
    recipe = df.iloc[index]
    return jsonify({
        "name": recipe['name'].title(),
        "ingredients": recipe['ingredients'].split(", "),
        "steps": recipe['steps'].split(". ")[:10]  # Ensure min 5, max 10 steps
    })

if __name__ == "__main__":
    app.run(debug=True)
