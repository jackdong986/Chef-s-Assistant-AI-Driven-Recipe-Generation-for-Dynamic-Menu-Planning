from flask import Flask, render_template, request, jsonify
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
    """Find top N recipes by name or description containing the prompt."""
    filtered_recipes = df[
        df['name'].str.contains(prompt, case=False) | df['description'].str.contains(prompt, case=False)
    ][['name', 'description', 'amount']].head(top_n)
    return filtered_recipes.to_dict(orient='records')

@app.route('/')
def home():
    """Render the main page."""
    return render_template('chefRecipeGenerator.html')

@app.route('/search', methods=['POST'])
def search_recipe():
    """Search for existing recipes by prompt."""
    prompt = request.json.get("prompt", "").lower()
    similar_recipes = find_similar_recipes(prompt)
    return jsonify({"similar_recipes": similar_recipes})

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
    return jsonify({
        "name": recipe['name'].title(),
        "ingredients": recipe['ingredients'].split(', '),
        "steps": recipe['steps'].split('. '),
        "description": recipe['description']
    })

if __name__ == "__main__":
    app.run(debug=True)
