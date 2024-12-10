from flask import Flask, render_template, request, jsonify
import pandas as pd
import random
import re
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Flask setup
app = Flask(__name__, template_folder='../ui')

# Load models
pipe = pipeline("text-generation", model="fine-tuned-gpt2-recipe")
sentence_model = SentenceTransformer('fine-tuned-minilm-similarity')

# Load dataset
dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'
if os.path.exists(dataset_path):
    recipes_df = pd.read_csv(dataset_path, encoding='ISO-8859-1', low_memory=False)
else:
    recipes_df = pd.DataFrame(columns=[
        'name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags',
        'nutrition', 'n_steps', 'steps', 'description', 'ingredients',
        'n_ingredients', 'amount'
    ])

# Utility function to clean text
def clean_text(text):
    """Remove unwanted characters and ensure uniform formatting."""
    text = re.sub(r"\s*\(.*?\)", "", text)  # Remove text inside parentheses
    text = re.sub(r"[^\w\s\-:,.]+", "", text).strip()  # Remove invalid characters
    return text

@app.route('/')
def home():
    """Render the main page."""
    return render_template('restaurantMenuGenerator.html', recipes=None, new_recipe=None)

@app.route('/search', methods=['POST'])
def search_recipe():
    """Search for recipes by name and calculate cosine similarity."""
    prompt = request.json.get("prompt", "").lower().strip()

    if not prompt:
        return jsonify({"recipes": []})

    # Find matching recipes by name in the CSV
    matching_recipes = recipes_df[recipes_df['name'].str.contains(prompt, case=False, na=False)]

    if matching_recipes.empty:
        return jsonify({"recipes": []})

    # Encode the search query using the sentence model
    prompt_embedding = sentence_model.encode(prompt)

    # Calculate cosine similarity for each recipe
    similarities = []
    for _, recipe in matching_recipes.iterrows():
        recipe_text = f"{recipe['name']} {recipe['description']}"
        recipe_embedding = sentence_model.encode(recipe_text)
        similarity = cosine_similarity([prompt_embedding], [recipe_embedding])[0][0]
        similarities.append((recipe['name'], similarity))

    # Sort by similarity and return the top 10 recipes
    sorted_recipes = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    recipes = [{"name": r[0], "similarity": f"{r[1]:.4f}"} for r in sorted_recipes]

    return jsonify({"recipes": recipes})

@app.route('/random', methods=['GET'])
def random_recipe():
    """Return a list of 10 random recipes."""
    random_recipes = recipes_df.sample(n=10)[['name', 'ingredients', 'steps']].to_dict(orient='records')
    return jsonify(random_recipes)

@app.route('/recipe_details', methods=['GET'])
def recipe_details():
    """Return details of a specific recipe."""
    name = request.args.get("name", "").lower().strip()
    matching_recipes = recipes_df[recipes_df['name'].str.lower() == name]

    if matching_recipes.empty:
        return jsonify({"error": "Recipe not found"}), 404

    recipe = matching_recipes.iloc[0]

    ingredients = recipe['ingredients']
    steps = recipe['steps']

    # Ensure ingredients and steps are correctly parsed
    ingredients = ingredients.strip('[]').split(', ') if pd.notna(ingredients) else []
    steps = steps.strip('[]').split('. ') if pd.notna(steps) else []

    return jsonify({
        "name": recipe['name'].title(),
        "ingredients": [ingredient.strip().strip('"').strip("'") for ingredient in ingredients if ingredient.strip()],
        "steps": [step.strip().strip('"').strip("'") for step in steps if step.strip()]
    })

@app.route('/create_recipe', methods=['POST'])
def create_recipe():
    """Create a new recipe."""
    name_prompt = f"The ultimate {request.json['name']} recipe."
    description_prompt = f"This {request.json['name']} recipe is perfect for {random.choice(['family dinners', 'special occasions'])}."
    steps_prompt = f"Step-by-step guide for making {request.json['name']}."
    ingredients_prompt = f"List the ingredients needed for {request.json['name']}."

    name = pipe(name_prompt, max_length=30, num_return_sequences=1)[0]['generated_text'].strip()
    description = pipe(description_prompt, max_length=30, num_return_sequences=1)[0]['generated_text'].strip()
    steps = pipe(steps_prompt, max_length=150, num_return_sequences=1)[0]['generated_text'].split('. ')
    ingredients = pipe(ingredients_prompt, max_length=100, num_return_sequences=1)[0]['generated_text'].split(', ')

    # Clean the generated data
    name = clean_text(name)
    description = clean_text(description)
    steps = [clean_text(step) for step in steps if step.strip()]
    ingredients = [clean_text(ingredient) for ingredient in ingredients if ingredient.strip()]

    new_recipe = {
        'name': name,
        'id': random.randint(100000, 999999),
        'minutes': random.randint(15, 60),
        'contributor_id': random.randint(1000, 9999),
        'submitted': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'tags': request.json.get('tags', ''),
        'n_steps': len(steps),
        'steps': '. '.join(steps),
        'description': description,
        'ingredients': ', '.join(ingredients),
        'n_ingredients': len(ingredients),
        'amount': random.randint(10, 50)
    }

    global recipes_df
    recipes_df = pd.concat([recipes_df, pd.DataFrame([new_recipe])], ignore_index=True)
    recipes_df.to_csv(dataset_path, index=False)

    return jsonify(new_recipe)

if __name__ == '__main__':
    app.run(debug=True)
