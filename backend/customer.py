from flask import Flask, render_template, request, jsonify
import random
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

app = Flask(__name__, template_folder='../ui')

# Initialize the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load saved embeddings and DataFrame
try:
    recipe_embeddings_path = os.path.join(os.path.dirname(__file__), 'backend', 'recipe_embeddings.pt')
    recipe_embeddings = torch.load(recipe_embeddings_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    recipes_df_path = os.path.join(os.path.dirname(__file__), 'backend', 'recipes_df.pkl')
    recipes_df = pd.read_pickle(recipes_df_path)
except FileNotFoundError as e:
    print(f"File not found: {e.filename}")
    recipe_embeddings = torch.tensor([]).cuda() if torch.cuda.is_available() else torch.tensor([])
    recipes_df = pd.DataFrame()

def find_similar_recipes(prompt, dietary_restrictions='', eating_habits='', budget=''):
    # Encode the prompt using the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prompt_embedding = model.encode(prompt, convert_to_tensor=True, device=device)

    if recipe_embeddings.size(0) == 0:
        return []  # Return an empty list if embeddings are not available

    # Calculate cosine similarities between the prompt and recipe embeddings
    similarities = torch.nn.functional.cosine_similarity(prompt_embedding.unsqueeze(0), recipe_embeddings.to(device))

    # Get indices of top 5 most similar recipes
    top_indices = similarities.argsort(descending=True).cpu().numpy()[:5]

    filtered_recipes = []
    for idx in top_indices:
        recipe = recipes_df.iloc[idx]

        # Filter based on dietary restrictions
        if dietary_restrictions:
            restrictions = dietary_restrictions.split(',')
            if any(restriction in recipe['ingredients'] for restriction in restrictions):
                continue

        # Filter based on eating habits
        if eating_habits:
            habits = eating_habits.split(',')
            if not any(habit in recipe['tags'] for habit in habits):
                continue

        # Filter based on budget
        if budget:
            budget = int(budget)
            if not (budget - 20 <= recipe['amount'] <= budget + 20):
                continue

        filtered_recipes.append({
            'name': recipe['name'],
            'description': recipe['description'],
            'ingredients': recipe['ingredients'],
            'amount': recipe['amount'],
            'tags': recipe['tags']
        })

    return filtered_recipes


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        dietary_restrictions = request.form.get('dietary_restrictions', '')
        eating_habits = request.form.get('eating_habits', '')
        budget = request.form.get('budget', '')

        # Find similar recipes based on user input
        similar_recipes = find_similar_recipes(prompt, dietary_restrictions, eating_habits, budget)

        return render_template('restaurantMenuGenerator.html', prompt=prompt, recipes=similar_recipes)

    return render_template('restaurantMenuGenerator.html', prompt='', recipes=None)

@app.route('/generate_random_recipe', methods=['GET'])
def random_recipe():
    random_index = random.randint(0, len(recipes_df) - 1)
    random_recipe = recipes_df.iloc[random_index]
    recipe_name = random_recipe['name']
    recipe_description = random_recipe['description']
    recipe_summary = f"{recipe_name}: {recipe_description[:100]}..."  # Example: Limit description length
    return jsonify(recipe_summary)

if __name__ == "__main__":
    app.run(debug=True)
