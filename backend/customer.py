from flask import Flask, render_template, request, jsonify
import random
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from transformers import pipeline

app = Flask(__name__, template_folder='../ui')

# Initialize the text generation pipeline
pipe = pipeline("text-generation", model="openai-community/gpt2-large")

# Initialize the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the dataset and saved embeddings
try:
    dataset_path = os.path.join(os.path.dirname(__file__), 'backend', 'RAW_recipes_ten_records.csv')
    recipes_df = pd.read_csv(dataset_path)
    
    recipe_embeddings_path = os.path.join(os.path.dirname(__file__), 'backend', 'recipe_embeddings.pt')
    recipe_embeddings = torch.load(recipe_embeddings_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
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

        # Filter out recipes based on dietary restrictions
        if dietary_restrictions:
            restrictions = dietary_restrictions.split(',')
            if any(restriction.strip().lower() in recipe['ingredients'].lower() for restriction in restrictions):
                continue

        # Prefer recipes with eating habits
        if eating_habits:
            habits = eating_habits.split(',')
            if not any(habit.strip().lower() in recipe['tags'].lower() for habit in habits):
                continue

        # Filter based on budget (within a range of ±20)
        if budget:
            budget = int(budget)
            if not (budget - 20 <= recipe['amount'] <= budget + 20):
                continue

        filtered_recipes.append({
            'name': recipe['name'],
            'description': recipe['description'],
            'amount': recipe['amount']
        })

    return filtered_recipes

@app.route('/', methods=['GET', 'POST'])
def home():
    global recipes_df  # Ensure we are modifying the global variable

    if request.method == 'POST':
        prompt = request.form['prompt']
        dietary_restrictions = request.form.get('dietary_restrictions', '')
        eating_habits = request.form.get('eating_habits', '')
        budget = request.form.get('budget', '')

        # Find similar recipes based on user input
        similar_recipes = find_similar_recipes(prompt, dietary_restrictions, eating_habits, budget)

        # Generate new recipes using GPT-2
        new_recipe = pipe(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']

        # Add the new recipe to the dataframe
        new_recipe_data = {
            'name': new_recipe[:30],  # Truncate name
            'description': new_recipe,
            'amount': random.randint(50, 200),  # Random budget
        }
        recipes_df = pd.concat([recipes_df, pd.DataFrame([new_recipe_data])], ignore_index=True)

        # Ensure the directory exists
        dataset_directory = os.path.dirname(dataset_path)
        if not os.path.exists(dataset_directory):
            os.makedirs(dataset_directory)

        # Save the new recipe to the CSV file
        recipes_df.to_csv(dataset_path, index=False)

        return render_template('restaurantMenuGenerator.html', prompt=prompt, recipes=similar_recipes)

    return render_template('restaurantMenuGenerator.html', prompt='', recipes=None)

@app.route('/generate_random_recipe', methods=['GET'])
def random_recipe():
    random_recipes = recipes_df.sample(n=10)
    random_recipes_list = random_recipes[['name', 'amount', 'description']].to_dict(orient='records')
    return jsonify(random_recipes_list)

if __name__ == "__main__":
    app.run(debug=True)
