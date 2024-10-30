from flask import Flask, render_template, request, jsonify
import random
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from transformers import pipeline

app = Flask(__name__, template_folder='../ui')

# v1 GPT model (hasn't been fine-tuned)
pipe = pipeline("text-generation", model="openai-community/gpt2-large")

# SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Path for dataset
dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'

# Load the dataset and embeddings if they exist
if os.path.exists(dataset_path):
    recipes_df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
else:
    recipes_df = pd.DataFrame(columns=[
        'name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags', 
        'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 
        'n_ingredients', 'amount'
    ])

recipe_embeddings_path = os.path.join(os.path.dirname(dataset_path), 'recipe_embeddings.pt')
if os.path.exists(recipe_embeddings_path):
    recipe_embeddings = torch.load(recipe_embeddings_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
else:
    recipe_embeddings = torch.tensor([]).cuda() if torch.cuda.is_available() else torch.tensor([])

def find_similar_recipes(prompt, dietary_restrictions='', eating_habits='', budget=''):
    # Encode the prompt using the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prompt_embedding = model.encode(prompt, convert_to_tensor=True, device=device)

    if recipe_embeddings.size(0) == 0:
        return [], None  # Return an empty list and None if embeddings are not available

    # Calculate cosine similarities between the prompt and recipe embeddings
    similarities = torch.nn.functional.cosine_similarity(prompt_embedding.unsqueeze(0), recipe_embeddings.to(device))

    # Get indices of top 10 most similar recipes
    top_indices = similarities.argsort(descending=True).cpu().numpy()[:10]

    filtered_recipes = []
    for idx in top_indices:
        recipe = recipes_df.iloc[idx]

        # Filter out recipes based on dietary restrictions
        if dietary_restrictions:
            restrictions = dietary_restrictions.split(',')
            if any(restriction.strip().lower() in recipe['description'].lower() for restriction in restrictions):
                continue

        # Prefer recipes with eating habits
        if eating_habits:
            habits = eating_habits.split(',')
            if not any(habit.strip().lower() in recipe['description'].lower() for habit in habits):
                continue

        # Filter based on budget (within a range of ±20)
        if budget:
            budget = int(budget)
            if not (budget - 20 <= recipe['amount'] <= budget + 20):
                continue

        filtered_recipes.append({
            'name': recipe['name'],
            'amount': recipe['amount'],
            'description': recipe['description']
        })

    most_similar_recipe = filtered_recipes[0] if filtered_recipes else None
    return filtered_recipes, most_similar_recipe

@app.route('/', methods=['GET', 'POST'])
def home():
    global recipes_df 

    if request.method == 'POST':
        prompt = request.form['prompt']
        dietary_restrictions = request.form.get('dietary_restrictions', '')
        eating_habits = request.form.get('eating_habits', '')
        budget = request.form.get('budget', '')

        # Find similar recipes based on user input
        similar_recipes, most_similar_recipe = find_similar_recipes(prompt, dietary_restrictions, eating_habits, budget)

        # Generate new recipe (using GPT-2 pipeline)
        new_recipe = pipe(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

        # Add the new recipe to the dataframe with all fields
        new_recipe_data = {
            'name': new_recipe[:30],  # Truncate name
            'id': random.randint(100000, 999999),  # Generate a random unique id
            'minutes': random.randint(15, 60),  # Random preparation time
            'contributor_id': random.randint(1000, 9999),  # Random contributor id
            'submitted': pd.Timestamp.now().strftime('%Y/%m/%d'),  # Current date
            'tags': 'generated',  # Tag indicating generated recipe
            
            # Randomly generated values
            'nutrition': [random.randint(100, 500) for _ in range(7)],  
            'n_steps': random.randint(3, 10),  
            'steps': ['Generated step'] * random.randint(3, 10),  
            'description': new_recipe,
            'ingredients': ['generated ingredient'] * random.randint(3, 10), 
            'n_ingredients': random.randint(3, 10),  
            'amount': random.randint(50, 200),  
        }

        # Append the new recipe to the DataFrame and save
        recipes_df = pd.concat([recipes_df, pd.DataFrame([new_recipe_data])], ignore_index=True)
        recipes_df.to_csv(dataset_path, index=False)  # Save the full dataset back to CSV

        # Pass both similar and new recipe to the template
        return render_template('restaurantMenuGenerator.html', prompt=prompt, recipes=similar_recipes, new_recipe=new_recipe_data)

    # Render template with no recipes if GET request
    return render_template('restaurantMenuGenerator.html', prompt='', recipes=None, new_recipe=None)

@app.route('/generate_random_recipe', methods=['GET'])
def random_recipe():
    random_recipes = recipes_df.sample(n=10)
    random_recipes_list = random_recipes[['name', 'amount', 'description', 'id', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'ingredients', 'n_ingredients']].to_dict(orient='records')
    return jsonify(random_recipes_list)

if __name__ == "__main__":
    app.run(debug=True)
