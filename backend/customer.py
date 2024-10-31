from flask import Flask, render_template, request, jsonify
import random
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='../ui')

pipe = pipeline("text-generation", model="openai-community/gpt2-large")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'

if os.path.exists(dataset_path):
    recipes_df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
else:
    recipes_df = pd.DataFrame(columns=[
        'name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags', 
        'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 
        'n_ingredients', 'amount'
    ])

def encode_text(text):
    """Encode text using AutoTokenizer and AutoModel, returning a normalized embedding."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)  

def find_similar_recipes(prompt, dietary_restrictions='', eating_habits='', budget=''):
    prompt_embedding = encode_text(prompt)

    recipe_embeddings = []
    recipe_details = []

    for _, recipe in recipes_df.iterrows():
        tags = recipe['tags']
        
        if isinstance(tags, float) and pd.isna(tags):  # Skip NaN tags
            continue
        elif isinstance(tags, str):
            try:
                tags = eval(tags)  # Convert string representation to a list
                if not isinstance(tags, list):
                    continue  # Skip if eval returns something other than a list
            except:
                continue  # Skip if eval fails
        elif not isinstance(tags, list):  # Skip if tags is not a list
            continue

        tags = [str(tag).lower() for tag in tags]  # Ensure all tags are lowercase strings

        # Filter based on dietary restrictions
        if dietary_restrictions:
            restrictions = [r.strip().lower() for r in dietary_restrictions.split(',')]
            if not all(restriction in tags for restriction in restrictions):
                continue

        # Filter based on eating habits
        if eating_habits:
            habits = [h.strip().lower() for h in eating_habits.split(',')]
            if not any(habit in tags for habit in habits):
                continue

        # Filter based on budget
        if budget:
            try:
                recipe_amount = float(recipe['amount'])  # Convert amount to float
                budget = float(budget)
                if not (budget - 20 <= recipe_amount <= budget + 20):
                    continue
            except ValueError:
                continue  # Skip if conversion fails

        # Encode recipe name and add to embeddings list
        recipe_embedding = encode_text(recipe['name'])
        recipe_embeddings.append(recipe_embedding)
        recipe_details.append({
            'name': recipe['name'],
            'amount': recipe['amount'],
            'description': recipe['description']
        })

    # Calculate cosine similarities
    if recipe_embeddings:
        recipe_embeddings = torch.cat(recipe_embeddings, dim=0)
        similarities = cosine_similarity(prompt_embedding.cpu().numpy(), recipe_embeddings.cpu().numpy())
        similarity_scores = similarities[0]

        # Sort recipes by similarity
        sorted_indices = similarity_scores.argsort()[::-1]
        filtered_recipes = [recipe_details[i] for i in sorted_indices[:10]]

        most_similar_recipe = filtered_recipes[0] if filtered_recipes else None
        return filtered_recipes, most_similar_recipe
    else:
        return [], None
    
@app.route('/', methods=['GET', 'POST'])
def home():
    global recipes_df 

    if request.method == 'POST':
        prompt = request.form['prompt']
        dietary_restrictions = request.form.get('dietary_restrictions', '')
        eating_habits = request.form.get('eating_habits', '')
        budget = request.form.get('budget', '')

        similar_recipes, most_similar_recipe = find_similar_recipes(prompt, dietary_restrictions, eating_habits, budget)

        name_prompt = f"Dish name: {prompt}."
        description_prompt = f"An enticing description of {prompt}, focusing on flavors and textures."

        tags_prompt = f"Relevant tags for a dish called '{prompt}'."
        steps_prompt = f"Step-by-step guide for making '{prompt}' in 10 steps or less."
        ingredients_prompt = f"List the ingredients needed for {prompt}."

        # Generate each part of the recipe using GPT-2
        name = pipe(name_prompt, max_length=10, num_return_sequences=1)[0]['generated_text'].strip()
        description = pipe(description_prompt, max_length=50, num_return_sequences=1)[0]['generated_text'].strip()
        tags = pipe(tags_prompt, max_length=50, num_return_sequences=1)[0]['generated_text'].split(', ')
        steps = pipe(steps_prompt, max_length=150, num_return_sequences=1)[0]['generated_text'].split('. ')
        ingredients = pipe(ingredients_prompt, max_length=100, num_return_sequences=1)[0]['generated_text'].split(', ')

        # Add the new recipe to the dataframe with all fields
        new_recipe_data = {
            'name': name,
            'id': random.randint(100000, 999999),  # Generate a random unique id
            'minutes': random.randint(15, 60),  # Random preparation time
            'contributor_id': random.randint(1000, 9999),  # Random contributor id
            'submitted': pd.Timestamp.now().strftime('%Y/%m/%d'),  # Current date
            'tags': ', '.join(tags),  # Join tags list into a string
            
            # Generated values and randomly assigned fields
            'nutrition': [random.randint(100, 500) for _ in range(7)],  
            'n_steps': len(steps),  
            'steps': steps,  
            'description': description,
            'ingredients': ingredients, 
            'n_ingredients': len(ingredients),  
            'amount': random.randint(50, 200),  
        }

        # Append the new recipe to the DataFrame and save
        recipes_df = pd.concat([recipes_df, pd.DataFrame([new_recipe_data])], ignore_index=True)
        recipes_df.to_csv(dataset_path, index=False)  # Save the full dataset back to CSV

        return render_template(
            'restaurantMenuGenerator.html', 
            recipes=similar_recipes, 
            new_recipe=new_recipe_data,
            most_similar_recipe=most_similar_recipe
        )

    return render_template('restaurantMenuGenerator.html', recipes=None, new_recipe=None, most_similar_recipe=None)

@app.route('/generate_random_recipe', methods=['GET'])
def random_recipe():
    random_recipes = recipes_df.sample(n=10)
    random_recipes_list = random_recipes[['name', 'amount', 'description', 'id', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'ingredients', 'n_ingredients']].to_dict(orient='records')
    return jsonify(random_recipes_list)

if __name__ == "__main__":
    app.run(debug=True)
