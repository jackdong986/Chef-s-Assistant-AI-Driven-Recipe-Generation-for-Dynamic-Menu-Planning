from flask import Flask, render_template, request, jsonify
import random
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='../ui')

#pipe = pipeline("text-generation", model="openai-community/gpt2-large")
pipe = pipeline("text-generation", model="fine-tuned-gpt2-recipe")

#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-minilm-similarity")
model = AutoModel.from_pretrained("fine-tuned-minilm-similarity")


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
                tags = eval(tags)  
                if not isinstance(tags, list):
                    continue  
            except:
                continue  
        elif not isinstance(tags, list): 
            continue

        tags = [str(tag).lower() for tag in tags]  

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
                recipe_amount = float(recipe['amount'])  
                budget = float(budget)
                if not (budget - 20 <= recipe_amount <= budget + 20):
                    continue
            except ValueError:
                continue  

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

        # Define multiple prompts for each part of the recipe
        name_prompts = [
            f"Dish name: {prompt}.",
            f"Provide a creative name for a dish inspired by {prompt}.",
            f"Suggest a catchy name for the dish: {prompt}."
        ]
        
        description_prompts = [
            f"An enticing description of {prompt}, focusing on flavors and textures.",
            f"Describe {prompt} with an emphasis on its culinary appeal.",
            f"Write a brief, delicious description of the dish: {prompt}."
        ]
        
        tags_prompts = [
            f"Relevant tags for '{prompt}' (comma-separated).",
            f"List keywords or tags associated with the dish {prompt}.",
            f"Suggest tags for {prompt} focusing on dietary and cuisine types."
        ]
        
        steps_prompts = [
            f"Step-by-step guide for making '{prompt}' in 10 steps or less.",
            f"Provide a concise recipe method for {prompt}.",
            f"Write a simple cooking procedure for the dish {prompt}."
        ]
        
        ingredients_prompts = [
            f"List the ingredients needed for {prompt}.",
            f"Provide the ingredient list for the dish {prompt}.",
            f"Suggest ingredients for making {prompt}."
        ]

        # Randomly pick one prompt from each category
        selected_name_prompt = random.choice(name_prompts)
        selected_description_prompt = random.choice(description_prompts)
        selected_tags_prompt = random.choice(tags_prompts)
        selected_steps_prompt = random.choice(steps_prompts)
        selected_ingredients_prompt = random.choice(ingredients_prompts)

        # Generate each part of the recipe using GPT-2
        name = pipe(selected_name_prompt, max_length=30, num_return_sequences=1)[0]['generated_text'].strip()
        description = pipe(selected_description_prompt, max_length=30, num_return_sequences=1)[0]['generated_text'].strip()
        tags = pipe(selected_tags_prompt, max_length=50, num_return_sequences=1)[0]['generated_text'].split(', ')
        steps = pipe(selected_steps_prompt, max_length=150, num_return_sequences=1)[0]['generated_text'].split('. ')
        ingredients = pipe(selected_ingredients_prompt, max_length=100, num_return_sequences=1)[0]['generated_text'].split(', ')

        new_recipe_data = {
            'name': name,
            'id': random.randint(100000, 999999), 
            'minutes': random.randint(15, 60),  
            'contributor_id': random.randint(1000, 9999), 
            'submitted': pd.Timestamp.now().strftime('%Y/%m/%d'),  
            'tags': ', '.join(tags),  # Join tags list into a string
            'nutrition': [random.randint(100, 500) for _ in range(7)],  
            'n_steps': len(steps),  
            'steps': steps,  
            'description': description,
            'ingredients': ingredients, 
            'n_ingredients': len(ingredients),  
            'amount': random.randint(50, 200),  
        }

        recipes_df = pd.concat([recipes_df, pd.DataFrame([new_recipe_data])], ignore_index=True)
        recipes_df.to_csv(dataset_path, index=False)  

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
