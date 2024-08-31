from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch

app = Flask(__name__, template_folder='../ui')

# Load fine-tuned model and tokenizer
generator = pipeline(
    'text-generation',
    model='./fine_tuned_gpt2',
    tokenizer='./fine_tuned_gpt2',
    device=0 if torch.cuda.is_available() else -1
)

# Load SentenceTransformer for similarity search
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load recipes dataset
dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'
recipes_df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Precompute embeddings for recipes (this can be optimized)
recipes_df['embedding'] = recipes_df['description'].apply(lambda x: similarity_model.encode(x, convert_to_tensor=True))

def find_similar_recipes(prompt, dietary_restrictions='', eating_habits='', budget=None, top_k=5):
    prompt_embedding = similarity_model.encode(prompt, convert_to_tensor=True)
    similarities = torch.tensor([
        torch.cosine_similarity(prompt_embedding, recipe_embedding, dim=0)
        for recipe_embedding in recipes_df['embedding']
    ])
    top_indices = torch.topk(similarities, k=top_k).indices.numpy()
    filtered_recipes = []
    for idx in top_indices:
        recipe = recipes_df.iloc[idx]
        
        # Apply dietary restrictions filter
        if dietary_restrictions:
            restrictions = [r.strip().lower() for r in dietary_restrictions.split(',')]
            ingredients = recipe['ingredients'].lower()
            if any(restriction in ingredients for restriction in restrictions):
                continue
        
        # Apply eating habits filter
        if eating_habits:
            habits = [h.strip().lower() for h in eating_habits.split(',')]
            tags = recipe['tags'].lower()
            if not any(habit in tags for habit in habits):
                continue
        
        # Apply budget filter
        if budget:
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
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        dietary_restrictions = request.form.get('dietary_restrictions', '')
        eating_habits = request.form.get('eating_habits', '')
        budget = request.form.get('budget', None)
        budget = int(budget) if budget else None

        # Generate new recipe
        generation_prompt = f"Create a recipe with the following details:\nDietary Restrictions: {dietary_restrictions}\nEating Habits: {eating_habits}\nBudget: {budget}\nRecipe:"
        generated = generator(
            generation_prompt,
            max_length=200,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        new_recipe_text = generated[0]['generated_text']

        # Optionally parse the generated text to extract structured information

        # Find similar recipes
        similar_recipes = find_similar_recipes(prompt, dietary_restrictions, eating_habits, budget)

        return render_template('restaurantMenuGenerator.html', prompt=prompt, new_recipe=new_recipe_text, recipes=similar_recipes)
    return render_template('restaurantMenuGenerator.html')

@app.route('/generate_random_recipe', methods=['GET'])
def random_recipe():
    random_recipe = recipes_df.sample(1).iloc[0]
    return jsonify({
        'name': random_recipe['name'],
        'description': random_recipe['description'],
        'amount': random_recipe['amount']
    })

if __name__ == "__main__":
    app.run(debug=True)
