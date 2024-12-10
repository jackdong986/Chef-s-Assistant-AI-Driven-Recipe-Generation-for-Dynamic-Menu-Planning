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
    adjectives = ["hearty", "rich", "flavorful", "delicious", "quick", "spicy", "creamy", "crispy", 
                      "easy", "healthy", "tangy", "savory", "zesty", "decadent", "aromatic", "refreshing", 
                      "wholesome", "chewy", "light", "smoky", "buttery", "indulgent", "velvety"]
        
    occasions = ["family gatherings", "weeknight dinners", "special occasions", "holiday meals", 
                     "picnics", "dinner parties", "quick lunch", "meal prep", "romantic dinners", 
                     "birthday celebrations", "anniversaries", "lazy weekends", "game nights"]
        
    meal_times = ["breakfast", "lunch", "dinner", "snack", "brunch", "midnight snack", 
                      "afternoon tea", "weekend brunch", "late-night cravings"]
        
    cooking_techniques = ["grilled", "baked", "stir-fried", "steamed", "roasted", 
                              "sautéed", "smoked", "poached", "seared", "deep-fried"]
        
    cuisine_styles = ["Italian", "Chinese", "Mexican", "Indian", "Thai", "French", 
                          "Japanese", "Korean", "Mediterranean", "Vietnamese", "American", "Caribbean"]
        
    ingredients_list = ["chicken", "beef", "pasta", "tofu", "mushrooms", "tomatoes", 
                            "spinach", "avocado", "cheese", "lemongrass", "ginger", 
                            "coriander", "coconut milk", "onions", "garlic"]
    # Define lists for additional prompts
    name_prompts = [
        f"The ultimate {random.choice(adjectives)} {request.json['name']} side dish.",
        f"{random.choice(cooking_techniques).capitalize()} {request.json['name']} with {random.choice(ingredients_list)}.",
        f"{random.choice(adjectives).capitalize()} {request.json['name']} for {random.choice(occasions)}.",
        f"{random.choice(cuisine_styles)}-style {request.json['name']} recipe.",
        f"A {random.choice(adjectives)} {request.json['name']} that delights."
    ]
    
    description_prompts = [
        f"This {request.json['name']} recipe is a {random.choice(adjectives)} dish that’s perfect for {random.choice(occasions)}.",
        f"A {random.choice(adjectives)} {request.json['name']} dish that’s {random.choice(adjectives)} and {random.choice(adjectives)}.",
        f"Try this {request.json['name']} for a perfect {random.choice(meal_times)}. It’s easy to make and delicious.",
        f"Enjoy this {random.choice(adjectives)} {request.json['name']}, a wonderful addition to {random.choice(occasions)}.",
        f"This {request.json['name']} recipe will impress your guests at {random.choice(occasions)} and is ideal for {random.choice(meal_times)}.",
        f"Packed with flavor and made in no time, {request.json['name']} is perfect for {random.choice(occasions)}.",
        f"An {random.choice(adjectives)} {request.json['name']} that’s {random.choice(adjectives)} and great for {random.choice(meal_times)}."
    ]
    
    tags_prompts = [
        f"Relevant tags for '{request.json['name']}' (comma-separated).",
        f"List keywords or tags associated with the dish {request.json['name']}.",
        f"Suggest tags for {request.json['name']} focusing on dietary and cuisine types.",
        f"Tags for {request.json['name']}: cuisine, occasion, and key ingredients.",
        f"What are the best descriptive tags for {request.json['name']}? Include its {random.choice(cuisine_styles)} origins."
    ]
    
    steps_prompts = [
        f"Step-by-step guide for making '{request.json['name']}' in 10 steps or less.",
        f"Step-by-step guide for a concise recipe method for {request.json['name']}.",
        f"Step-by-step guide for a simple cooking procedure for the dish {request.json['name']}.",
        f"Step-by-step guide for the preparation of {request.json['name']} using {random.choice(cooking_techniques)} techniques.",
        f"Step-by-step guide for how to prepare {request.json['name']} for a {random.choice(meal_times)}."
    ]
    
    ingredients_prompts = [
        f"List the ingredients needed for {request.json['name']}.",
        f"Provide the ingredient list for the dish {request.json['name']}.",
        f"Suggest ingredients for making {request.json['name']}.",
        f"What are the essential {random.choice(cuisine_styles)} ingredients for {request.json['name']}?",
        f"Include {random.choice(ingredients_list)} in the ingredients list for {request.json['name']}."
    ]
    
    # Randomly pick one prompt from each category
    selected_name_prompt = random.choice(name_prompts)
    selected_description_prompt = random.choice(description_prompts)
    selected_tags_prompt = random.choice(tags_prompts)
    selected_steps_prompt = random.choice(steps_prompts)
    selected_ingredients_prompt = random.choice(ingredients_prompts)
    
    # Use the randomly selected prompts to generate content
    name = pipe(selected_name_prompt, max_length=30, num_return_sequences=1)[0]['generated_text'].strip()
    description = pipe(selected_description_prompt, max_length=30, num_return_sequences=1)[0]['generated_text'].strip()
    tags = pipe(selected_tags_prompt, max_length=30, num_return_sequences=1)[0]['generated_text'].strip()
    steps = pipe(selected_steps_prompt, max_length=150, num_return_sequences=1)[0]['generated_text'].split('. ')
    ingredients = pipe(selected_ingredients_prompt, max_length=100, num_return_sequences=1)[0]['generated_text'].split(', ')

    # Clean the generated data
    name = clean_text(name)
    description = clean_text(description)
    steps = [clean_text(step) for step in steps if step.strip()]
    ingredients = [clean_text(ingredient) for ingredient in ingredients if ingredient.strip()]
    tags = [tag.strip() for tag in tags.split(',') if tag.strip()]

    # Create new recipe entry
    new_recipe = {
        'name': name,
        'id': random.randint(100000, 999999),
        'minutes': random.randint(15, 60),
        'contributor_id': random.randint(1000, 9999),
        'submitted': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'tags': ', '.join(tags),
        'n_steps': len(steps),
        'steps': '. '.join(steps),
        'description': description,
        'ingredients': ', '.join(ingredients),
        'n_ingredients': len(ingredients),
        'amount': random.randint(10, 50)
    }

    # Add the new recipe to the dataset
    global recipes_df
    recipes_df = pd.concat([recipes_df, pd.DataFrame([new_recipe])], ignore_index=True)
    recipes_df.to_csv(dataset_path, index=False)

    return jsonify(new_recipe)

if __name__ == '__main__':
    app.run(debug=True)
