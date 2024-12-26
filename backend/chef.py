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

#allRecipes.html
@app.route('/all_recipes_page')
def all_recipes_page():
    """Render the All Recipes page."""
    return render_template('allRecipes.html')

import re

@app.route('/filtered_recipes', methods=['GET'])
def filtered_recipes():
    """Fetch filtered recipes based on letter, category, or keyword with pagination."""
    letter = request.args.get('letter', '').lower()
    category = request.args.get('category', '').lower()
    page = int(request.args.get('page', 1))
    page_size = 10
    start = (page - 1) * page_size
    end = start + page_size

    if recipes_df.empty:
        return jsonify({"error": "No recipes found"}), 404
    
    def clean_name(name):
        # Remove leading numbers or unwanted words like "the" or "in"
        name = re.sub(r'^[0-9]+', '', name)  # Remove leading numbers
        name = re.sub(r'^\s*(the|in)\s+', '', name, flags=re.IGNORECASE)  # Remove "the" or "in" at the start
        return name.strip()

    # Filter by letter (starts with the letter)
    if letter:
        recipes_df['name'] = recipes_df['name'].fillna('')  # Replace NaN with an empty string
        recipes_df['clean_name'] = recipes_df['name'].apply(lambda x: clean_name(x))  # Clean the name
        filtered = recipes_df[recipes_df['clean_name'].str.lower().str.startswith(letter)]
    
    # Filter by category (use the category column directly)
    if category:
        filtered = recipes_df[recipes_df['category'].str.lower() == category]
    
    # If no valid filter is applied
    elif not letter and not category:
        return jsonify({"error": "Invalid filter"}), 400

    # Handle case where no recipes match the filter
    if filtered.empty:
        return jsonify({"error": "No recipes match the filter on this page."}), 404

    # Apply pagination
    total_filtered_recipes = len(filtered)  # Get total count of filtered results
    total_pages = (total_filtered_recipes + page_size - 1) // page_size  # Calculate total pages

    # Ensure pagination is within valid range
    if page > total_pages:
        page = total_pages  # If the requested page is greater than total pages, reset to last page

    # Now apply pagination properly
    start = (page - 1) * page_size
    end = start + page_size
    paginated = filtered.iloc[start:end]

    # Populate recipes from the paginated data
    recipes = []
    for _, row in paginated.iterrows():
        recipes.append({
            "name": row.get('name', 'Unknown'),
            "description": row.get('description', 'No description available'),
            "ingredients": row.get('ingredients', '').strip('[]').split(', '),
            "steps": row.get('steps', '').strip('[]').split('. ')
        })

    # Return the response with the recipes and pagination data
    return jsonify({
        "recipes": recipes,
        "total_pages": total_pages,
        "current_page": page
    })

#chefRecipeGenerator.html
@app.route('/')
def home():
    """Render the main page."""
    return render_template('chefRecipeGenerator.html', recipes=None, new_recipe=None)

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
    """Return a random selection of recipes, filtered by category if specified."""
    category = request.args.get('category', 'all').lower()

    # Define keywords for categories (can be reused for preprocessing)
    chinese_keywords = [
        "dumplings", "stir-fry", "peking duck", "dim sum", "hot pot", "sweet and sour",
        "noodles", "wonton", "kung pao", "szechuan", "chow mein", "spring rolls",
        "sweet and sour pork", "kung pao chicken", "mapo tofu", "char siu", "egg foo young",
        "hot and sour soup", "chinese dumplings", "szechuan peppercorns", "beef and broccoli",
        "general tso's chicken", "fried rice", "baozi", "shumai", "peking duck", "lobster cantonese style",
        "chinese bbq ribs", "wonton soup", "dim sum platter", "gong bao chicken", "shanghai soup dumplings"
    ]
    western_keywords = [
        "burger", "steak", "pizza", "sandwich", "pasta", "barbecue", "roast", "salad", "cheesecake",
        "french fries", "chicken wings", "fried chicken", "bbq ribs", "grilled cheese", "fish and chips",
        "buffalo wings", "spaghetti", "lasagna", "beef wellington", "cobb salad", "roast chicken",
        "chicken alfredo", "tacos", "pastrami sandwich", "caesar salad", "pork chops", "meatloaf", "cheeseburger",
        "pulled pork", "clam chowder", "bangers and mash", "bacon and eggs", "steak frites", "currywurst", "goulash",
        "sloppy joes", "quiche", "apple pie", "chicken parmesan", "moussaka", "cornbread"
    ]

    # Pre-compute category classifications if missing
    if 'category' not in recipes_df.columns:
        def classify_recipe(name):
            if pd.isna(name):  # Handle missing names
                return 'other'
            name = name.lower()
            if any(keyword in name for keyword in chinese_keywords):
                return 'chinese'
            elif any(keyword in name for keyword in western_keywords):
                return 'western'
            else:
                return 'other'

        recipes_df['category'] = recipes_df['name'].apply(classify_recipe)
        recipes_df.to_csv(dataset_path, index=False)  # Save back to CSV for future use

    # Filter recipes based on category if provided
    if category == 'chinese':
        filtered_recipes = recipes_df[recipes_df['category'] == 'chinese']
    elif category == 'western':
        filtered_recipes = recipes_df[recipes_df['category'] == 'western']
    else:
        filtered_recipes = recipes_df  # No filtering for 'all'

    if filtered_recipes.empty:
        return jsonify({"error": "No recipes found for the specified category."}), 404

    random_recipes = filtered_recipes.sample(n=min(10, len(filtered_recipes)), random_state=None)[['name']].to_dict(orient='records')

    return jsonify(random_recipes)

@app.route('/recipe_details', methods=['GET'])
def recipe_details():
    """Return details of a specific recipe."""
    name = request.args.get("name", "").strip().lower()

    # Check if name is provided
    if not name:
        return jsonify({"error": "Recipe name is required"}), 400

    # Perform case-insensitive match for the recipe name
    matching_recipes = recipes_df[recipes_df['name'].str.contains(name, case=False, na=False)]

    if matching_recipes.empty:
        return jsonify({"error": "Recipe not found"}), 404

    # Get the first match
    recipe = matching_recipes.iloc[0]

    # Safely extract and parse ingredients and steps
    ingredients = recipe.get('ingredients', '')
    steps = recipe.get('steps', '')

    ingredients = ingredients.strip('[]').split(', ') if pd.notna(ingredients) else []
    steps = steps.strip('[]').split('. ') if pd.notna(steps) else []

    return jsonify({
        "name": recipe['name'].title(),
        "ingredients": [ingredient.strip().strip('"').strip("'") for ingredient in ingredients if ingredient.strip()],
        "steps": [step.strip().strip('"').strip("'") for step in steps if step.strip()]
    })

@app.route('/create_recipe', methods=['POST'])
def create_recipe():
    """Create a new recipe based on a description provided by the user."""
    user_description = request.json.get("description", "").strip()
    if not user_description:
        return jsonify({"error": "Description is required"}), 400

    # Define the prompts dynamically using the user's description
    name_prompt = f"Generate a creative name for: {user_description}."
    description_prompt = f"Describe this dish in detail as a complete sentence: {user_description}."
    ingredients_prompt = f"List the main ingredients such as salt, sugar for {user_description}."
    steps_prompt = f"Provide step-by-step instructions for preparing: {user_description}."

    # Generate the recipe components
    name = pipe(name_prompt, max_length=30, num_return_sequences=1)[0]['generated_text'].strip()
    description = pipe(description_prompt, max_length=100, num_return_sequences=1)[0]['generated_text'].strip()
    steps = pipe(steps_prompt, max_length=150, num_return_sequences=1)[0]['generated_text'].split('. ')
    
    # Adjust the prompt to return the correct ingredients list
    ingredients_response = pipe(ingredients_prompt, max_length=30, num_return_sequences=1)[0]['generated_text']
    
    # Clean and format the ingredients (no splitting into 3 words)
    ingredients = [clean_text(ingredient.strip()) for ingredient in ingredients_response.split(',') if ingredient.strip()]

    # Clean other generated data
    name = clean_text(name)
    description = clean_text(description)
    steps = [clean_text(step) for step in steps if step.strip()]

    # Enhance ingredients list with suitable additions
    additional_ingredients = get_suitable_ingredients(user_description)
    ingredients.extend([ingredient for ingredient in additional_ingredients if ingredient not in ingredients])

    # Create the new recipe entry
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

    # Add the new recipe to the dataset
    global recipes_df
    recipes_df = pd.concat([recipes_df, pd.DataFrame([new_recipe])], ignore_index=True)
    recipes_df.to_csv(dataset_path, index=False)

    return jsonify(new_recipe)

def get_suitable_ingredients(description):
    """Suggest additional ingredients based on the description."""
    additional_ingredients_map = {
    # Indian Recipes
    "spicy chicken curry": ["garam masala", "cumin", "turmeric", "coriander", "ginger", "chilies", "cardamom", "cloves", "bay leaves", "yogurt"],
    
    # Western Recipes
    "pasta": ["parmesan cheese", "basil", "olive oil", "garlic", "oregano", "tomato paste", "black pepper", "cream", "spinach", "mushrooms"],
    "spaghetti": ["parmesan cheese", "basil", "garlic", "olive oil", "tomato sauce", "oregano", "black pepper", "parmesan cheese"],
    "salad": ["lettuce", "cucumber", "olive oil", "lemon juice", "feta cheese", "avocado", "cherry tomatoes", "red onions", "croutons", "balsamic vinegar"],
    "soup": ["bay leaves", "thyme", "celery", "carrots", "parsley", "chicken broth", "cream", "leeks", "garlic", "potatoes"],
    "steak": ["salt", "pepper", "garlic", "butter", "rosemary", "thyme", "onion powder", "paprika", "red wine", "mushrooms"],
    "rib-eye steak": ["rib-eye steak", "olive oil", "garlic", "butter", "rosemary", "salt", "black pepper", "balsamic vinegar"],
    "filet mignon": ["filet mignon", "butter", "garlic", "thyme", "salt", "pepper", "olive oil", "shallots"],
    "burger": ["ground beef", "cheddar cheese", "lettuce", "tomatoes", "pickles", "ketchup", "mustard", "brioche buns", "onions", "bacon"],
    "mac and cheese": ["elbow pasta", "cheddar cheese", "milk", "butter", "flour", "parmesan cheese", "breadcrumbs", "paprika", "garlic powder", "black pepper"],
    "roast chicken": ["whole chicken", "butter", "thyme", "rosemary", "garlic", "lemon", "olive oil", "paprika", "onions", "carrots"],
    "fried chicken": ["chicken pieces", "flour", "buttermilk", "paprika", "garlic powder", "onion powder", "salt", "pepper", "vegetable oil"],
    "grilled chicken": ["chicken breast", "lemon", "garlic", "olive oil", "rosemary", "thyme", "paprika", "salt", "black pepper"],
    "mashed potatoes": ["potatoes", "butter", "milk", "cream", "garlic", "parsley", "salt", "pepper", "chives", "cheddar cheese"],
    "caesar salad": ["romaine lettuce", "parmesan cheese", "croutons", "caesar dressing", "olive oil", "anchovies", "garlic", "lemon juice", "black pepper", "mustard"],
    "lasagna": ["lasagna sheets", "ground beef", "tomato sauce", "ricotta cheese", "mozzarella cheese", "parmesan cheese", "onions", "garlic", "basil", "oregano"],
    "fish and chips": ["cod fish", "potatoes", "flour", "egg", "beer", "salt", "pepper", "lemon", "tartar sauce"],
    "grilled salmon": ["salmon fillets", "olive oil", "lemon", "dill", "garlic", "butter", "salt", "pepper"],
    "clam chowder": ["clams", "potatoes", "onions", "celery", "bacon", "cream", "butter", "flour", "salt", "thyme"],
    "beef stroganoff": ["beef strips", "sour cream", "mushrooms", "onions", "garlic", "paprika", "flour", "beef broth", "egg noodles", "parsley"],
    "shepherd's pie": ["ground lamb", "onions", "carrots", "peas", "potatoes", "butter", "cream", "thyme", "garlic", "cheddar cheese"],
    "quiche": ["eggs", "cream", "cheddar cheese", "spinach", "bacon", "onions", "butter", "flour", "nutmeg", "black pepper"],

    # Chinese Recipes
    "fried rice": ["soy sauce", "eggs", "spring onions", "carrots", "peas", "garlic", "ginger", "sesame oil", "cooked rice", "chicken"],
    "dumplings": ["ground pork", "ginger", "spring onions", "soy sauce", "sesame oil", "wonton wrappers", "cabbage", "garlic", "chili oil", "black vinegar"],
    "hot and sour soup": ["tofu", "wood ear mushrooms", "bamboo shoots", "white pepper", "black vinegar", "egg", "soy sauce", "ginger", "chicken broth", "sesame oil"],
    "sweet and sour pork": ["pork", "pineapple", "bell peppers", "soy sauce", "ketchup", "sugar", "white vinegar", "cornstarch", "garlic", "onions"],
    "kung pao chicken": ["chicken breast", "peanuts", "dried chilies", "soy sauce", "hoisin sauce", "garlic", "ginger", "spring onions", "sesame oil", "cornstarch"],
    "mapo tofu": ["tofu", "ground pork", "doubanjiang (chili bean paste)", "soy sauce", "ginger", "garlic", "spring onions", "Sichuan peppercorns", "sesame oil", "chicken broth"],
    "beef chow mein": ["chow mein noodles", "beef strips", "soy sauce", "oyster sauce", "garlic", "ginger", "carrots", "cabbage", "spring onions", "sesame oil"],
    "Peking duck": ["duck", "hoisin sauce", "cucumber", "spring onions", "mandarin pancakes", "garlic", "ginger", "soy sauce", "honey", "rice vinegar"],
    "char siu pork": ["pork shoulder", "hoisin sauce", "soy sauce", "honey", "five-spice powder", "garlic", "ginger", "rice vinegar", "sesame oil", "sugar"],
    "egg drop soup": ["chicken broth", "eggs", "cornstarch", "spring onions", "soy sauce", "ginger", "white pepper", "sesame oil", "salt", "water"],
    "lo mein": ["lo mein noodles", "soy sauce", "oyster sauce", "ginger", "garlic", "carrots", "cabbage", "mushrooms", "spring onions", "sesame oil"],
    "sesame chicken": ["chicken breast", "soy sauce", "cornstarch", "honey", "garlic", "ginger", "sesame oil", "sesame seeds", "vinegar", "sugar"],
    "wonton soup": ["wonton wrappers", "ground pork", "spring onions", "soy sauce", "ginger", "sesame oil", "chicken broth", "spinach", "garlic", "water chestnuts"],

    # Additional Recipes
    "chicken": ["chicken breast", "garlic", "olive oil", "lemon", "rosemary", "thyme", "butter", "paprika", "onions", "potatoes"],
    "duck": ["duck breast", "orange zest", "soy sauce", "hoisin sauce", "ginger", "garlic", "scallions", "hoisin sauce", "rice vinegar", "star anise"],
    "lamb": ["ground lamb", "garlic", "rosemary", "olive oil", "mint", "yogurt", "cumin", "onions", "paprika", "coriander"],
    "beef": ["ground beef", "onions", "garlic", "olive oil", "tomato paste", "parsley", "oregano", "pepper", "chili flakes", "paprika"],
    "cow": ["beef steaks", "salt", "pepper", "butter", "garlic", "rosemary", "onion powder", "olive oil", "paprika", "mushrooms"],
    "lamb chops": ["lamb chops", "garlic", "rosemary", "lemon", "olive oil", "thyme", "salt", "black pepper", "mint"],
    "beef brisket": ["beef brisket", "brown sugar", "paprika", "garlic", "onion powder", "mustard powder", "black pepper", "salt", "bay leaves"],
    "rack of lamb": ["rack of lamb", "garlic", "rosemary", "olive oil", "salt", "black pepper", "thyme", "lemon", "mustard"],
    "beef wellington": ["beef tenderloin", "puff pastry", "mushrooms", "prosciutto", "egg yolk", "garlic", "onions", "butter", "parmesan"],
    "grilled pork chops": ["pork chops", "garlic", "rosemary", "lemon", "olive oil", "black pepper", "salt", "thyme", "paprika"],
    "grilled shrimp": ["shrimp", "olive oil", "garlic", "lemon", "parsley", "paprika", "salt", "black pepper", "cayenne"],
    "chicken tikka masala": ["chicken breast", "garam masala", "yogurt", "garlic", "onions", "tomato paste", "cream", "cilantro", "cumin", "coriander"],
    "beef fajitas": ["beef strips", "bell peppers", "onions", "garlic", "lime", "chili powder", "cumin", "olive oil", "flour tortillas", "jalapenos"]
}

    # Match based on description keywords
    description = description.lower()
    for key, additional_ingredients in additional_ingredients_map.items():
        if key in description:
            return additional_ingredients

    # Default suggestion if no specific match is found
    return additional_ingredients_map["general"]

if __name__ == '__main__':
    app.run(debug=True)
