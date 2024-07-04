from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import random

app = Flask(__name__)

# Load the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to generate menu based on prompt
def generate_menu(prompt, dietary_restrictions, eating_habits, budget):
    # Generate embeddings for the input prompt
    embeddings = model.encode(prompt + f" Dietary restrictions: {dietary_restrictions}, Eating habits: {eating_habits}, Budget: {budget}")
    # For simplicity, we'll return a fixed set of menu suggestions based on embeddings
    menu = "Salad, Grilled Chicken, and Fruit Salad"
    return menu

# Function to generate random recipe
def generate_random_recipe():
    recipes = [
        "Spaghetti Carbonara: Ingredients - spaghetti, eggs, pancetta, Parmesan cheese, black pepper. Preparation - Boil spaghetti. Cook pancetta. Mix eggs and cheese. Combine all.",
        "Chicken Curry: Ingredients - chicken, curry powder, coconut milk, onions, garlic. Preparation - Sauté onions and garlic. Add chicken and curry powder. Stir in coconut milk. Simmer until chicken is cooked."
    ]
    return random.choice(recipes)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        dietary_restrictions = request.form.get('dietary_restrictions', '')
        eating_habits = request.form.get('eating_habits', '')
        budget = request.form.get('budget', '')
        menu = generate_menu(prompt, dietary_restrictions, eating_habits, budget)
        return render_template('restaurantMenuGenerator.html', prompt=prompt, menu=menu)
    return render_template('restaurantMenuGenerator.html', prompt='', menu='')

@app.route('/generate_random_recipe', methods=['GET'])
def random_recipe():
    recipe = generate_random_recipe()
    return jsonify(recipe)

if __name__ == "__main__":
    app.run(debug=True)
