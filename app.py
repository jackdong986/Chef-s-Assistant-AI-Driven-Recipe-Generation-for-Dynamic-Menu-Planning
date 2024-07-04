from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

app = Flask(__name__)

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Function to generate menu based on prompt
def generate_menu(prompt, dietary_restrictions, eating_habits, budget):
    inputs = tokenizer.encode(prompt + f" Dietary restrictions: {dietary_restrictions}, Eating habits: {eating_habits}, Budget: {budget}", return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    menu = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
