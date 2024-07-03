# app.py
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load a pre-trained model for text generation
generator = pipeline('text-generation', model='gpt-3.5-turbo')

def generate_menu(prompt):
    # Generate text based on the input prompt
    response = generator(prompt, max_length=100)
    return response[0]['generated_text']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        menu = generate_menu(prompt)
        return render_template('index.html', prompt=prompt, menu=menu)
    return render_template('index.html', prompt='', menu='')

if __name__ == "__main__":
    app.run(debug=True)
