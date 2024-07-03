from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_menu(prompt):
    # For the sake of this example, we'll simply return the embeddings of the prompt
    # In a real application, you might use these embeddings in various ways
    embeddings = model.encode(prompt)
    return str(embeddings)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        menu = generate_menu(prompt)
        return render_template('index.html', prompt=prompt, menu=menu)
    return render_template('index.html', prompt='', menu='')

if __name__ == "__main__":
    app.run(debug=True)
