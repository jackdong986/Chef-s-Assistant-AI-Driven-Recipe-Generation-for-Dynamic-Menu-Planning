# Chef’s Assistant: AI-Driven Recipe Generation for Dynamic Menu Planning

## Project Description

The Chef's Recipe Generator is a web-based application that leverages AI to provide chefs, home cooks, and food enthusiasts with an intuitive platform for exploring, searching, and creating recipes. It integrates natural language processing and machine learning models to deliver personalized recipe recommendations, dynamic menu generation, and easy-to-use features for creativity in the kitchen.

### Key Features

1. **Search for Recipes**: 
   - Search recipes using a text-based query with AI-powered similarity scoring..
   - Provides detailed information, including ingredients, steps, and descriptions.

2. **Random Recipe Generator**:
   - Generate random recipes from specific categories (e.g., Chinese, Western)
   - Explore new and diverse culinary ideas

3. **Custom Recipe Creator**:
   - Describe a dish in your words, and the system generates a name, ingredients, and step-by-step instructions.
   - Automatically suggests complementary ingredients for better results.

4. **All Recipes Viewer**:
   - Filter recipes by alphabetical order or category.
   - Supports pagination for easy navigation.

5. **User-Friendly Interface**:
   - Intuitive design with categorized sections for search, random generation, and recipe creation.
   - Real-time recipe previews and editing options.

### Objectives

1. **Provide a comprehensive recipe management system for personal and professional use.**

2. **Utilize NLP and ML models to enhance the relevance and creativity of recipe suggestions.**

3. **Ensure scalability and performance for handling large recipe datasets.**

## Tools and Technologies:

1. Programming Languages: 
   - Python (Backend)
   - HTML, CSS, and JavaScript (Frontend)
2. Frameworks and Libraries: 
   - Flask (Web Framework)
   - PyTorch (Deep Learning)
   - SentenceTransformers and Hugging Face Transformers (NLP Models)
   - Pandas (Data Processing)
3. Development Tools: 
   - Visual Studio Code
   - Jupyter Notebook
4. Models Used:
   - Hugging Face (get the pre-trained model to fine-tuned)
      - Fine-tuned GPT-2 for text generation
      - Fine-tuned MiniLM for similarity scoring

### Getting Started

To get started with the project, follow these steps:

1. Clone the repository:
   ```sh
   https://github.com/jackdong986/ResMenuFYP.git

2. Download the Dataset fromm Kaggle (without amount column):
   ```sh
   https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions/data

   ```
   or
   With amount column:
   ```sh
   https://drive.google.com/file/d/1GnGDxejJjNbnhYn0Gm6v7S9IOKetM4vX/view?usp=sharing
   ```

3. Change the dataset path to here:
   - dataset_path = r'.\.\RAW_recipes_with_amount.csv'

4. Run the "chef.py"


### Project Structure

1. **./backend/chef.py: Flask server containing the backend API endpoints.**

2. **ui/: Frontend HTML files for the application.**
   - **restaurantMenuGenerator.html: Main interface for recipe creation and random recipes.**
   - **allRecipes.html: Interface for browsing and filtering all recipes.**

3. **fine-tuning/: Directory for storing fine tuned model.**

### Usage
1. **Search Recipes**:
   - Enter keywords to find recipes based on name or description.
   - View sorted results with similarity scores.

2. **Random Recipes**:
   - Select a category to explore randomly generated recipes.
   - View and analyze recipe details interactively.

3. **Create Recipes**:
   - Provide a brief description of the recipe you want.
   - The system generates all components, including ingredients and steps.

4. **Explore All Recipes**:   
   - Use filters and pagination to browse the recipe collection.

   ### Future Enhancements
1. Enable user authentication and personalized recipe saving.
2. Integrate user feedback for improving AI-generated recipes.
3. Expand recipe dataset for broader culinary diversity.
4. Add a mobile-friendly version of the application.
5. Implement to vercel.

# Contributors
   Jack Dong

!Also include the "RAW_recipes_with_amount.csv" in this file!!


Content in this file:
1)backend:
 - chef.py  -> backend for chef side, got a copy for pdf
 - allRecipes.html -> > backend for chef side (show all recipe), got a copy for pdf
2)gpt2-finetuned-recipes -> model which have already fine tuned (gpt2)
3)fine-tuned-minilm-similarity -> model which have already fine tuned (sentences-transformer model (all-MiniLM-L6-v2))
4)fine-tuning -> process for fine tuning, got a copy for pdf
 - fine-tuned-dataset (include data cleaning & preprocessing) -> process for fine tuning dataset
 - fine-tuned-gpt2-recipe -> process for fine tuning gpt2 model
 - fine-tuned-sentences-transformer -> process for fine tuning sentences-transformer model (all-MiniLM-L6-v2)

5) gpt2-finetuned-recipes -> model which have already fine tuned (gpt2)
6) ui, got a copy for pdf:
 - chefRecipeGenerator.html -> ui design for chef side

7)recipe_generation_dataset.txt -> fine tuned dataset, used for gpt2
8)recipe_similarity_pairs.csv -> fine tuned dataset, used for sentences-transformer model (all-MiniLM-L6-v2)


