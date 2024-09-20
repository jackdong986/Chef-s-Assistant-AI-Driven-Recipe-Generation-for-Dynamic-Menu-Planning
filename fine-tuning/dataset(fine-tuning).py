import pandas as pd
from datasets import Dataset

# Load the dataset (for read & write)
dataset_path = r'C:\Users\Jack\Desktop\foodRecipeAndInteractions\RAW_recipes_with_amount.csv'
recipes_df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

# Prepare the txt data
def create_recipe_text(row):
    return f"""
Recipe Name: {row['name']}
ID: {row['id']}
Time: {row['minutes']} minutes
Contributor: {row['contributor_id']}
Date Submitted: {row['submitted']}
Tags: {row['tags']}
Nutrition: {row['nutrition']}
Steps: {row['steps']}
Description: {row['description']}
Ingredients: {row['ingredients']}
Number of Ingredients: {row['n_ingredients']}
Price: {row['amount']}
"""

# Apply the function to each row
recipes_df['text'] = recipes_df.apply(create_recipe_text, axis=1)

# Save to a txt file
with open(r'C:/Users/Jack/Documents/GitHub/ResMenuFYP/fine-tuning/train.txt', 'w', encoding='utf-8') as f:
    for line in recipes_df['text']:
        f.write(line + '\n')
