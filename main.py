import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = 'Data.xlsx'
ingredient_df = pd.read_excel(file_path, sheet_name='Ingredients')
customer_df = pd.read_excel(file_path, sheet_name='Customers')

ingredient_df['Details'] = ingredient_df['Nutritional Value'].apply(lambda x: ' '.join(x.split(', ')))

vectorizer = TfidfVectorizer()


tfidf_matrix = vectorizer.fit_transform(ingredient_df['Details'])

def recommend_smoothies(customer_id):
    
    customer = customer_df[customer_df['Customer ID'] == customer_id]
    if customer.empty:
        raise ValueError(f"No customer found with ID {customer_id}")
    customer = customer.iloc[0]
    
    favorite_ingredients = customer['Favorite Ingredients'].split(', ')
    
    
    ingredient_details = []
    for ing in favorite_ingredients:
        details = ingredient_df[ingredient_df['Ingredient'] == ing]['Details'].values
        if len(details) > 0:
            ingredient_details.append(details[0])
    
    if not ingredient_details:
        raise ValueError("None of the favorite ingredients were found in the ingredients dataset.")

    
    customer_vector = vectorizer.transform(ingredient_details)

    
    similarity_scores = cosine_similarity(customer_vector, tfidf_matrix).flatten()
    
    if similarity_scores.size == 0:
        raise ValueError("No similarity scores could be calculated.")
    
    
    recommended_indices = similarity_scores.argsort()[-5:][::-1]

    valid_indices = [index for index in recommended_indices if 0 <= index < len(ingredient_df)]
    
    if not valid_indices:
        raise IndexError("All recommended indices are out-of-bounds.")
    
    recommended_ingredients = ingredient_df.iloc[valid_indices]

    return recommended_ingredients[['Ingredient', 'Calories', 'Nutritional Value']]


customer_id = 101 
recommendations = recommend_smoothies(customer_id)
print(f"Recommendations for Customer {customer_id}:")
print(recommendations)
