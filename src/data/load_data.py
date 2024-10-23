import pandas as pd

#Path from data
recipes_path = "data/raw/recipes.parquet"
review_path = "data/raw/reviews.parquet"

def load_parquet_data():
    recipes_raw = pd.read_parquet(recipes_path)
    review_raw = pd.read_parquet(review_path)

    print("Recipes data loaded. Number of recipes:", recipes_raw.shape[0])
    print("Reviews data loaded. Number of Reviews:", review_raw.shape[0])

    return recipes_raw, review_raw

def inspect_data(recipes, reviews):
    print("\nFirst rows of the recipes dataset:")
    print(recipes_raw.head())
    
    print("\nFirst rows of the reviews dataset:")
    print(review_raw.head())
    
    print("\nRecipes dataset info:")
    print(recipes_raw.info())
    
    print("\nReviews dataset info:")
    print(review_raw.info())

if __name__ == "__main__":
    recipes_raw, review_raw = load_parquet_data()
    inspect_data(recipes_raw, review_raw)

