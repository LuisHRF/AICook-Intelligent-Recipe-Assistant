from flask import Flask, request, jsonify
from flask_cors import CORS  
from models.recommend_recipes import find_most_similar_recipe
from models.create_recipe_ai import create_recipe_from_ingredients
from models.find_similar_recipes import find_similar_recipe_flow

app = Flask(__name__)
CORS(app)  

@app.route('/')
def home():
    return jsonify({"message": "Welcome to AICook: Recipe Intelligent Assistant!"}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    user_ingredients = request.json.get('ingredients')
    if not user_ingredients:
        return jsonify({"error": "No ingredients provided"}), 400

    best_recipe = find_most_similar_recipe(user_ingredients)
    if best_recipe:
        return jsonify([best_recipe]), 200  # Asegúrate de retornar un array
    else:
        return jsonify([]), 404 

@app.route('/create', methods=['POST'])
def create():
    try:
        # Get the ingredients from the request body
        ingredients = request.json.get('ingredients')
        if not ingredients:
            return jsonify({"error": "No ingredients provided"}), 400

        # Call the function to create the recipe
        recipe_response = create_recipe_from_ingredients(ingredients)
        if "error" in recipe_response:
            return jsonify({"message": recipe_response["error"]}), 404

        # Return the generated recipe
        return jsonify(recipe_response), 201

    except Exception as e:
        print(f"Error processing /create: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/find_similar', methods=['POST'])
def find_similar():
    recipe_name = request.json.get('recipe_name')
    if not recipe_name:
        return jsonify({"error": "No recipe name provided"}), 400

    similar_recipes = find_similar_recipe_flow(recipe_name)
    if similar_recipes:
        return jsonify(similar_recipes), 200
    else:
        return jsonify({"message": "No similar recipes found."}), 404

if __name__ == '__main__':
    app.run(debug=True)