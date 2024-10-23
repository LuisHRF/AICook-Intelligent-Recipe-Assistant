import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; 
import logo from './assets/aicook_logo.png'; 

function RecipeApp() {
    const [ingredients, setIngredients] = useState('');
    const [recommendations, setRecommendations] = useState([]);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleRecommend = async () => {
        setLoading(true);
        setError(''); // Limpiar errores anteriores
        try {
            const response = await axios.post('http://localhost:5000/recommend', {
                ingredients: ingredients.split(', ').map(item => item.trim())
            });
            setRecommendations(response.data);
        } catch (err) {
            setError('Failed to fetch recommendations. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    // Function to create a recipe with ingredients
    const handleCreate = async () => {
        setLoading(true);
        setError('');
        try {
            const response = await axios.post('http://localhost:5000/create', {
                ingredients: ingredients.split(', ').map(item => item.trim())
            });
            setRecommendations([response.data]);  
        } catch (err) {
            setError('Failed to create a recipe. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="app">
        {/* Logo at the top */}
        <div className="logo-container">
            <img src={logo} alt="AICook Logo" className="logo" />
        </div>

        <h1>AICook: Recipe Intelligent Assistant</h1>

        {/* Input to get recommendations by ingredients */}
        <input
            type="text"
            value={ingredients}
            onChange={(e) => setIngredients(e.target.value)}
            placeholder="Enter ingredients separated by commas"
            className="input"
        />
        <div className="button-container">
            <button onClick={handleRecommend}>Get Recommendations</button>
            <button onClick={handleCreate}>Create Recipe with Ingredients</button>
        </div>

        {/* Show loading spinner */}
        {loading && <p>Loading...</p>}

        {/* Only show error or results, hide "No recipes found" if nothing has been done yet */}
        {error && <p style={{ color: 'red' }}>{error}</p>}
        <div>
            {/* Rendering recipe results */}
            {!loading && !error && recommendations.length === 0 ? null : (
                recommendations.map((recipe, index) => (
                    <div className="recipe" key={index}>
                        <h2>{recipe.title ? recipe.title : 'No title available'}</h2>
                        <h3>Ingredients:</h3>
                        <ul>
                            {/* Verifica que recipe.ingredients sea un string antes de intentar dividirlo */}
                            {typeof recipe.ingredients === 'string'
                                ? recipe.ingredients.split(',').map((ingredient, idx) => (
                                    <li key={idx}>{ingredient.trim()}</li>
                                ))
                                : <li>No ingredients available</li>}
                        </ul>
                        <h3>Instructions:</h3>
                        <p>{recipe.instructions ? recipe.instructions : "No instructions available"}</p>
                    </div>
                ))
            )}
        </div>
    </div>
);
}

export default RecipeApp;


