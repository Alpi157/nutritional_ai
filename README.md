# 🍲 Image-to-Recipe Nutritional AI

Automatically estimate the calories, macronutrients, and recipe from a single photo of your food.

## 📌 Project Overview

This AI system integrates computer vision, natural language processing, and semantic mapping to estimate per-serving nutritional content from food images. The pipeline predicts:
- Dish class from the Food-101 dataset
- Multi-label ingredients from the RecipeNLG dataset
- Recipes and cooking instructions
- Nutritional values using the USDA FoodData Central API

A Flask-based web interface allows easy interaction from the browser.

## 🧠 Features

- 🔍 **Dish Classification** – ResNet50 trained on Food-101 (86% top-1 accuracy)
- 🧂 **Ingredient Prediction** – ResNet18 trained on top 500 ingredients from RecipeNLG
- 📚 **Recipe Retrieval** – Uses semantic similarity from Sentence-BERT
- ⚖️ **Quantity Estimation** – Hybrid rule-based and T5 transformer parser
- 🥗 **Nutrition Estimation** – USDA nutritional database mapping
- 🌐 **Web App** – Upload a photo, get nutrition, recipe & breakdown

## 🧪 Tech Stack

- PyTorch, Torchvision, Sentence-BERT, T5
- Flask, HTML/CSS
- Datasets: Food-101, RecipeNLG, USDA FoodData Central

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Alpi157/nutritional_ai.git
   cd nutritional_ai
Install dependencies:
pip install -r requirements.txt
Run the web app:
python app.py
Open localhost:5000 in your browser.

📊 Results
Top-1 classification accuracy: 86.3%

Nutrition profile includes: calories, protein, carbohydrates, fat

Real-time processing via web interface

📁 Project Structure
csharp
Copy
Edit
nutritional_ai/
│
├── classification/        # ResNet50 model training
├── ingredient_model/      # ResNet18 model training
├── app.py                 # Flask server
├── utils/                 # Parsing, semantic matching
├── templates/             # HTML templates
├── static/                # CSS and sample outputs
├── USDA_lookup/           # USDA matching scripts
├── requirements.txt
└── README.md
📝 Credits
Developed by [Team 6 - Image-to-Recipe Nutritional AI] as part of ECE 569A - Artificial Intelligence (UVic).
