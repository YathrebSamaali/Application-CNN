# app/app.py - Version simplifiée
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Charger le modèle
model = load_model("models/best_model.keras")
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def prepare_image(img):
    """Préparer l'image pour la prédiction"""
    img = image.load_img(img, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET'])
def index():
    """Page d'accueil avec formulaire HTML"""
    return render_template('index.html', prediction='')

@app.route('/classify', methods=['POST'])
def classify():
    """API endpoint pour la classification"""
    try:
        # Vérifier si un fichier a été envoyé
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        # Vérifier si un fichier a été sélectionné
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Sauvegarder temporairement le fichier
        temp_path = "temp_image.jpg"
        file.save(temp_path)
        
        # Préparer et prédire
        img_array = prepare_image(temp_path)
        predictions = model.predict(img_array, verbose=0)
        
        # Obtenir les résultats
        predicted_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100
        
        # Nettoyer le fichier temporaire
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2)
        })
        
    except Exception as e:
        # Nettoyer en cas d'erreur
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")
        
        return jsonify({
            'success': False,
            'error': f'Error during processing: {str(e)}'
        }), 500

# Ancienne route pour compatibilité (optionnelle)
@app.route('/', methods=['POST'])
def index_post():
    """Ancienne route pour compatibilité avec le formulaire HTML simple"""
    prediction = ''
    if request.method == 'POST':
        file = request.files['file']
        file.save("temp.jpg")
        img = prepare_image("temp.jpg")
        pred = model.predict(img)
        prediction = class_names[np.argmax(pred)]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)