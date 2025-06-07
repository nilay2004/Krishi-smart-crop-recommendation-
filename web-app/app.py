import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='.') # Use '.' to look for templates in the current directory

@app.route('/')
def index():
    return render_template('index.html')

# Placeholder for Crop Recommendation route
@app.route('/crop_recommendation', methods=['POST'])
def crop_recommendation():
    # Implement your crop recommendation logic here
    data = request.json
    # Example: Process data and return a recommendation
    return jsonify({'recommendation': 'Recommended Crop based on input'}) # Replace with actual logic

# Placeholder for Plant Disease Identification route
@app.route('/plant_disease_identification', methods=['POST'])
def plant_disease_identification():
    # Implement your plant disease identification logic here
    # This might involve handling file uploads for images
    return jsonify({'disease': 'Disease detected', 'treatment': 'Treatment suggestions'}) # Replace with actual logic

if __name__ == '__main__':
    # Use PORT environment variable for Render deployment, default to 5000 for local
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port) 