import os
from flask import Flask, request, render_template, url_for
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import load_model

# Load the model
model = load_model()

# Create Flask app
app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    filename = image_file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Save the uploaded image
    image_file.save(filepath)

    # Open and preprocess image
    image = Image.open(filepath).convert('RGB')
    preprocessed_image = preprocess_image(image)

    # Predict with model
    with torch.no_grad():
        outputs = model(preprocessed_image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.max(probabilities, dim=0)

    # CIFAR-10 class labels
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    predicted_label = CIFAR10_CLASSES[top_class.item()]

    # Return result
    return render_template('index.html', 
                           user_image=url_for('static', filename=f'uploads/{filename}'), 
                           pred=predicted_label, 
                           prob=round(top_prob.item() * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)