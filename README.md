# CIFAR-10 Image Classification

## **Overview**
This project implements an image classification model trained on the **CIFAR-10 dataset** using **PyTorch**. The model classifies images into one of **10 categories**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

To make the model accessible, a **Flask API** is built, and the application is containerized using **Docker**.

### **Project Structure:**
- **Train the model** in Jupyter Notebook (`model_train.ipynb`)
- **Load and use the trained model** in `model.py`
- **Create an API** for predictions using `Flask` (`app.py`)
- **Containerize the application** using Docker (`Dockerfile`)
- **Host the web UI** using `Flask` and `HTML/CSS` (`templates/index.html`)

---
## **Model**
### **Architecture**
The model is based on a custom **AdvancedCNN**, a convolutional neural network (CNN) designed for CIFAR-10 classification. 

#### **AdvancedCNN Architecture:**
- **Convolutional Layers**:
  - `conv1`: Conv2d(3, 64, kernel_size=3, stride=1, padding=1) → `bn1`
  - `conv2`: Conv2d(64, 64, kernel_size=3, stride=1, padding=1) → `bn2` → `pool1`
  - `conv3`: Conv2d(64, 128, kernel_size=3, stride=1, padding=1) → `bn3`
  - `conv3_extra`: Conv2d(128, 128, kernel_size=3, stride=1, padding=1) → `bn3_extra` → `pool2`
  - `conv5`: Conv2d(128, 256, kernel_size=3, stride=1, padding=1) → `bn5`
  - `conv5_extra`: Conv2d(256, 256, kernel_size=3, stride=1, padding=1) → `bn5_extra` → `pool3`

- **Fully Connected Layers**:
  - `fc1`: Linear(4096, 512) → `bn_fc1` → `dropout`
  - `fc2`: Linear(512, 10) → Output (Softmax)

### **Data Preprocessing**
1. **Image Resizing**: All images are resized to **128x128** pixels.
2. **Normalization**: Images are normalized to **[-1,1]** to match model training.
3. **Data Augmentation**: Random transformations (flips, rotations, zoom) improve generalization.

### **Training**
- The dataset is split into **training (80%)**, **validation (10%)**, and **test (10%)**.
- The model is trained using **Cross-Entropy Loss** and optimized using **SGD**.
- Achieved **accuracy of about 85% CIFAR-10 test data.**

---
## **Requirements**
- Jupyter Notebook / Google Colab
- Python 3.7+
- Pip 20.3+
- PyTorch & Torchvision
- Flask for API development
- Docker (for containerization)

---
## **Installation & Running the Web App**

### **1️⃣ Set Up Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate    # On Windows
```

### **2️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/ImageClassificationCIFAR-10.git
cd ImageClassificationCIFAR-10
```
### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```
### 4️⃣ **Run Flask API**
```bash
For Mac/Linux:

export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000

For Windows (CMD):

set FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```
Now, open `http://127.0.0.1:5000` in your browser.

---

## **Docker Deployment**
### 1️⃣ **Build the Docker Image**
```bash
docker build -t cifar10-flask-app .
```
### 2️⃣ **Run the Docker Container**
```bash 
docker run -p 5000:5000 cifar10-flask-app
```

Access the app at `http://localhost:5000`

## **Conclusion**
This project demonstrates image classification with PyTorch, Flask API development, and Docker containerization. Future improvements include:

-Trying different architectures (ResNet-34, MobileNet, EfficientNet)
-Implementing a CI/CD pipeline with GitHub Actions
-Deploying on AWS or other cloud platforms for production use

