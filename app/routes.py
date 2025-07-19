from flask import Flask, request, Response
from flask.templating import render_template
from flask import request
from werkzeug.utils import secure_filename
from app import app
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.models
import os

# Load model once at startup
MODEL = None
CLASSES = ['acanthosis-nigricans', 'acne', 'acne-scars', 'alopecia-areata', 
           'dry', 'melasma', 'oily', 'vitiligo', 'warts']

def load_model():
    global MODEL
    if MODEL is None:
        try:
            MODEL = torch.load('./skin-model-pokemon.pt', map_location=torch.device('cpu'), weights_only=False)
            MODEL.eval()
            print("✅ Model loaded successfully at startup")
        except Exception as load_error:
            try:
                torch.serialization.add_safe_globals([torchvision.models.efficientnet.EfficientNet])
                MODEL = torch.load('./skin-model-pokemon.pt', map_location=torch.device('cpu'), weights_only=True)
                MODEL.eval()
                print("✅ Model loaded with safe globals")
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                MODEL = None
    return MODEL

# Load model at startup
load_model()

def predict(model, img, tr, classes):
    with torch.no_grad():  # Disable gradient computation for inference
        img_tensor = tr(img)
        output = model(img_tensor.unsqueeze(0))
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get top 3 predictions for better insight
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        # Format results with top predictions
        results = []
        for i in range(3):
            class_name = classes[top_indices[0][i].item()]
            confidence = top_probs[0][i].item() * 100
            results.append(f"{class_name.replace('-', ' ').title()}: {confidence:.1f}%")
        
        # Primary prediction
        primary_class = classes[top_indices[0][0].item()]
        primary_confidence = top_probs[0][0].item() * 100
        
        # Format final result
        if primary_confidence > 60:
            result = f"{primary_class.replace('-', ' ').title()} ({primary_confidence:.1f}%)"
        else:
            result = f"{primary_class.replace('-', ' ').title()} (Low confidence: {primary_confidence:.1f}%)"
            
        # Add note about multiple possibilities if confidence is low
        if primary_confidence < 60:
            result += f" | Also possible: {results[1]} or {results[2]}"
        
        return result

def get_transforms():
    # Try without normalization as the model might have been trained without it
    # Original 512x512 size as in the original project
    transform = [
        T.Resize((512, 512)),  # Original size from the project
        T.ToTensor(),
        # Removing normalization as the model might not expect it
    ]
    return T.Compose(transform)

@app.route('/', methods=['GET', 'POST'])
def home_page():
    res = None
    error = None
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                error = "No file uploaded. Please select an image file."
                return render_template("index.html", res=res, error=error)
            
            f = request.files['file']
            
            # Check if file was actually selected
            if f.filename == '':
                error = "No file selected. Please choose an image file."
                return render_template("index.html", res=res, error=error)
            
            # Check file extension
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
            file_ext = f.filename.rsplit('.', 1)[1].lower() if '.' in f.filename else ''
            if file_ext not in allowed_extensions:
                error = "Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP)."
                return render_template("index.html", res=res, error=error)
            
            # Check if model is loaded
            model = load_model()
            if model is None:
                error = "Model failed to load. Please check the model file."
                return render_template("index.html", res=res, error=error)
            
            filename = secure_filename(f.filename)
            path = os.path.join(app.config['UPLOAD_PATH'], filename)
            f.save(path)
            
            # Process image
            img = Image.open(path).convert("RGB")
            tr = get_transforms()
            res = predict(model, img, tr, CLASSES)
            
            # Clean up uploaded file
            os.remove(path)
            
        except Exception as e:
            error = f"An error occurred while processing the image: {str(e)}"
            res = None
    
    return render_template("index.html", res=res, error=error)
