from flask import Flask, request, jsonify, render_template_string
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import time
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the ResNet-50 model
model_path = './resnet50-v2-7.onnx'
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Load full ImageNet labels
import requests
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(IMAGENET_LABELS_URL).json()

def preprocess_image(image):
    # Resize to 224x224, convert to RGB, normalize
    img = image.resize((224, 224)).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # CHW format
    img = img.astype(np.float32) / 255.0  # Normalize
    return img[np.newaxis, :]  # Add batch dimension

# Upload form with JavaScript for preview
UPLOAD_FORM = '''
<!doctype html>
<title>Upload an Image for Prediction</title>
<h1>Upload an Image - Ryzen 9 9950X Server</h1>
<form method=post enctype=multipart/form-data action="/predict">
  <input type="file" name="image" onchange="previewImage(event)">
  <br><br>
  <img id="preview" src="#" alt="Image preview will appear here" style="max-width: 300px; display: none;">
  <br><br>
  <input type="submit" value="Upload and Predict">
</form>

<script>
function previewImage(event) {
    var reader = new FileReader();
    reader.onload = function(){
        var output = document.getElementById('preview');
        output.src = reader.result;
        output.style.display = 'block';
    };
    reader.readAsDataURL(event.target.files[0]);
}
</script>
'''

# Prediction result template
RESULT_PAGE = '''
<!doctype html>
<title>Prediction Result</title>
<h1>Prediction Result - Ryzen 9 9950X Server</h1>
<img src="data:image/jpeg;base64,{{image_data}}" style="max-width:300px;"><br><br>
<p><strong>Predicted Class:</strong> {{predicted_class}}</p>
<p><strong>Confidence:</strong> {{confidence}}</p>
<p><strong>Inference Time:</strong> {{inference_time_ms}} ms</p>
<br>
<a href="/">Upload another image</a>
'''

from flask import render_template_string

@app.route('/', methods=['GET'])
def home():
    return render_template_string(UPLOAD_FORM)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    img_bytes = image_file.read()
    image = Image.open(io.BytesIO(img_bytes))
    input_data = preprocess_image(image)
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_data})[0]
    inference_time = time.time() - start_time
    
    predicted_class_idx = np.argmax(outputs[0])
    predicted_class = labels[predicted_class_idx]
    confidence = float(np.max(outputs[0]))
    
    # Convert image to base64 to embed in HTML
    encoded_img = base64.b64encode(img_bytes).decode('utf-8')
    
    return render_template_string(RESULT_PAGE,
                                  predicted_class=predicted_class,
                                  confidence=f"{confidence:.4f}",
                                  inference_time_ms=f"{inference_time*1000:.2f}",
                                  image_data=encoded_img)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
