import onnxruntime as ort
import numpy as np
import time
from PIL import Image
import os

# Configuration
model_path = './resnet50-v2-7.onnx'  # Path to ResNet-50 model
image_dir = 'resized_images'  # Directory with 224x224 images (or None for synthetic data)
num_inferences = 1000  # Number of inference requests
batch_size = 1  # Single-image requests to simulate real-time use

# Load the model
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Prepare input data
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).transpose(2, 0, 1)  # Convert to CHW format
    img = img.astype(np.float32) / 255.0  # Normalize
    return img[np.newaxis, :]  # Add batch dimension

if image_dir and os.path.exists(image_dir):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    inputs = [load_image(img) for img in image_files[:num_inferences]]
else:
    # Synthetic data: random 224x224 RGB images
    inputs = [np.random.randn(batch_size, 3, 224, 224).astype(np.float32) for _ in range(num_inferences)]

# Warm-up run
session.run(None, {input_name: inputs[0]})

# Measure latency and throughput
latencies = []
start_time = time.time()
for i in range(num_inferences):
    start = time.time()
    session.run(None, {input_name: inputs[i % len(inputs)]})
    end = time.time()
    latencies.append(end - start)
total_time = time.time() - start_time

# Calculate metrics
avg_latency = np.mean(latencies)
throughput = num_inferences / total_time

# Print results
print(f'Average Latency: {avg_latency*1000:.2f} ms per inference')
print(f'Throughput: {throughput:.2f} inferences/second')