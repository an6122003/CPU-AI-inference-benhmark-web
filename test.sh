sudo apt update && sudo apt install git
pip install onnxruntime
wget https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx
pip3 install --ignore-installed --break-system-packages Flask

## Deploy the model
pip3 install gunicorn --break-system-packages
pip install flask-cors --break-system-packages
gunicorn --bind 0.0.0.0:5000 app:app
# Run in background and save output to nohup.out (nohangup when terminal is closed)
nohup gunicorn --bind 0.0.0.0:5000 app:app &

## other models
wget https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet152-v1-7.onnx
