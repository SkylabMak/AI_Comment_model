from flask import Flask, request, jsonify
import joblib
import torch
from transformers import BertTokenizer

from model2 import transformerModel02

app = Flask(__name__)

# Set the device to CPU because Render is a CPU-only environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the PyTorch model saved with joblib and map it to CPU
model = joblib.load('model/bert_text_classification_model.joblib')

# Ensure the model is loaded on the CPU
model.to(device)
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load label mappings (assuming they're saved in .joblib format)
label2id = joblib.load('model/label2id.joblib')
id2label = joblib.load('model/id2label.joblib')

def transformerModel(text):
    print("model 1 run")
    # Tokenize input text using the same tokenizer used during training
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Move the input tensors to the correct device (CPU in this case)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and determine the predicted class
    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=1).item()
    predicted_label = id2label.get(predicted_class_id, 'Unknown')

    return predicted_label

def transformerModelVersion2(text):
    print("model 1 run")
    predicted_label = transformerModel02(text)
    return predicted_label

# API route to predict based on input text
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    model_type = data.get('model')

    if not text or model_type is None:
        return jsonify({"error": "No text or model type provided"}), 400

    # Based on the model type, choose the corresponding model
    if model_type == 1:
        predicted_label = transformerModel(text)
    else:
        predicted_label = transformerModelVersion2(text)

    return jsonify({
        'text': text,
        'prediction': predicted_label,
        'model': model_type
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
