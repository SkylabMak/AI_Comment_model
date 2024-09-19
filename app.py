from flask import Flask, request, jsonify
import joblib
import torch
from transformers import BertTokenizer

app = Flask(__name__)

# Set the device to CUDA (GPU) if available, else use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Load the PyTorch model saved with joblib and move it to the device
model = joblib.load('model/bert_text_classification_model.joblib')
model.to(device)
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load label mappings (assuming they're saved in .joblib format)
label2id = joblib.load('model/label2id.joblib')
id2label = joblib.load('model/id2label.joblib')

def transformerModel(text):
    # Tokenize input text using the same tokenizer used during training
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Move the input tensors to the correct device (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and determine the predicted class
    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=1).item()
    predicted_label = id2label.get(predicted_class_id, 'Unknown')

    return predicted_label

def blstm(text):
    return 9

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
        predicted_label = blstm(text)

    return jsonify({
        'text': text,
        'prediction': predicted_label,
        'model': model_type
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
