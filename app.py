from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Set the device to CPU because Render is a CPU-only environment
device = torch.device('cpu')

# Load the model's architecture and state dict from the .pth file
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)  # Ensure num_labels matches your training
model.load_state_dict(torch.load('model/bert_text_classification_model.pth', map_location=device))

# Move the model to the CPU and set it to evaluation mode
model.to(device)
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load label mappings (assuming they are saved using torch.save as .pth)
label2id = torch.load('model/label2id.pth')
id2label = torch.load('model/id2label.pth')

def transformerModel(text):
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

def blstm(text):
    return -1

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
