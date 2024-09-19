from flask import Flask, request, jsonify
import joblib
import torch
from transformers import BertTokenizer

app = Flask(__name__)

# Load the PyTorch model saved with joblib
model = joblib.load('model/bert_text_classification_model.joblib')
model.eval()  # Set the model to evaluation mode

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load label mappings (assuming they're saved in .joblib format)
label2id = joblib.load('model/label2id.joblib')
id2label = joblib.load('model/id2label.joblib')

def transformerModel(text):
    # Tokenize input text using the same tokenizer used during training
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    # Move the input tensors to the same device as the model (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and determine the predicted class
    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=1).item()
    predicted_label = id2label.get(predicted_class_id, 'Unknown')

    # Return the prediction as a JSON response
    return predicted_label

def blstm(text):
    return 9
# API route to predict based on input text
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    model = data.get('model')

    if not text or not model:
        return jsonify({"error": "No text provided"}), 400
    predicted_label = -1
    if(model == 1):
        predicted_label = transformerModel(text)
    else:
        predicted_label = blstm(text)

    # Return the prediction as a JSON response
    return jsonify({
        'text': text,
        'prediction': predicted_label,
        'model':model
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
