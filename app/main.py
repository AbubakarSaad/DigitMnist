from flask import Flask, request, jsonify
from app.torch_utils import transform_image, get_predication

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if (request.method == 'POST'):
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})

        if not allowed_file(file.filename):
            return jsonify({'error': 'not supported'})

        try: 
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_predication(tensor)

            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}

            return jsonify(data)

        except Exception as e:
            print(e)
            return jsonify({'error': 'Error during Predictions'})
    # 1. Load the image 

    # 2. image -> tensor
    # 3. predication
    # 4. return json
    return jsonify({'result': 1})


