import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
file = open('model.pkl', 'rb')
model = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def result():
    input_features = [int(x) for x in request.form.values()]
    features = [np.array(input_features)]
    result = model.predict(features)
    output = round(result[0], 2)
    return render_template('home.html', param='Predicted Salary is Rs. {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
