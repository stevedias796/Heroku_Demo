import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
file = open('model.pkl', 'rb')
model = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['get', 'post'])
def result():
    if request.method == 'POST':
        '''CRIM = float(request.form.get('r1'))
        ZN = float(request.form.get('r2'))
        INDUS = float(request.form.get('r3'))
        CHAS = float(request.form.get('r4'))
        NOX = float(request.form.get('r5'))
        RM = float(request.form.get('r6'))
        AGE = float(request.form.get('r7'))
        DIS = float(request.form.get('r8'))
        RAD = float(request.form.get('r9'))
        TAX = float(request.form.get('r10'))
        PTRATIO = float(request.form.get('r11'))
        BR = float(request.form.get('r12'))
        LSTAT = float(request.form.get('r13'))
        #MEDV = request.form.get('r14')
        features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, BR, LSTAT]])'''
        input_features = [int(x) for x in request.form.values()]
        features = [np.array(input_features)]
        result = model.predict(features)
        #output = round(result, 2)
    return render_template('home.html', param='Predicted Salary is Rs. {}'.format(result[0]))


if __name__ == '__main__':
    app.run(debug=True)
