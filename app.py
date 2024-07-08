import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

model = joblib.load('model.joblib')
minmax = MinMaxScaler(feature_range=(0,1))

def predict(age, gender, height):
    data = [[age, gender, height]]
    data = minmax.fit_transform(data)
    prediction = model.predict(data)[0]
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_status():
    age = float(request.form['age'])
    gender = int(request.form['gender'])
    height = float(request.form['height'])
    result = predict(age, gender, height)
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
