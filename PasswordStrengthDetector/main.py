from flask import Flask, render_template, request
import joblib
import requests
import xgboost as xgb
import numpy as np
import sklearn
app = Flask(__name__)
model=xgb.XGBClassifier()
model.load_model("xgb_classifier.json")
#Function to seperate char in password
import __main__
def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

__main__.word_divide_char=word_divide_char

#model = pickle.load(open('xgb_pwd_model.pkl', 'rb'))
#vectorizer= pickle.load(open('vectorizer.pkl', 'rb'))
vectorizer=joblib.load("vectorizer.sav")

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        pwd = request.form['pwd']
        X_predict=[pwd]
        X_predict=vectorizer.transform(X_predict)
        prediction=model.predict(X_predict)
        if prediction==0:
            prediction_text='Weak Password'
        elif prediction==1:
            prediction_text='Medium Strength Password'
        else:
            prediction_text='Strong Password'
        return render_template('index.html',prediction_text=prediction_text)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)

