from flask import Flask, render_template, request
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from text_preprocessing import preprocessor

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def predict():
        if request.method == 'POST':
            text = request.form.get("review_text")
            # Preprocess the text using the preprocessor function
            processed_text = preprocessor(text)
            
            model = joblib.load("Model/Sentiment Analysis.pkl")
            prediction = model.predict([processed_text])
            if (processed_text != '') and (len(processed_text)>2):
                for i in prediction:
                    if i ==0:
                        prediction='Negative'
                    elif i==1:
                        prediction='Positive'
            elif (processed_text != '') and (len(processed_text)<3):
                prediction='Write a proper review'
            else:
                prediction='Please write your review'
            return render_template('home.html', prediction=prediction, processed_text = processed_text)
        else:
            return render_template('home.html')



if __name__ == "__main__":
    app.run(debug=True, host = "0.0.0.0")