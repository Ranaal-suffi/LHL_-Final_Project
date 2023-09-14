from flask import Flask, render_template, request, jsonify
from flask_restful import Resource, Api, request
import pickle

app = Flask(__name__, static_url_path ='/static')
api = Api(app)


model = pickle.load(open('../models/pipeline.pkl', 'rb'))



@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    gender = request.form['gen']
    age = float(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type=int(request.form['work_type'])
    Residence_type= int(request.form['Residence_type'])
    avg_glucose_level= float(request.form['avg_glucose_level'])
    bmi=float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])



    pred = model.predict([[gender,age,hypertension,heart_disease,ever_married, work_type,Residence_type,
                                avg_glucose_level, bmi, smoking_status]])
 

    print(pred)
    print(gender,age,hypertension,heart_disease,ever_married, work_type,Residence_type,
                                avg_glucose_level, bmi, smoking_status)
    

    return render_template('index.html', results = pred)
# api.add_resource(predict, '/predict',)

if __name__ == "__main__":
    app.run(debug = True)
