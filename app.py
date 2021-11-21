import pickle
import numpy as np
from flask import Flask, request, render_template

model = pickle.load(open("diamond_prediction.pkl", "rb"))
label = pickle.load(open("Label_enc.pkl", "rb"))
scaler = pickle.load(open("standard_scaler.pkl", "rb"))

app = Flask(__name__,template_folder='template')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods = ['POST'])
def predict():
    # carat = request.form['carat']
    # cut = request.form['cut']
    # color = request.form['color']
    # clarity = request.form['clarity']
    # depth = request.form['depth']
    # table = request.form['table']
    carat = request.form.get('carat')
    cut = request.form.get('cut')
    color = request.form.get('color')
    clarity = request.form.get('clarity')
    depth = request.form.get('depth')
    table = request.form.get('table')
    

    cut = label['cut'].transform([cut])
    color = label['color'].transform([color])
    clarity = label['clarity'].transform([clarity])
    list_to_predict = np.array([carat,cut,color,clarity,depth,table])
    list_to_predict = scaler.transform([list_to_predict])
    result = model.predict(list_to_predict)
    # print(result)
    # return render_template('index.html')
    return render_template('index.html', prediction_text='The price should be $ {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)

    
    