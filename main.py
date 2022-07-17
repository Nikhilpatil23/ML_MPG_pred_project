from flask import Flask,request,jsonify
import pickle
from model_files.ml_model import predict_mpg




app = Flask('mpg_prediction')
@app.route('/',methods = ['POST'])
def predict() :
    vehicle = request.get_json()
    print(vehicle)

    model = pickle.load(open('./model_files/model.bin','rb'))

    prediction =  predict_mpg(vehicle,model)
    result = {
        'My Prediction ': list(prediction)
    }
    return jsonify(result)

if __name__ == '__main__' :
    app.run(debug = True, host= '0.0.0.0' , port= 9696)

