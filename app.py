import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from flask import Flask,jsonify,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/')
def hello():
    return 'testing!'

@app.route('/Bs',methods=["POST"])
def Bs():
    if request.method=="POST":
        req=request.json
        heloo=req["list"]

        df_test = pd.read_csv("Testing.csv")
        X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]
        X_test.loc[2]=0
        for str in heloo:
            X_test.loc[2, str] = 1
        label_encoder_y = LabelEncoder()
        y_testencoded = label_encoder_y.fit_transform(y_test)

        loaded = pickle.load(open('model.pkl', 'rb'))

        y_train_pred = loaded.predict(X_test.iloc[[2]])
        result={
            "Disease":label_encoder_y.inverse_transform(y_train_pred)[0]
        }
        return jsonify(result)


if __name__=="__main__":
    app.run(debug=True)


