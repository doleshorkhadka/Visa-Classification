from dataclasses import dataclass
from flask import Flask, jsonify,render_template,request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

app = Flask(__name__,static_url_path='',template_folder='web/templates',static_folder='web/static')
lr_model = pickle.load(open('Logistic_reg.pkl','rb'))
rf_model = pickle.load(open('Random_forrest.pkl','rb'))

@app.route("/")
def home():
    return render_template('base.html')

@app.route("/predict",methods=['POST'])
def predict():
    features = np.array([x for x in request.form.values()])
    for i,x in enumerate(features):
        if x == '' or x=='nan' or x=='NaN':
            features[i] = np.nan
        elif x == 'Y' or x == 'Yes':
            features[i] = 1
        elif x == 'N' or x == 'No':
            features[i] = 0
    columns = ['NEW_EMPLOYMENT','NEW_CONCURRENT_EMPLOYMENT','CHANGE_PREVIOUS_EMPLOYMENT', 'CONTINUED_EMPLOYMENT',               
        'AMENDED_PETITION','CHANGE_EMPLOYER', 'TOTAL_WORKER_POSITIONS','WAGE_UNIT_OF_PAY_1',
        'FULL_TIME_POSITION','SECONDARY_ENTITY_1','AGENT_REPRESENTING_EMPLOYER',
       'H-1B_DEPENDENT', 'WILLFUL_VIOLATOR', 'WAGE_RATE_OF_PAY_FROM_1',
       'WAGE_RATE_OF_PAY_TO_1', 'PREVAILING_WAGE_1']
    input_df = pd.DataFrame(features.reshape(1,16),
                            columns=columns,index=[335210])
    data = pd.read_csv('Modelled_data.csv')
    data.drop('CASE_STATUS',axis=1,inplace=True)
    data = pd.concat([data,input_df],axis=0)
    data['WAGE_UNIT_OF_PAY_1'] = LabelEncoder().fit_transform(data['WAGE_UNIT_OF_PAY_1'])
    data = pd.DataFrame( SimpleImputer().fit_transform(data),columns=data.columns)
    data = pd.DataFrame(MinMaxScaler(feature_range=(0,1)).fit_transform(data),columns=data.columns)
    data = data.loc[335210]
    data = pd.DataFrame(data.values.reshape(1,16),columns=data.index)
    output = lr_model.predict(data)
    prob = lr_model.predict_proba(data)
    print(output)
    if output[0] == 1:
        output = 'Certified !'
    else:
        output = 'Denied or Rejected !'
    return render_template('base.html',
                            prediction="The status of the Visa is : '{}'".format(output),
                            prediction_prob='The prediction probability is [Denied,Certified] ~ [{:.2f}%,{:.2f}%]'.format(prob[0,0]*100,prob[0,1]*100)
                            )


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    input_data = pd.read_json(data)
    data = pd.read_csv('Modelled_data.csv')
    data.drop('CASE_STATUS',axis=1,inplace=True)
    data = pd.concat([data,input_data],axis=0)
    data['WAGE_UNIT_OF_PAY_1'] = LabelEncoder().fit_transform(data['WAGE_UNIT_OF_PAY_1'])
    data = pd.DataFrame( SimpleImputer().fit_transform(data),columns=data.columns)
    data = pd.DataFrame(MinMaxScaler(feature_range=(0,1)).fit_transform(data),columns=data.columns)
    data = data.loc[335210]
    data = pd.DataFrame(data.values.reshape(1,16),columns=data.index)
    prediction = lr_model.predict(input_data)
    prob = lr_model.predict_proba(input_data)
    return jsonify(result=int(prediction[0]),
                    prediction_prob=[float(prob[0,0]),float(prob[0,1])]
                    )
    # Another way
    # output = {'result':int(prediction[0]}
    # return jsonify(output)
    