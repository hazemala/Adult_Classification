# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransfomer(BaseEstimator, TransformerMixin):
    def __init__(self,col):
        self.col=col

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        assert self.n_features_in_ == X.shape[1]
        return np.log1p(X)
    
    def get_feature_names_out(self,X,y=None):
        return self.col

class LogTransfomer_0(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        assert self.n_features_in_ == X.shape[1]
        return np.sign(X) * np.log1p(np.abs(X))

    def get_feature_names_out(self, input_features=None):
        return self.col

class Handle_Ub_Lb(BaseEstimator, TransformerMixin):
    def __init__(self,col):
        self.col = col
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        q1 = np.percentile(X, 25)
        q2 = np.percentile(X, 50)
        q3 = np.percentile(X, 75)
        iqr = q3 - q1
        ub_train = q3 + 1.5 * iqr
        lb_train = q1 - 1.5 * iqr
        self.ub_train = ub_train
        self.lb_train = lb_train
        return self # must return it's self

    def transform(self, X, y=None):
        assert self.n_features_in_ == X.shape[1]
        X[X > self.ub_train] = self.ub_train
        X[X < self.lb_train] = self.lb_train
        return X
    def get_feature_names_out(self,X,y=None):
        return self.col


app = Flask(__name__)
model = joblib.load("model.pkl")  

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    df=data['data']

    formated_data = []
    
    for row in data['data']:
        correct_data_type = []
        for index in row:
            try:
                correct_data_type.append(float(index))
            except:
                correct_data_type.append(index)
        formated_data.append(correct_data_type)

    df = pd.DataFrame(formated_data,columns=['age', 'workclass', 'fnlwgt', 'education', 'educational_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'])
    df = df.replace('?',np.nan)

    df.drop(df[df['native_country']=='Holand-Netherlands'].index,axis=0,inplace=True)

    def age_group(age):
        if 17 <= age <=24:
            return 'Youth'
        elif 25 <= age <=34  :
            return 'young_adult'
        elif 35 <= age <= 44 :
            return 'early_middle_age'
        elif 45<= age <= 54 :
            return 'middle_age'
        elif 55<= age <= 64 :
            return 'late_middle_age'
        elif 65<= age <= 74 :
            return 'senior'
        elif 75<=age<=90:
            return 'elderly'
        else :
            return 'unknown'
    df['age_group'] = df['age'].apply(age_group)
    def edu_level(edu_level):
        if edu_level in ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th']:
            return 'low_education'
        elif edu_level in['12th', 'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc']:
            return'intermediate_education'
        elif edu_level in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school']:
            return 'higher_education'
        else :
            return 'Unknown'
    df['education_level'] = df['education'].apply(edu_level)
    def matrital_status_classification(status):
        if status  == 'Never-married':
            return 'single'
        elif status in ['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse']:
            return 'married'
        elif status in ['Divorced','Separated']:
            return 'divorced/separated'
        elif status =='Widowed':
            return 'widowed'

    df['marital_status_calss']=df['marital_status'].apply(matrital_status_classification)
    def jop_classification(job):
        if job in ['Adm-clerical','Exec-managerial','Sales']:
            return 'office jobs'
        elif job in ['Prof-specialty', 'Tech-support']:
            return 'prof_educa_jobs'
        elif job in ['Craft-repair', 'Handlers-cleaners', 'Machine-op-inspct']:
            return 'workers_&_craftsmen'
        elif job in ['Protective-serv', 'Other-service', 'Priv-house-serv']:
            return 'security_&_services'
        elif job in ['Farming-fishing', 'Transport-moving']:
            return 'transport_&_agricultre'
        elif job == 'Armed-Forces':
            return 'military'
        else :
            return np.nan
    df['job_classifition']=df['occupation'].apply(jop_classification)
    def group_countries_by_continent(countrie):
        if countrie in ['United-States', 'Canada', 'Mexico', 'Cuba', 'Jamaica', 'Dominican-Republic',
                        'Puerto-Rico', 'Haiti', 'Honduras', 'Nicaragua', 'Guatemala', 
                        'Outlying-US(Guam-USVI-etc)', 'Trinadad&Tobago']:
            return 'north_america'
        elif countrie in ['Peru', 'Ecuador', 'Columbia']:
            return 'south_america'
        elif countrie in ['England', 'Germany', 'Ireland', 'France', 'Italy', 'Portugal',
                'Poland', 'Greece', 'Scotland', 'Hungary', 'Yugoslavia']:
            return 'europe'
        elif countrie in ['India', 'Iran', 'China', 'Japan', 'Taiwan', 'Vietnam',
            'Cambodia', 'Thailand', 'Laos', 'Philippines', 'Hong']:
            return 'asia'
        else :
            return np.nan
    df['continent']=df['native_country'].apply(group_countries_by_continent)
    def classify_work_hours(hour):
        if hour < 20 :
            return 'very_low_hours'
        elif hour < 35 :
            return 'part_time'
        elif hour < 50:
            return 'full_time'
        else :
            return 'over_time'
    df['work_hours_class'] = df['hours_per_week'].apply(classify_work_hours)
    df['is_young_single'] = ((df['age_group'] == 'young_adult') & (df['marital_status_calss'] == 'single')).astype(int)
    df['is_high_edu_fulltime'] = ((df['education_level'] == 'higher_education') & (df['work_hours_class'] == 'full_time')).astype(int)

    df['capital_net'] = df['capital_gain'] - df['capital_loss']

    print(df)

    predections = model.predict(df)

    return jsonify({"predictions": ['more-50K' if i == 1 else 'less-equal-50K' for i in predections]})

if __name__ == "__main__":
    app.run(debug=True)
