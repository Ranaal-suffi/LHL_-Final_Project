import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def uniqueValue(data):
    for column in data.select_dtypes('object'):
        print(column, ':', data[column].unique())



def get_preprocessed_data(_data):
    # brain_df= pd.read_csv('full_data.csv')
    data_with_adult = _data[_data['age'] >= 18]
    data = data_with_adult[data_with_adult['stroke'] == 1]
    data_with_adult_2 = pd.concat([data_with_adult, data])
    data_with_adult_2 = pd.concat([data_with_adult, data])
    data_with_adult_2 = pd.concat([data_with_adult, data])
    data_with_adult_2 = pd.concat([data_with_adult, data])
    data_with_adult_2 = pd.concat([data_with_adult, data])
    data_with_adult_2 = pd.concat([data_with_adult, data])
    data_with_adult_2 = pd.concat([data_with_adult, data])

    lableen=LabelEncoder()
    data_with_adult_2["gender"]=lableen.fit_transform(data_with_adult_2["gender"])
    data_with_adult_2["ever_married"]=lableen.fit_transform(data_with_adult_2["ever_married"])
    data_with_adult_2["work_type"]=lableen.fit_transform(data_with_adult_2["work_type"])
    data_with_adult_2["Residence_type"]=lableen.fit_transform(data_with_adult_2["Residence_type"])
    data_with_adult_2["smoking_status"]=lableen.fit_transform(data_with_adult_2["smoking_status"])

    y = data_with_adult_2['stroke']
    X = data_with_adult_2.drop(columns='stroke')
    

    return X, y