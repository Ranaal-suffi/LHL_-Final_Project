import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score
import preprocessing


def get_model():
    with open('pipeline.pkl', 'rb') as f:
        return pickle.load(f)
    
if __name__ == 'main':
    model = get_model()
    test_data = pd.read_csv('../data/full_filled_stroke_data (1).csv')
    test_X, test_Y = preprocessing.get_preprocessed_data(test_data)

    y_pred = model.predict(test_X)
    y_true = test_data['stroke']

    cross_val_scores = cross_val_score(model, test_X, test_Y , cv = 5)

    print('Cross validation scores: ', cross_val_scores)
    print('Model F1 score: {0:0.2f}'.format(f1_score(y_true, y_pred)))
    print('Model accuracy score with: {0:0.2f}'.format(accuracy_score(y_true, y_pred)))
    print('Model precision score with: {0:0.2f}'.format(precision_score(y_true, y_pred, average='macro')))
    print('Model roc_auc_score with: {0:0.2f}'.format(roc_auc_score(y_true, y_pred)))
    print('Model recall: {0:0.2f}'.format(recall_score(y_true, y_pred)))