import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
import pickle
import preprocessing


def get_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 20, stratify = y)

    return X_train, X_test, y_train, y_test


def create_pipeline():
    feature_union = FeatureUnion([('pca', PCA(5)), 
                              ('select_best', SelectKBest(k=3))])
    pipeline = Pipeline(steps=[('scaling', StandardScaler()),
                           ('features', feature_union),
                           ('classifier', RandomForestClassifier())])
    
    return pipeline

if __name__ == 'main':
    brain_df = pd.read_csv('../data/full_data.csv')
    X, y = preprocessing.get_preprocessed_data(brain_df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    pipeline = create_pipeline()

    # fit pipeline
    pipeline.fit(X_train, y_train)

    # save model
    with open('../models/pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print('Model pickled in src/models/pipeline.pkl')

