######################
## Import Libraries ##

import pickle as pk

import pandas as pd
import numpy as np
from math import log

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

import xgboost as xgb


######################
## Helper Functions ##

def _featuresTransform(input_dataframe):
    ''' Applies some necessary variable transformations to the original dataset. '''

    # Replace any blanks with NaN values:
    data = input_dataframe.replace(r'^\s*$', np.nan, regex = True)

    # Create date-length variable:
    data['date1'] = pd.to_datetime(data['date1'])
    data['date2'] = pd.to_datetime(data['date2'])
    data['date_diff'] = (data['date2'] - data['date1']).map(lambda x: x.days)

    # Create new cat3_1D variable (simulates SIC-1D):
    data['cat3_1D'] = data['cat3'].map(lambda x: str(x)[0], na_action = 'ignore')

    # Handle dummy_rating:
    data['dummy_rating_cat'] = np.where(data['dummy_rating'].str.contains('A'), 1, -1)
    data['dummy_rating_cat'] = np.where(pd.isnull(data['dummy_rating']), np.nan, data['dummy_rating_cat'])

    # Handle num1 and num5:
    data['num1'] = data['num1'] - 1900
    data['num5'] = data['num5'].map(lambda x: log(x + 1) if x > 0 else 0, na_action = 'ignore')

    # Output:
    return data

def _oheHotEncode(input_dataframe, feature_to_ohe_path_dict):
    '''
        One-hot-encodes selected categorical features. A dictionary must be provided for the features to be encoded,
        and their corresponding OHE object paths.
    '''

    data = input_dataframe

    for f, p in feature_to_ohe_path_dict.items():
        # Load saved object and OHE:
        feature = data[f].map(str, na_action = 'ignore').fillna('null')

        # Applies saved OHE transformations:
        with open(p, 'rb') as pickle_file:
            onehot_encoder = pk.load(pickle_file)
        f_enc = onehot_encoder.transform(feature.values.reshape(len(data), 1))

        # Append new OHE features and drop old:
        f_col_names = onehot_encoder.get_feature_names([f])
        data = pd.concat([data.drop(f, axis = 1).reset_index(drop = True), pd.DataFrame(f_enc, columns = f_col_names)], axis = 1)

    # Output:
    return data

def _pcaTransform(input_dataframe, pca_fields, missing_imputer_obj_path, variable_scaler_obj_path, pca_obj_path, pca_output_prefix = 'pca_out'):
    ''' Applies PCA transformations to selected fields. Object paths for value imputation, variable scaling, and PCA are required inputs. '''

    # Isolate PCA fields:
    pca_cols = input_dataframe[pca_fields]

    # Missing values imputation:
    with open(missing_imputer_obj_path, 'rb') as pickle_file:
        missing_imputer = pk.load(pickle_file)
    pca_cols = missing_imputer.transform(pca_cols)

    # Variable scaling:
    with open(variable_scaler_obj_path, 'rb') as pickle_file:
        variable_scaler = pk.load(pickle_file)
    pca_cols_std = variable_scaler.transform(pca_cols)

    # PCA features:
    with open(pca_obj_path, 'rb') as pickle_file:
        pca = pk.load(pickle_file)
    pca_cols = pca.transform(pca_cols_std)

    # Combine with overall dataset:
    pca_cols_df = pd.DataFrame(data = pca_cols, columns = [pca_output_prefix + str(i) for i in range(pca.n_components)])
    data_dropped = input_dataframe.drop(pca_fields, axis = 1).reset_index(drop = True)
    data_w_pca = pd.concat([data_dropped, pca_cols_df], axis = 1)

    # Output:
    return data_w_pca


#####################
## Main Model Call ##

# Define some input objects for this specific model:
cat_features_to_ohe_obj_paths = {'cat1':'ohe_cat1.pkl', 'cat2':'ohe_cat2.pkl', 'cat3_1D':'ohe_cat3_1D.pkl'}
pca_fields_list = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10']
xgb_column_order = col_order = ['num1', 'num2', 'num3', 'num4', 'num5', 'binary1', 'binary2', 'binary3', 'date_diff', 'dummy_rating_cat', 'cat1_I', 'cat1_N',
                                'cat1_P', 'cat1_R', 'cat1_null', 'cat2_AMP', 'cat2_ECR', 'cat2_EP', 'cat2_FL', 'cat2_FLD', 'cat2_MFP', 'cat2_MN', 'cat2_N1C',
                                'cat2_PE', 'cat2_PP', 'cat3_1D_1', 'cat3_1D_2', 'cat3_1D_3', 'cat3_1D_4', 'cat3_1D_5', 'cat3_1D_6', 'cat3_1D_7', 'cat3_1D_8',
                                'cat3_1D_9', 'cat3_1D_null', 'pca_out0', 'pca_out1', 'pca_out2', 'pca_out3']

def callModel(input_record, model_obj_path):
    ''' Main processing function. Calls the model given in model_obj_path and produces a prediction. '''

    # Load input record into pandas:
    data = pd.DataFrame(input_record)

    # Apply feature engineering & transformation steps:
    data = _featuresTransform(data)
    data = _oheHotEncode(data, cat_features_to_ohe_obj_paths)
    data = _pcaTransform(data, pca_fields_list, 'missing_imputer.pkl', 'variable_scaler.pkl', 'pca.pkl', pca_output_prefix = 'pca_out')

    # Drop unnecessary fields and reorder columns:
    data.drop(['row_id', 'dummy_name', 'date1', 'date2', 'cat3', 'dummy_rating'], axis = 1, inplace = True)
    data = data[xgb_column_order]

    # Load XGB Model and make prediction:
    xgb_model = xgb.Booster({'nthread':8})
    xgb_model.load_model(model_obj_path)
    rec_dm = xgb.DMatrix(data.values, feature_names = data.columns)
    xgb_pred = xgb_model.predict(rec_dm)

    # Output prediction:
    return xgb_pred[0]