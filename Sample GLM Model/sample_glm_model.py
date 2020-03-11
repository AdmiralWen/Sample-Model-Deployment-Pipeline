######################
## Import Libraries ##

import pickle as pk

import pandas as pd
import numpy as np
from math import log

import statsmodels.api as sm
import statsmodels.formula.api as smf


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
    data['log_date_diff'] = data['date_diff'].map(lambda x: log(x + 1) if x > 0 else 0, na_action = 'ignore')

    # Output:
    return data


#####################
## Main Model Call ##

# Define some input objects for this specific model:
glm_fields_list = ['num1', 'num2', 'num5', 'binary1', 'binary2', 'log_date_diff', 'cat1', 'cat3_1D', 'dummy_rating_cat']

def callModel(input_record, model_obj_path):
    ''' Main processing function. Calls the model given in model_obj_path and produces a prediction. '''

    # Load input record into pandas:
    data = pd.DataFrame(input_record)

    # Apply feature engineering & transformation steps:
    data = _featuresTransform(data)

    # Drop unnecessary fields and reorder columns:
    data_glm = data[glm_fields_list]

    # Load XGB Model and make prediction:
    data_glm = sm.add_constant(data_glm)
    with open(model_obj_path, 'rb') as pickle_file:
        glm_model = pk.load(pickle_file)
    glm_pred = glm_model.predict(data_glm)

    # Output prediction:
    return glm_pred.values[0]