#import
import shap
import shapLib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
import xgboost as xgb
from sklearn.model_selection import train_test_split


#import dataset
adult = fetch_ucirepo(id=2)
print(type(adult.data))
#Feature Engineering
X = adult.data.features
features = X.columns
y = adult.data.targets
data = pd.concat([X,y],axis=1)
print(data)
print(y.head())

def mappingData(varName):
    unique_categories = data[varName].unique()
    s = {category: index for index, category in enumerate(unique_categories)}
    data.loc[:, varName] = data[varName].map(s)
varNames = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
for varName in varNames: mappingData(varName)
data = data.astype('int')

X_train, X_test, y_train, y_test = train_test_split(data[features], ## predictors only
                                                    data['income'],
                                                    test_size=0.30, 
                                                    random_state=0)
print(X_train)

#Modelling
#Train model
model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)
print('model ok')

#Get predictions
y_pred = model.predict(X_train)

'''
#Model evaluation
plt.figure(figsize=(5,5))
plt.scatter(y,y_pred)
plt.plot([0,30],[0,30],color='r',linestyle='-',linewidth=2)
plt.ylabel('Predicted',size=20)
plt.xlabel('Actual',size=20)
plt.show()
'''

#Get shap values
explainer = shap.Explainer(model)
shap_values = explainer(X_train)
#shap_value = explainer(X[0:100])

print(np.shape(shap_values.values))

#Waterfall plot
#Waterfall plot for first observation
shap.plots.waterfall(shap_values[0])

shap.initjs()

## kernel shap sends data as numpy array which has no column names, so we fix it
def xgb_predict(data_asarray):
    data_asframe =  pd.DataFrame(data_asarray, columns=features)
    return model.predict(data_asframe)
#Kernel SHAP
X_summary = shap.kmeans(X_train, 10)
shap_kernel_explainer = shap.KernelExplainer(xgb_predict, X_summary)
## shapely values with kernel SHAP
shap_Kernel_values = shap_kernel_explainer.shap_values(X=X_test, nsamples=50)
shap.plots._waterfall.waterfall_legacy(shap_kernel_explainer.expected_value, shap_Kernel_values[0], feature_names=X_train.columns)

#shap.waterfall_plot(shap.Explanation(values=shap_Kernel_values[1][0],base_values=shap_kernel_explainer.expected_value[1], data=X_test[0],feature_names=features))
#shap.force_plot(shap_kernel_explainer.expected_value, shap_values_single, X_test.iloc[[5]])


