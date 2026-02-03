import pandas as pd

data = pd.read_csv('1save.csv')

print(data.info())
print(data.head())

from sklearnex import sklearn_is_patched
sklearn_is_patched()
from sklearn.preprocessing import MinMaxScaler
from tensorflow.Keras.Models import Sequential
from tensorflow.keras.layers import Dense
