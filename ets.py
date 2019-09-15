from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pandas import read_csv

data = read_csv('us-weather-sample.csv')
data = data['TMAX']

# Single Alpha
model = SimpleExpSmoothing(data)
model_fit = model.fit()

print(model_fit.summary())

# Double or triple Expotenial Smoothing
from statsmodels.tsa.holtwinters import ExpotenialSmoothing

# prepare data
data = data['TMAX']
# create class
model = ExponentialSmoothing(data, ...)
# fit model
model_fit = model.fit(...)
# make prediction
yhat = model_fit.predict(...)

# https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/
