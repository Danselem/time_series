import json
import os
import pprint

import numpy as np
import matplotlib.pyplot as plt

from merlion.utils.time_series import TimeSeries
from ts_datasets.forecast import M4

# Import models & configs
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.models.forecast.smoother import MSES, MSESConfig

# Import data pre-processing transforms
from merlion.transform.base import Identity
from merlion.transform.resample import TemporalResample

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.ensemble.combine import Mean, ModelSelector
from merlion.models.ensemble.forecast import ForecasterEnsemble, ForecasterEnsembleConfig

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.factory import ModelFactory

# Load the time series
# time_series is a time-indexed pandas.DataFrame
# trainval is a time-indexed pandas.Series indicating whether each timestamp is for training or testing
time_series, metadata = M4(subset="Hourly")[5]
trainval = metadata["trainval"]

# Is there any missing data?
timedeltas = np.diff(time_series.index)
print(f"Has missing data: {any(timedeltas != timedeltas[0])}")

# Visualize the time series and draw a dotted line to indicate the train/test split
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.plot(time_series)
ax.set_xlabel(r"$Time$ (days)", fontsize=16)
ax.set_ylabel(r"$H6$", fontsize=16)
ax.axvline(time_series[trainval].index[-1], ls="--", lw="2", c="k")
plt.show()

# Split the time series into train/test splits, and convert it to Merlion format
train_data = TimeSeries.from_pd(time_series[trainval])
test_data  = TimeSeries.from_pd(time_series[~trainval])
print(f"{len(train_data)} points in train split, "
      f"{len(test_data)} points in test split.")



# All models are initialized using the syntax ModelClass(config),
# where config is a model-specific configuration object. This is where
# you specify any algorithm-specific hyperparameters, as well as any
# data pre-processing transforms.

# ARIMA assumes that input data is sampled at a regular interval,
# so we set its transform to resample at that interval. We must also specify
# a maximum prediction horizon.
config1 = ArimaConfig(max_forecast_steps=100, order=(20, 1, 5),
                      transform=TemporalResample(granularity="1h"))
model1  = Arima(config1)


# Prophet has no real assumptions on the input data (and doesn't require
# a maximum prediction horizon), so we skip data pre-processing by using
# the Identity transform.
config2 = ProphetConfig(max_forecast_steps=None, transform=Identity())
model2  = Prophet(config2)


# MSES assumes that the input data is sampled at a regular interval,
# and requires us to specify a maximum prediction horizon. We will
# also specify its look-back hyperparameter to be 60 here
config3 = MSESConfig(max_forecast_steps=100, max_backstep=60,
                     transform=TemporalResample(granularity="1h"))
model3  = MSES(config3)



# The ForecasterEnsemble is a forecaster, and we treat it as a first-class model.
# Its config takes a combiner object, specifying how you want to combine the 
# predictions of individual models in the ensemble. There are two ways to specify
# the actual models in the ensemble, which we cover below.

# The first way to specify the models in the ensemble is to provide them when
# initializing the ForecasterEnsembleConfig.
#
# The combiner here will simply take the mean prediction of the ensembles here
ensemble_config = ForecasterEnsembleConfig(
    combiner=Mean(), models=[model1, model2, model3])
ensemble = ForecasterEnsemble(config=ensemble_config)


# Alternatively, you can directly specify the models when initializing the
# ForecasterEnsemble itself.
#
# The combiner here uses the sMAPE to compare individual models, and
# selects the model with the lowest sMAPE
selector_config = ForecasterEnsembleConfig(
    combiner=ModelSelector(metric=ForecastMetric.sMAPE))
selector = ForecasterEnsemble(
    config=selector_config, models=[model1, model2, model3])

# Train the models
print(f"Training {type(model1).__name__}...")
forecast1, stderr1 = model1.train(train_data)

print(f"\nTraining {type(model2).__name__}...")
forecast2, stderr2 = model2.train(train_data)

print(f"\nTraining {type(model3).__name__}...")
forecast3, stderr3 = model3.train(train_data)

print("\nTraining ensemble...")
forecast_e, stderr_e = ensemble.train(train_data)

print("\nTraining model selector...")
forecast_s, stderr_s = selector.train(train_data)

print("Done!")

# Inference

# Truncate the test data to ensure that we are within each model's maximum
# forecast horizon.
sub_test_data = test_data[:50]

# Obtain the time stamps corresponding to the test data
time_stamps = sub_test_data.univariates[sub_test_data.names[0]].time_stamps

# Get the forecast & standard error of each model. These are both
# merlion.utils.TimeSeries objects. Note that the standard error is None for
# models which don't support uncertainty estimation (like MSES and all
# ensembles).
forecast1, stderr1 = model1.forecast(time_stamps=time_stamps)
forecast2, stderr2 = model2.forecast(time_stamps=time_stamps)

# You may optionally specify a time series prefix as context. If one isn't
# specified, the prefix is assumed to be the training data. Here, we just make
# this dependence explicit. More generally, this feature is useful if you want
# to use a pre-trained model to make predictions on data further in the future
# from the last time it was trained.
forecast3, stderr3 = model3.forecast(time_stamps=time_stamps, time_series_prev=train_data)

# The same options are available for ensembles as well, though the stderr is None
forecast_e, stderr_e = ensemble.forecast(time_stamps=time_stamps)
forecast_s, stderr_s = selector.forecast(time_stamps=time_stamps, time_series_prev=train_data)


# We begin by computing the sMAPE of ARIMA's forecast (scale is 0 to 100)
smape1 = ForecastMetric.sMAPE.value(ground_truth=sub_test_data,
                                    predict=forecast1)
print(f"{type(model1).__name__} sMAPE is {smape1:.3f}")

# Next, we can visualize the actual forecast, and understand why it
# attains this particular sMAPE. Since ARIMA supports uncertainty
# estimation, we plot its error bars too.
fig, ax = model1.plot_forecast(time_series=sub_test_data,
                               plot_forecast_uncertainty=True)
plt.show()

# We begin by computing the sMAPE of Prophet's forecast (scale is 0 to 100)
smape2 = ForecastMetric.sMAPE.value(sub_test_data, forecast2)
print(f"{type(model2).__name__} sMAPE is {smape2:.3f}")

# Next, we can visualize the actual forecast, and understand why it
# attains this particular sMAPE. Since Prophet supports uncertainty
# estimation, we plot its error bars too.
# Note that we can specify time_series_prev here as well, though it
# will not be visualized unless we also supply the keyword argument
# plot_time_series_prev=True.
fig, ax = model2.plot_forecast(time_series=sub_test_data,
                               time_series_prev=train_data,
                               plot_forecast_uncertainty=True)
plt.show()

# We begin by computing the sMAPE of MSES's forecast (scale is 0 to 100)
smape3 = ForecastMetric.sMAPE.value(sub_test_data, forecast3)
print(f"{type(model3).__name__} sMAPE is {smape3:.3f}")

# Next, we visualize the actual forecast, and understand why it 
# attains this particular sMAPE.
fig, ax = model3.plot_forecast(time_series=sub_test_data,
                               plot_forecast_uncertainty=True)
plt.show()

# Compute the sMAPE of the ensemble's forecast (scale is 0 to 100)
smape_e = ForecastMetric.sMAPE.value(sub_test_data, forecast_e)
print(f"Ensemble sMAPE is {smape_e:.3f}")

# Visualize the forecast.
fig, ax = ensemble.plot_forecast(time_series=sub_test_data,
                                 plot_forecast_uncertainty=True)
plt.show()

# Compute the sMAPE of the selector's forecast (scale is 0 to 100)
smape_s = ForecastMetric.sMAPE.value(sub_test_data, forecast_s)
print(f"Selector sMAPE is {smape_s:.3f}")

# Visualize the forecast.
fig, ax = selector.plot_forecast(time_series=sub_test_data,
                                 plot_forecast_uncertainty=True)
plt.show()



# Save the model
os.makedirs("../../models/forecasting/", exist_ok=True)
path = os.path.join("../../models/forecasting", "prophet")
model2.save(path)

# Print the config saved
pp = pprint.PrettyPrinter()
with open(os.path.join(path, "config.json")) as f:
    print(f"{type(model2).__name__} Config")
    pp.pprint(json.load(f))

# Load the model using Prophet.load()
model2_loaded = Prophet.load(dirname=path)

# Load the model using the ModelFactory
model2_factory_loaded = ModelFactory.load(name="Prophet", model_path=path)