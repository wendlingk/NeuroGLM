# NeuroGLM
Neural encoding of stimuli into spike trains via generalized linear models and decoding of spike trains into stimuli via Bayesian optimization. MATLAB code used in paper "Statistical Analysis of Decoding Performances of Diverse Populations of Neurons with Weakly Electric Fish Data"

Example Data: FakeData_All.mat

Three main scripts:
- EncodeDecode: Creates randomly selected networks of neurons for encoding GLMs and then decodes a composite stimulus. Useful for evaluating neural heterogeneity, as training sets are randomly selected from a time interval rather than a specific index. This makes it possible to select the same cell multiple times, just at different time steps, producing a wide domain of heterogeneity from fairly homogeneous networks to highly heterogeneous networks. 
- EncodeDecode_ValFollowing: Similar to EncodeDecode.m except training sets are drawn from a randomly selected specific time index, so cells are randomly drawn WITHOUT replacement. The decoding of the stimulus is completed on a validation time interval immediately following the training sets' time interval. Encoding GLMs can also be validated against cumulative spikes across the validation time interval in this script.
- Prewhiten_Correlations: To evaluate correlation strength of actual stimulus vs. decoded stimulus, the time series first need to be "prewhitened" to produce unbiased results. This script evaluates stationarity of the time series (a necessary condition before ARIMA time series model fitting) and evaluates stationarity of the once-differenced time series. Then ARIMA models are fit to the once-differenced actual and decoded stimuli (to ensure stationarity on all time series), residuals are captured from the model fits against these pairs of actual and decoded stimuli, and unbiased Pearson's correlation coefficients are recorded for each pair.

Directory contains one .mex file (spikeconv_mex.mexw64) but otherwise all files are from MATLAB (.m or .mat)
