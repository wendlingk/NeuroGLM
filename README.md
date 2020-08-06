# NeuroGLM
Neural encoding of stimuli into spike trains via generalized linear models and decoding of spike trains into stimuli via Bayesian optimization. MATLAB code used in paper "Statistical Analysis of Decoding Performances of Diverse Populations of Neurons with Weakly Electric Fish Data"

Directory contains one .mex file (**spikeconv_mex.mexw64**) but otherwise all files are from MATLAB (.m or .mat)

Example Data: **FakeData_All.mat**

Pillow Lab's directory *GLMspiketools* (https://github.com/pillowlab/GLMspiketools) was helpful in the creation of the encoding and decoding scripts for this project and for some of the sub-functions. In addition, Shreejoy Tripathy's directory *mitral_cell_diversity* (https://github.com/stripathy/mitral_cell_diversity) proved useful for providing some of the other sub-functions.

Four main scripts:
- **EncodeDecode.m**: Creates randomly selected networks of neurons for encoding GLMs and then decodes a composite stimulus. Parameters can be adjusted in the first section. This script is useful for evaluating neural heterogeneity, as training sets are randomly selected from a time interval (whose size is declared by the *range* parameter) rather than a specific time index. This makes it possible to select the same cell multiple times, just at different time steps, producing a wide domain of heterogeneity from fairly homogeneous networks to highly heterogeneous networks. Training sets are still somewhat synchronous in time between cells, with the decoding validation interval coming after the training selection region of the time series.
- **EncodeDecode_ValFollowing.m**: Similar to EncodeDecode.m except training sets are randomly drawn from the same specific time index (also randomly seleccted), so all cells are perfectly synchronous in time and thus must be randomly drawn *without* replacement to avoid duplicates. The decoding of the stimulus is completed on a validation time interval immediately following the training sets' time interval. Encoding GLMs can also be validated against cumulative spikes across the validation time interval in this script using the **EncodingModel_Validation** function called within the script.
- **EncodingModel_FilterLengths.m**: Splits the time series into equal segments, the number of which is declared by the *sets* parameter. The script iterates through different stimulus and post-spike filter lengths, creating encoding GLMs for all neurons at each time segment. Composite negative log likelihood values are recorded for each filter length combination in order to determine what is the optimal length for the models. Whatever parameters are chosen for *nkt* (stimulus filter length) and *hpeakvector* (time in seconds of last basis vector post-spike filter construction) can be updated in **EncodeDecode.m** and **EncodeDecode_ValFollowing.m** scripts. Generally with this project, validation of GLMs is done by decoding the stimulus or comparing actual versus predicted cumulative spikes in a time interval just after the training region, but the structure here with the **CrossValidationSets** function is well-designed to instead do cross-validation across all segments of the time series.
- **Prewhiten_Correlations.m**: To evaluate correlation strength of actual stimulus vs. decoded stimulus, the time series first need to be "prewhitened" to produce unbiased results. This script evaluates stationarity of the time series (a necessary condition before ARIMA time series model fitting) and evaluates stationarity of the once-differenced time series. If further differencing is needed Then ARIMA models are fit to the once-differenced actual and decoded stimuli (to ensure stationarity on all time series), residuals are captured from the model fits against these pairs of actual and decoded stimuli, and unbiased Pearson's correlation coefficients are recorded for each pair.

Training and validation sets are created differently depending on the script used. Each of the three encoding scripts has an associated function: 
- **EncodingSets.m** is a function called within **EncodeDecode.m**
- **EncodingSets_ValFollowing.m** is a function called within **EncodeDecode_ValFollowing.m**
- **CrossValidationSets.m** is a function called within **EncodingModel_FilterLengths.m**

The functions **bayesStimDecoder.m** and **bayesStimDecodeLogli.m** are modifications of the same files in Tripathy's neural GLM directory. All other functions in the directory not yet mentioned come directly from Shreejoy Tripathy's directory *mitral_cell_diversity*.
