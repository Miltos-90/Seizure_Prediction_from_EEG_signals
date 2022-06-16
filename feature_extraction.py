from scipy import stats
import pywt
import utils
from tqdm.notebook import tqdm

''' Extracts features from one level of the DWT'''
def extractLevelFeatures(signals):
    
    dSignals  = np.diff(signals, axis = 1)
    ddSignals = np.diff(dSignals, axis = 1)

    dSigStd   = dSignals.std(axis = 1)
    ddSigStd  = ddSignals.std(axis = 1)

    avg       = signals.mean(axis = 1)
    absAvg    = np.abs(signals).mean(axis = 1)
    std       = signals.std(axis = 1)
    cvar      = std / avg
    skew      = stats.skew(signals, axis = 1)
    kurt      = stats.kurtosis(signals, axis = 1)
    absMed    = np.median(np.abs(signals), axis = 1)
    minVal    = signals.min(axis = 1)
    maxVal    = signals.max(axis = 1)
    rmsVal    = np.sqrt(np.square(signals).mean(axis = 1))
    curvLen   = np.abs(dSignals).sum(axis = 1)
    iqr       = stats.iqr(signals, axis = 1)
    mobil     = dSigStd / std
    complx    = ddSigStd * std / (dSigStd ** 2)

    feats = np.hstack((avg, absAvg, std, cvar, skew, kurt, absMed, 
                       minVal, maxVal, rmsVal, curvLen, iqr, mobil, complx))
    
    return feats


''' Averages out signals whose probes are symmetrically located on the two brain hemispheres 
    Electrode placement: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4524640/
'''
def averageChannels(chNames, signals):
    
    s1 = signals[[0, 7],:].mean(axis = 0)
    s2 = signals[[1, 8],:].mean(axis = 0)
    s3 = signals[[2, 9],:].mean(axis = 0)
    s4 = signals[10,:] # Left hemisphere signal has been removed on 105 of the dataset
    s5 = signals[[3, 11],:].mean(axis = 0)
    s6 = signals[[4, 12],:].mean(axis = 0)
    s7 = signals[[5, 13],:].mean(axis = 0)
    s8 = signals[[6, 14],:].mean(axis = 0)
    signals = np.vstack((s1, s2, s3, s4, s5, s6, s7, s8))
    chNames = ['s' + str(c) for c in range(8)]
    
    return chNames, signals


''' Extracts features from the entire dataset'''
def extractFeatures(df, waveletName, waveletLevels):
    
    X, y = [], []
    iterable = tqdm(df.iterrows(), total = df.shape[0])
    
    for index, record in iterable:

        # Load file
        _, _, chNames, _, signals = utils.parseFile(record['path']) # Read mat. file
        chNames, signals = utils.removeChannel(chNames, signals)    # Remove missing channel from all data
        chNames, signals = averageChannels(chNames, signals)        # Avera according to electrode placement

        # Extract features
        feats = []
        for _ in range(waveletLevels):
            signals, _ = pywt.dwt(signals, waveletName, axis = 1)   # Run wavelet transformation
            feats.append(extractLevelFeatures(signals))             # Extract features
        
        X.append(np.concatenate(feats))                             # Flatten to single numpy array
        y.append(1 if record['label'] == 'seizure' else 0)          # Make label

    # to numpy
    X = np.array(X)
    y = np.array(y)
    
    return X, y