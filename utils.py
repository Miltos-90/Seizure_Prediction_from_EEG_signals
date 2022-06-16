import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from tqdm.notebook import tqdm
import gc
import mne

# Path with the raw data
DATA_PATH = '/kaggle/input/seizure-prediction/'

def _isMatFile(filename):  return '.mat' in filename
def _isTestFile(filename): return '_test_' in filename


''' Parses a filename for the data indexer'''
def _parsePathname(pathname):
    
    folderTree = pathname.split('/')
    subject    = folderTree[-2]
    filename   = folderTree[-1].strip('.mat')
    fnameParts = filename.split('_')
    segment    = int(fnameParts[-1])
    label      = 'normal' if 'interictal' in filename else 'seizure'
            
    return pathname, subject, segment, label


''' Data indexer '''
def indexFiles(path = DATA_PATH):

    metadata = []
    for dirname, _, pathnames in os.walk(path):
        for pathname in pathnames:
            if _isMatFile(pathname) and not _isTestFile(pathname):
                fullPath = os.path.join(dirname, pathname)
                metadata.append(_parsePathname(fullPath))

    df = pd.DataFrame(metadata, columns = ['path', 'subject', 'segment', 'label'])
    df = df[df['subject'].str.contains("Dog")] # Keep only dog subjects

    return df


''' Imports one mat. file '''
def parseFile(filename):
    
    for key, contents in loadmat(filename).items():

        # Skip matlab's metadata
        if '__' not in key: 
            contents = contents[0][0] # Strip not needed values

            duration   = float(contents[1])
            frequency  = float(contents[2])
            chNames    = [elem[0] for elem in contents[3][0]]
            sequenceNo = int(contents[4])
            EEGdata    = contents[0].astype(float)

            break
        
    return duration, frequency, chNames, sequenceNo, EEGdata


''' Plots the entire sequence for one subject '''
def plotSubject(df):
    
    df.sort_values(['label', 'segment'], inplace = True)
    
    data   = []
    for row, record in tqdm(df.iterrows(), total = df.shape[0]):
        _, _, _, _, signals = parseFile(record['path'])
        data.append(signals[:, 0::400].astype(np.float16))

    data    = np.concatenate(data, axis = 1)
    fig, ax = plt.subplots(figsize = (16, 4))

    for ch in range(data.shape[0]):
        plt.subplot(16, 1, ch + 1)
        plt.plot(data[ch, :])
        plt.axis('off')
    plt.show()
    
    return


''' Removes one channel from the data '''
def removeChannel(chNames, signals, suffix = '004'):
    
    for chIdx, chName in enumerate(chNames):
        if chName.endswith(suffix):
            signals = np.delete(signals, chIdx, axis = 0)
            chNames = np.delete(chNames, chIdx, axis = 0)
            break

    return list(chNames), signals