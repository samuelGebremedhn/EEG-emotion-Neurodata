import numpy as np
from eeg_emotion.preprocessing import bandpass_filter, epoch_trial
from eeg_emotion.features import extract_features
from eeg_emotion.models import EEGNet

def test_preprocessing_and_features():
    rng=np.random.default_rng(1); x=rng.normal(size=(4,32,512)).astype('float32')
    y=bandpass_filter(x,128); e=epoch_trial(y[0],128,4,0.5); assert e.shape[1:]==(32,512)
    F,n=extract_features(e,128,{'alpha':(8,13),'beta':(13,30)}); assert F.shape[0]==e.shape[0]; assert F.shape[1]==len(n)

def test_eegnet_forward():
    model=EEGNet(32,512,2); out=model(np.zeros((2,32,512),dtype='float32')); assert out.shape==(2,2)
