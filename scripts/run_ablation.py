"""Frequency-band ablation helper: reruns the classical baseline with one band at a time."""
import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from eeg_emotion.config import Config
from eeg_emotion.data import load_windows
from eeg_emotion.evaluation import split_indices,metrics
from eeg_emotion.features import extract_features
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--target',default='valence'); p.add_argument('--output',required=True); args=p.parse_args()
cfg=Config.from_yaml(ROOT/'configs/default.yaml'); d=load_windows(args.input); y=d['y_'+args.target]; tr,te=split_indices(y,d['subjects'],'subject',seed=cfg.seed)
rows=[]
for band in cfg.bands:
    F,_=extract_features(d['X'],cfg.sampling_rate,{band:cfg.bands[band]})
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=cfg.seed)); m.fit(F[tr],y[tr]); pred=m.predict(F[te]); prob=m.predict_proba(F[te])[:,1]
    rows.append({'band':band,**metrics(y[te],pred,prob)})
pd.DataFrame(rows).to_csv(ROOT/args.output,index=False); print(pd.DataFrame(rows).to_string(index=False))
