import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from eeg_emotion.config import Config
from eeg_emotion.data import load_windows
from eeg_emotion.evaluation import split_indices,metrics
from eeg_emotion.features import extract_features

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--model',required=True); p.add_argument('--target',default='valence'); p.add_argument('--output',required=True); args=p.parse_args()
d=load_windows(args.input); cfg=Config.from_yaml(ROOT/'configs/default.yaml'); y=d['y_'+args.target]; F,_=extract_features(d['X'],cfg.sampling_rate,cfg.bands); tr,te=split_indices(y,d['subjects'],'subject',seed=cfg.seed); bundle=joblib.load(args.model); m=bundle['model']; pred=m.predict(F[te]); prob=m.predict_proba(F[te])[:,1] if hasattr(m,'predict_proba') else None
r=metrics(y[te],pred,prob); r['confusion_matrix']=confusion_matrix(y[te],pred).tolist(); Path(args.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame([r]).to_json(ROOT/args.output,orient='records',indent=2); print(r)
