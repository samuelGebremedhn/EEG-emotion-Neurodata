import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from eeg_emotion.config import Config
from eeg_emotion.data import load_windows
from eeg_emotion.evaluation import split_indices
from eeg_emotion.models import EEGNet
from eeg_emotion.utils import seed_everything

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--target',choices=['valence','arousal'],default='valence'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--output',required=True); p.add_argument('--config',default='configs/default.yaml'); args=p.parse_args()
cfg=Config.from_yaml(ROOT/args.config); seed_everything(cfg.seed)
d=load_windows(args.input); X=d['X']; y=d['y_'+args.target]; subjects=d['subjects']; tr,te=split_indices(y,subjects,'subject',seed=cfg.seed)
# Per-channel standardization using training data only.
mu=X[tr].mean(axis=(0,2),keepdims=True); sd=X[tr].std(axis=(0,2),keepdims=True)+1e-6
X=(X-mu)/sd
Xt=torch.tensor(X[tr],dtype=torch.float32); yt=torch.tensor(y[tr]); Xv=torch.tensor(X[te],dtype=torch.float32); yv=torch.tensor(y[te])
loader=DataLoader(TensorDataset(Xt,yt),batch_size=args.batch_size,shuffle=True)
device='cuda' if torch.cuda.is_available() else 'cpu'; model=EEGNet(X.shape[1],X.shape[2],2).to(device)
opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss()
for epoch in range(args.epochs):
    model.train();
    for xb,yb in loader:
        xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
model.eval();
with torch.no_grad(): pred=model(Xv.to(device)).argmax(1).cpu().numpy()
print({'accuracy':accuracy_score(yv,pred),'balanced_accuracy':balanced_accuracy_score(yv,pred),'macro_f1':f1_score(yv,pred,average='macro'),'device':device})
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
torch.save({'state_dict':model.state_dict(),'channels':X.shape[1],'samples':X.shape[2],'mean':mu,'std':sd,'target':args.target},ROOT/args.output)
