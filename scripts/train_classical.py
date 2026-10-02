import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_emotion.config import Config
from eeg_emotion.data import load_windows
from eeg_emotion.evaluation import split_indices, metrics
from eeg_emotion.features import extract_features
from eeg_emotion.utils import seed_everything

p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--target',choices=['valence','arousal'],default='valence'); p.add_argument('--split',choices=['subject','random'],default='subject'); p.add_argument('--output',required=True); p.add_argument('--config',default='configs/default.yaml'); args=p.parse_args()
cfg=Config.from_yaml(ROOT/args.config); seed_everything(cfg.seed)
d=load_windows(args.input); y=d['y_'+args.target]; X=d['X']; subjects=d['subjects']
F,names=extract_features(X,cfg.sampling_rate,cfg.bands)
tr,te=split_indices(y,subjects,args.split,seed=cfg.seed)
models={
 'logistic_regression':Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=cfg.seed))]),
 'random_forest':RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=-1,random_state=cfg.seed),
 'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=200,random_state=cfg.seed)
}
rows=[]
for name,m in models.items():
    m.fit(F[tr],y[tr]); pred=m.predict(F[te]); prob=m.predict_proba(F[te])[:,1] if hasattr(m,'predict_proba') else None
    r={'model':name,'target':args.target,'split':args.split,'n_train':len(tr),'n_test':len(te),**metrics(y[te],pred,prob)}; rows.append(r)
    if hasattr(m, 'feature_importances_'):
        pd.DataFrame({'feature': names, 'importance': m.feature_importances_}).sort_values('importance', ascending=False).to_csv(ROOT/'results'/f'{name}_{args.target}_feature_importance.csv', index=False)
    elif hasattr(m, 'named_steps') and hasattr(m.named_steps.get('clf'), 'coef_'):
        coef = abs(m.named_steps['clf'].coef_[0])
        pd.DataFrame({'feature': names, 'importance': coef}).sort_values('importance', ascending=False).to_csv(ROOT/'results'/f'{name}_{args.target}_feature_importance.csv', index=False)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); joblib.dump({'model':m,'feature_names':names,'config':cfg.__dict__},ROOT/'models'/f'{name}_{args.target}_{args.split}.joblib')
pd.DataFrame(rows).to_csv(ROOT/args.output,index=False); print(pd.DataFrame(rows).to_string(index=False))
