# 모델을 모아둔 모듈
# + 모델 저장 기능
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

class ClassificationModels:
  def __init__(self, model:str='random_forest', **kwargs):
    self.model = model
    self.model = self._init_model(**kwargs)
  
  def _init_model(self, **kwargs):
    if self.model == 'random_forest':
      return RandomForestClassifier(**kwargs)
    elif self.model == 'xgboost':
      return XGBClassifier(**kwargs)
    elif self.model == 'lightgbm':
      return LGBMClassifier(**kwargs)
    elif self.model == 'catboost':
      return CatBoostClassifier(**kwargs)
    else:
      raise ValueError("model : 'random_forest', 'xgboost', 'lightgbm', 'catboost")
  
  def train(self, feature, target):
      self.model.fit(feature, target)
  
  def predict(self, X):
    return self.model.predict(X)
  
  def predict_proba(self, X):
    return self.model.predict_proba(X)

  def threshold_pred(self, X, threshold = 0.5):
    return self.model.predict_proba(X)[:, 1] > threshold
  
  def get_model(self):
    return self.model
  
  def save_model(self, root:Path = Path('/models'), ):
    joblib.dump(self.model, root / self.model)

  def load_model(self, root:Path = Path('/models')):
    self.model = joblib.load(root / self.model)

# 모델을 불러와서 예측하는 코드
def data_pred(data, root:Path = Path('/models'), model_name:str = 'randomforest'):
    model = joblib.load(root / model_name)
    return  model.predict(data)