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
  
  def train(self, X_train, y_train):
      self.model.fit(X_train, y_train)
  
  def predict(self, X_test):
      return self.model.predict(X_test)
  
  def predict_proba(self, X_test):
      return self.model.predict_proba(X_test)
  
  def get_model(self):
      return self.model
  
  def save_model(self, root:Path = Path('/models')):
      joblib.dump(self.model, root)

  def load_model(self, root:Path = Path('/models')):
        self.model = joblib.load(root)

# 모델을 불러와서 예측하는 코드
def data_pred(data, root:Path = Path('/models')):
    model = joblib.load(root)
    return  model.predict(data)