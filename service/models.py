# 모델을 모아둔 모듈
# + 모델 저장 기능
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

from service.utils import reset_seeds

class ClassificationModels:
  def __init__(self, model:str='random_forest', **kwargs):
    self.model = model
    self.model = self._init_model(**kwargs)
  
  # 모델 초기 설정
  @reset_seeds
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
  
  # 학습하기 위한 함수
  @reset_seeds
  def train(self, feature, target):
      self.model.fit(feature, target)
  
  # 예측하기 위한 함수
  def predict(self, X):
    return self.model.predict(X)
  
  # 예측 확률값하기 위한 함수
  def predict_proba(self, X):
    return self.model.predict_proba(X)

  # 예측 확률값을 threshold로 조정하기 위한 함수
  def threshold_pred(self, X, threshold = 0.5):
    return self.model.predict_proba(X)[:, 1] > threshold
  
  # 모델을 불러오는 함수 (아마 필요 없을거같습니다)
  def get_model(self):
    return self.model
  
  # 모델 저장하기 위한 함수
  # def save_model(self, root:Path = Path('service/models')):
  #   joblib.dump(self.model, root / self.model)

  # # 모델 불러오기 위한 함수
  # def load_model(self, root:Path = Path('service/models')):
  #   self.model = joblib.load(root / self.model)

# 모델을 불러와서 예측하는 코드
def data_pred(input, root:Path = Path('models'), model_name:str = 'rf.pkl'):
    model = joblib.load(root / model_name)
    return  model.predict(input)
