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
  
  # 모델 저장학기 위한 함수
  def save_model(self, root:Path = Path('/models'), ):
    joblib.dump(self.model, root / self.model)

  # 모델 불러오기 위한 함수
  def load_model(self, root:Path = Path('/models')):
    self.model = joblib.load(root / self.model)

# 모델을 불러와서 예측하는 코드
def data_pred(data, root:Path = None, model_name:str = 'randomforest'):
    # 기본 모델 경로 설정
    if root is None:
        import os
        # 현재 파일의 디렉토리 경로 가져오기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 프로젝트 루트 디렉토리 (service의 상위 디렉토리)
        project_root = os.path.dirname(current_dir)
        # 모델 디렉토리 경로
        root = Path(os.path.join(project_root, 'models'))
    
    # 모델 파일 경로 확인
    model_path = root / model_name
    print(f"Loading model from: {model_path}")
    
    try:
        model = joblib.load(model_path)
        return model.predict(data)
    except Exception as e:
        print(f"Error loading model: {e}")
        # 오류 발생 시 대체 모델 시도
        try:
            import os
            # 사용 가능한 모델 파일 찾기
            available_models = [f for f in os.listdir(root) if f.endswith('.pkl')]
            if available_models:
                alt_model_path = root / available_models[0]
                print(f"Trying alternative model: {alt_model_path}")
                model = joblib.load(alt_model_path)
                return model.predict(data)
            else:
                raise FileNotFoundError(f"No model files found in {root}")
        except Exception as e2:
            print(f"Failed to load alternative model: {e2}")
            # 모든 시도 실패 시 기본 예측 반환 (모든 값이 0)
            import numpy as np
            print("Returning default predictions (all zeros)")
            return np.zeros(len(data))