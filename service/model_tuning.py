import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from models import ClassificationModels
from data_setup import data_loader, data_preprocessing, input_mode

class ModelTuner(ClassificationModels):
    """
    ClassificationModels를 확장하여 그리드서치와 랜덤서치 기능을 추가한 클래스
    """
    def __init__(self, model='random_forest', **kwargs):
        super().__init__(model=model, **kwargs)
        self.best_grid_model = None
        self.best_random_model = None
        self.grid_results = None
        self.random_results = None
        
    def get_default_param_grid(self):
        """
        모델 유형에 따른 기본 하이퍼파라미터 그리드 반환
        """
        if self.model.__class__.__name__ == 'RandomForestClassifier':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
        elif self.model.__class__.__name__ == 'LGBMClassifier':
            return {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
        elif self.model.__class__.__name__ == 'XGBClassifier':
            return {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
        elif self.model.__class__.__name__ == 'CatBoostClassifier':
            return {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5, 7],
            }
        else:
            return {}
            
    def tune_hyperparameters(self, X_train, y_train, X_test=None, y_test=None, 
                            param_grid=None, cv=5, scoring='roc_auc', n_jobs=-1, 
                            random_iter=10, verbose=1):
        """
        그리드서치와 랜덤서치를 사용하여 하이퍼파라미터 튜닝 수행
        
        Parameters:
        -----------
        X_train : DataFrame
            학습 데이터 특성
        y_train : Series
            학습 데이터 타겟
        X_test : DataFrame, optional
            테스트 데이터 특성
        y_test : Series, optional
            테스트 데이터 타겟
        param_grid : dict, optional
            하이퍼파라미터 그리드 (None인 경우 기본값 사용)
        cv : int, optional
            교차 검증 폴드 수
        scoring : str, optional
            최적화할 평가 지표
        n_jobs : int, optional
            병렬 처리에 사용할 CPU 코어 수
        random_iter : int, optional
            랜덤서치에서 시도할 파라미터 조합 수
        verbose : int, optional
            출력 상세도
            
        Returns:
        --------
        dict
            튜닝 결과를 포함하는 딕셔너리
        """
        model_name = self.model.__class__.__name__
        print(f"\n{model_name} 하이퍼파라미터 튜닝 시작:")
        
        # 파라미터 그리드가 제공되지 않은 경우 기본값 사용
        if param_grid is None:
            param_grid = self.get_default_param_grid()
            
        # 그리드서치 수행
        print(f"\n{model_name} 그리드서치 수행 중...")
        grid_search = GridSearchCV(
            self.model, 
            param_grid=param_grid, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=n_jobs, 
            verbose=verbose
        )
        grid_search.fit(X_train, y_train)
        
        self.best_grid_model = grid_search.best_estimator_
        
        print(f"{model_name} 그리드서치 최적 파라미터:", grid_search.best_params_)
        print(f"{model_name} 그리드서치 최고 점수:", grid_search.best_score_)
        
        # 랜덤서치 수행
        print(f"\n{model_name} 랜덤서치 수행 중...")
        random_search = RandomizedSearchCV(
            self.model, 
            param_distributions=param_grid, 
            n_iter=random_iter, 
            cv=cv, 
            scoring=scoring, 
            n_jobs=n_jobs, 
            verbose=verbose
        )
        random_search.fit(X_train, y_train)
        
        self.best_random_model = random_search.best_estimator_
        
        print(f"{model_name} 랜덤서치 최적 파라미터:", random_search.best_params_)
        print(f"{model_name} 랜덤서치 최고 점수:", random_search.best_score_)
        
        # 테스트 데이터가 제공된 경우 성능 평가
        results = {
            'grid_search': {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            },
            'random_search': {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'cv_results': random_search.cv_results_
            }
        }
        
        if X_test is not None and y_test is not None:
            # 타겟 변수가 문자열인 경우 숫자로 변환
            if isinstance(y_test.iloc[0], str):
                y_test = y_test.map({'Yes': 1, 'No': 0})
            
            # 그리드서치 모델 평가
            grid_y_pred = self.best_grid_model.predict(X_test)
            grid_y_prob = self.best_grid_model.predict_proba(X_test)[:, 1]
            
            grid_metrics = {
                'accuracy': accuracy_score(y_test, grid_y_pred),
                'precision': precision_score(y_test, grid_y_pred),
                'recall': recall_score(y_test, grid_y_pred),
                'f1': f1_score(y_test, grid_y_pred),
                'roc_auc': roc_auc_score(y_test, grid_y_prob)
            }
            
            print(f"\n{model_name} 그리드서치 테스트 성능:")
            for metric, value in grid_metrics.items():
                print(f"{metric}: {value:.4f}")
                
            # 랜덤서치 모델 평가
            random_y_pred = self.best_random_model.predict(X_test)
            random_y_prob = self.best_random_model.predict_proba(X_test)[:, 1]
            
            random_metrics = {
                'accuracy': accuracy_score(y_test, random_y_pred),
                'precision': precision_score(y_test, random_y_pred),
                'recall': recall_score(y_test, random_y_pred),
                'f1': f1_score(y_test, random_y_pred),
                'roc_auc': roc_auc_score(y_test, random_y_prob)
            }
            
            print(f"\n{model_name} 랜덤서치 테스트 성능:")
            for metric, value in random_metrics.items():
                print(f"{metric}: {value:.4f}")
                
            results['grid_search']['test_metrics'] = grid_metrics
            results['random_search']['test_metrics'] = random_metrics
        
        self.grid_results = results['grid_search']
        self.random_results = results['random_search']
        
        return results
    
    def save_best_models(self, save_path='./models'):
        """
        최적의 모델을 저장
        
        Parameters:
        -----------
        save_path : str
            모델을 저장할 경로
        """
        os.makedirs(save_path, exist_ok=True)
        model_name = self.model.__class__.__name__.lower()
        
        if self.best_grid_model is not None:
            grid_path = os.path.join(save_path, f'best_{model_name}_grid.pkl')
            self.save_model(self.best_grid_model, grid_path)
            print(f"그리드서치 최적 모델 저장됨: {grid_path}")
            
        if self.best_random_model is not None:
            random_path = os.path.join(save_path, f'best_{model_name}_random.pkl')
            self.save_model(self.best_random_model, random_path)
            print(f"랜덤서치 최적 모델 저장됨: {random_path}")
    
    def save_model(self, model, path):
        """
        모델을 지정된 경로에 저장
        """
        import joblib
        joblib.dump(model, path)
    
    def get_best_model(self, method='grid'):
        """
        최적의 모델 반환
        
        Parameters:
        -----------
        method : str
            'grid' 또는 'random' (기본값: 'grid')
            
        Returns:
        --------
        model
            최적의 모델
        """
        if method.lower() == 'grid':
            return self.best_grid_model
        elif method.lower() == 'random':
            return self.best_random_model
        else:
            raise ValueError("method는 'grid' 또는 'random'이어야 합니다.")


def train_and_tune_models(data_root='./data', data_filename='churn_data.csv', save_path='./models'):
    """
    데이터를 로드하고 모델을 학습 및 튜닝하는 함수
    
    Parameters:
    -----------
    data_root : str
        데이터 파일이 위치한 디렉토리 경로
    data_filename : str
        데이터 파일명 (CSV 파일)
    save_path : str
        학습된 모델을 저장할 경로
        
    Returns:
    --------
    dict
        모델 튜너 객체들을 포함하는 딕셔너리
    """
    # 데이터 로드 및 전처리
    print("데이터 로딩 중...")
    data = data_loader(data_root, data_filename)
    
    # 데이터 전처리 및 분할
    print("데이터 전처리 및 분할 중...")
    # 'Churn'이 타겟 컬럼이라고 가정
    if 'Churn' not in data.columns:
        raise ValueError("데이터에 'Churn' 컬럼이 없습니다. 타겟 컬럼을 확인하세요.")
    
    # Churn 컬럼을 숫자로 변환 (Yes=1, No=0)
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # 타겟 컬럼 이름 변경 (input_mode 함수에서 'target'을 찾기 때문)
    data = data.rename(columns={'Churn': 'target'})
    
    # 데이터 전처리 및 분할
    X_train, X_test, y_train, y_test = input_mode(data, train_mode=True)
    
    # 모델 튜너 객체 생성
    model_tuners = {
        'random_forest': ModelTuner(model='random_forest', random_state=42),
        'lightgbm': ModelTuner(model='lightgbm', random_state=42)
    }
    
    # 각 모델 튜닝
    for name, tuner in model_tuners.items():
        print(f"\n{name.upper()} 모델 튜닝 중...")
        tuner.tune_hyperparameters(X_train, y_train, X_test, y_test)
        tuner.save_best_models(save_path)
    
    return model_tuners


if __name__ == "__main__":
    # 모델 학습 및 튜닝 실행
    # 데이터 경로 수정 (./.data -> ./data)
    model_tuners = train_and_tune_models(data_root='./.data', data_filename='WA_Fn-UseC_-Telco-Customer-Churn.csv', save_path='./models')
    
    # 최적의 모델 가져오기 예시
    best_rf_model = model_tuners['random_forest'].get_best_model(method='grid')
    best_lgbm_model = model_tuners['lightgbm'].get_best_model(method='random')
    
    print("\n최적 모델 정보:")
    print(f"Random Forest: {best_rf_model}")
    print(f"LightGBM: {best_lgbm_model}") 