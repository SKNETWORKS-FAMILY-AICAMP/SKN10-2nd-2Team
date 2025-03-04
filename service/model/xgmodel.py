import sys
import os
# 현재 스크립트의 위치를 기준으로 상위 폴더(프로젝트 루트) 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib  # 모델 저장을 위한 라이브러리
from sklearn.model_selection import GridSearchCV



# 위에서 만든 데이터 처리 모듈 불러오기
from service.data_setup import data_loader, input_mode , data_preprocessing  # 파일 경로에 맞게 수정


# 데이터 로드
root = "./.data/"  # 데이터 파일이 위치한 폴더
filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # 사용자의 데이터 파일명으로 변경
data = data_loader(root, filename)

# 데이터 전처리
data = data_preprocessing(data) 

# # 데이터 분할
X_train, X_test, y_train, y_test = input_mode(data, train_mode=True)


# XGBoost 모델 생성 및 학습
model = xgb.XGBClassifier(
    objective='binary:logistic', 
    eval_metric='auc', 
    random_state=42
)

param_grid = {
    'n_estimators': [100, 200, 300],  # 트리 개수
    'learning_rate': [0.01, 0.05, 0.1],  # 학습률
    'max_depth': [3, 5, 7],  # 트리 깊이
    'subsample': [0.7, 0.8, 1.0],  # 샘플링 비율
    'colsample_bytree': [0.7, 0.8, 1.0]  # 각 트리에서 사용할 피처 비율
}

grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    scoring='roc_auc',  # AUC 기준으로 평가
    cv=3,  # 3-Fold Cross Validation
    n_jobs=-1,  # 모든 CPU 코어 사용
    verbose=2
)

# 하이퍼파라미터 최적화 실행
grid_search.fit(X_train, y_train)


# 최적의 모델 선택
best_model = grid_search.best_estimator_

# 최적 하이퍼파라미터 출력
print(f"Best Parameters: {grid_search.best_params_}")

# 테스트 데이터로 평가
y_pred = best_model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc_score:.4f}")
