import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def data_loader(root: str, filename: str) -> pd.DataFrame:
  '''
  입력변수 : root, filename
  root에서 csv파일을 불러와서 DataFrame형태로 return하는 함수
  '''
  file_path = os.path.join(root, filename)
  data = pd.read_csv(file_path, encoding='utf-8')
  return data

def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    '''
    입력변수 : data(DataFrame)
    1. data를 입력받아 전처리해줌
    2. return data
    '''
    # 1) 불필요한 컬럼 제거
    drop_cols = ['customerID', 'gender']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])

    # 2) 결측치 처리 (TotalCharges -> numeric 변환 후 중앙값 대체)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

    # 3) tenure 그룹화
    data['tenure_group'] = pd.cut(
        data['tenure'],
        bins=[0, 12, 24, 48, 72, 1000],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True
    ).astype('Int64')  # Int64로 변환하여 NaN 문제 해결
    data['tenure_group'] = data['tenure_group'].fillna(5)

    # 4) 요금 관련 변수 추가
    data['ChargesRatio'] = data['TotalCharges'] / (data['MonthlyCharges'] + 1)
    data['AverageMonthlyCharge'] = data['TotalCharges'] / (data['tenure'] + 1)
    data['ChargeChange'] = data['MonthlyCharges'] - data['AverageMonthlyCharge']

    # 5) 가족 여부 (Partner 또는 Dependents가 'Yes'이면 1)
    data['Family'] = ((data['Partner'] == 'Yes') | (data['Dependents'] == 'Yes')).astype(int)

    # 6) 자동 결제 여부 (결제 방식에 'check'가 들어있으면 0, 아니면 1)
    data['AutoPayment'] = data['PaymentMethod'].apply(lambda x: 0 if 'check' in str(x).lower() else 1)

    # 7) 범주형 변수 변환 (One Hot Encoding)
    categorical_cols = ['InternetService', 'Contract', 'PaymentMethod']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # 8) 이진 변수(Yes/No) 처리
    binary_cols = [
        "Partner", "MultipleLines", "Dependents", "PhoneService",
        "PaperlessBilling", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map({'Yes': 1, 'No': 0})
            data[col] = data[col].fillna(0)

    # 9) 서비스 관련 변수 합산 (MultipleLines, OnlineSecurity 등)
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    data['TotalServices'] = data[service_cols].sum(axis=1)

    # 10) 타겟 변수 변환 (Churn -> 0/1)
    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

    # 11) 사용 후 불필요한 컬럼 제거 (ChargesRatio, AverageMonthlyCharge)
    drop_cols_2 = ['ChargesRatio', 'AverageMonthlyCharge']
    data = data.drop(columns=[col for col in drop_cols_2 if col in data.columns])

    return data

def input_mode(data: pd.DataFrame, train_mode: bool = True, apply_smote: bool = True):
    '''
    입력변수 : data(DataFrame), train_mode(boolean), apply_smote(boolean)
    1. train_mode가 True일 경우
        - X(feature), y(target)을 나누어 줌
        - train_test_split을 이용하여 7:3으로 나누어줌
        - return X_train, X_test, y_train, y_test 
    2. train_mode가 False일 경우 data_preprocessing 실행
        - 사용처 : streamlit에서 입력받았을 때 사용
    '''
    if train_mode:
        # 예: 'Churn' 컬럼을 타겟으로 사용
        if 'Churn' not in data.columns:
            raise ValueError("DataFrame에 'Churn' 컬럼이 존재하지 않습니다. 타겟 컬럼명을 확인하세요.")

        # X, y 분리
        X = data.drop(columns=['Churn'])
        y = data['Churn']

        # 7:3으로 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True
        )

        #  데이터 타입을 float으로 변환 (SMOTE와 XGBoost 오류 방지)
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)

        # 오버샘플링 적용 여부 확인
        if apply_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test

    else:
        # train_mode가 False면 전체 data에 대해 전처리만 수행 후 반환
        return data_preprocessing(data)
