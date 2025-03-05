import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from service.data_setup import *
from service.input_pred import input_preprocessing, tenure_predict, fit_columns

def main():
    # 페이지 설정
    st.set_page_config(layout="wide")

    # 사이드바
    with st.sidebar:
        st.header("Menu")
        
        # CSS를 사용하여 탭 스타일 적용
        st.markdown(
            """
            <style>
            div[data-testid="stVerticalBlock"] div[class="row-widget stButton"] {
                width: 100%;
                margin-bottom: 5px;
            }
            .stButton > button {
                width: 100%;
                background-color: transparent;
                color: black;
                border: none;
                text-align: left;
                padding: 15px;
                border-radius: 0;  /* 모서리를 각지게 */
            }
            .stButton > button:hover {
                background-color: #262730;
                color: white;
            }
            .stButton > button[kind="primary"] {
                background-color: black;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # 세션 스테이트로 현재 선택된 메뉴 관리
        if 'current_menu' not in st.session_state:
            st.session_state.current_menu = "EDA & 모델 성능 비교"

        # 메뉴 버튼
        col1, = st.columns(1)  # 전체 너비 사용
        with col1:
            if st.button("EDA & 모델 성능 비교", 
                        key="eda",
                        type="primary" if st.session_state.current_menu == "EDA & 모델 성능 비교" else "secondary"):
                st.session_state.current_menu = "EDA & 모델 성능 비교"
                st.rerun()

            if st.button("예측 프로그램",
                        key="predict",
                        type="primary" if st.session_state.current_menu == "예측 프로그램" else "secondary"):
                st.session_state.current_menu = "예측 프로그램"
                st.rerun()

    # 선택된 메뉴에 따라 다른 내용 표시
    if st.session_state.current_menu == "EDA & 모델 성능 비교":
        # 메인 탭
        st.subheader("통신사 고객 이탈 예측")

        tab1, tab2, tab3 = st.tabs(["시각화", "Feature Engineering", "모델 성능비교"])

        with tab1:
            # 시각화 영역
            st.write("#### 고객 특성별 이탈 현황")
            
            # 캐러셀 구현
            if 'carousel_index' not in st.session_state:
                st.session_state.carousel_index = 0
            
            images = [
                {"path": "image.png", "caption": "서비스별 이탈 현황"},
                {"path": "image copy.png", "caption": "계약 유형별 이탈 현황"},
                {"path": "image copy 2.png", "caption": "인구통계별 이탈 현황"}
            ]
            
            # 이미지 표시를 위한 컨테이너
            image_container = st.container()
            with image_container:
                st.image(
                    images[st.session_state.carousel_index]["path"],
                    caption=images[st.session_state.carousel_index]["caption"],
                    use_container_width=True
                )
            
            # 네비게이션 버튼
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀ 이전"):
                    st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(images)
                    st.rerun()
            
            # 현재 이미지 위치 표시
            with col2:
                st.markdown(
                    f"<div style='text-align: center;'>{st.session_state.carousel_index + 1} / {len(images)}</div>", 
                    unsafe_allow_html=True
                )
            
            with col3:
                if st.button("다음 ▶"):
                    st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(images)
                    st.rerun()

        with tab2:
            st.subheader("Feature Engineering")
            
            # 특성 중요도 시각화
            st.write("#### 상위 20개 특성 중요도")
            st.image("image copy 3.png", 
                    caption="특성 중요도 분석 결과",
                    use_container_width=True)
            
            # 특성 중요도 설명
            st.write("""
            주요 특성 분석 결과:
            - 서비스당 요금, 월별 요금(MonthlyCharges), 이용 기간(tenure)이 가장 중요한 특성으로 나타남
            - 고객 위험도와 계약 기간 수치도 높은 중요도를 보임
            - 디지털 친화도와 인터넷 서비스 등급이 중간 정도의 중요도를 가짐
            - 인구통계학적 특성(성별, SeniorCitizen 등)은 상대적으로 낮은 중요도를 보임
            """)
            
            # EDA 섹션 추가
            st.write("#### EDA")
            
            # 데이터 전처리 테이블 생성
            preprocessing_data = {
                '작업내용': [
                    '불필요한 컬럼 제거',
                    '결측치 처리',
                    '범주형 변수 변환',
                    '이진 변수 변환',
                    '새로운 변수 변환',
                    '요금 관련 변수 추가',
                    '자동 결제 여부'
                ],
                '설명': [
                    'customerID, gender 삭제 (모델 학습에 불필요)',
                    'TotalCharges를 조치로 변환 후, 중앙값으로 처리',
                    '원핫인코딩 (InternetService, Contract, PaymentMethod)',
                    'Yes → 1, No → 0 변환 (Partner, MultipleLines, OnlineSecurity 등)',
                    'tenure_group (가입 기간 그룹화), TotalServices (가입한 서비스 수)',
                    'ChargeChange (월 요금 변동량 계산)',
                    'AutoPayment 변수 추가 (check 포함이면 0, 아니면 1)'
                ]
            }
            
            # DataFrame 생성 및 스타일링
            df_preprocessing = pd.DataFrame(preprocessing_data)
            
            # 테이블 표시
            st.table(df_preprocessing.style.set_properties(**{
                'background-color': 'white',
                'color': 'black',
                'border-color': 'darkgrey'
            }))

        with tab3:
            st.subheader("모델 성능 비교")
            
            # 모델 성능 비교 테이블 생성
            model_comparison = {
                '모델': ['Random Forest', 'LightGBM', 'XGBoost', 'ExtraTrees'],
                'Accuracy': ['78.5%', '76.3%', '73.1%', '74.8%'],
                'Precision': ['0.79', '0.71', '0.74', '0.75'],
                'Recall': ['0.76', '0.74', '0.72', '0.78'],
                'F1-score': ['0.77', '0.74', '0.71', '0.72'],
                'AUC-ROC': ['0.85', '0.82', '0.70', '0.79']
            }
            
            # DataFrame 생성
            df_model_comparison = pd.DataFrame(model_comparison)
            
            # 테이블 스타일링 및 표시
            st.table(df_model_comparison.style.set_properties(**{
                'background-color': 'white',
                'color': 'black',
                'border-color': 'darkgrey',
                'text-align': 'center'
            }))
            
            # 성능 비교 결과 설명
            st.write("""
            #### 모델 성능 분석 결과
            - Random Forest 모델이 전반적으로 가장 우수한 성능을 보임
                * 가장 높은 정확도(78.5%)와 AUC-ROC(0.85) 달성
                * Precision과 F1-score에서도 최고 성능
            - LightGBM은 두 번째로 좋은 성능을 보이며, 특히 처리 속도가 빠름
            - ExtraTrees는 Recall에서 가장 좋은 성능을 보임
            - XGBoost는 상대적으로 낮은 성능을 보였으나, 안정적인 예측을 제공
            """)
    
    else:
        st.header("이탈 가능성 예측 프로그램")
        # 딕셔너리 정리
        # 계약 기간
        contract_dict = {
            '1개월 계약' : 'Month-to-month',
            '1년 계약' : 'One year',
            '2년 계약' : 'Two year'
        }
        # 바이너리 dict
        binary_dict_ver1 = {
            '아니오' : 'No',
            '예' : 'Yes'
        }
        binary_dict_ver2 = {
            '없음' : 'No',
            '있음' : 'Yes'
        }
        internet_dict = {
            '가입하지 않음' : 'No',
            'DSL 인터넷 (전화선 인터넷)' : 'DSL',
            '광섬유 인터넷 (광랜)' : 'Fiber optic'
        }
        internet_service_dict = {
            '온라인 보안 서비스' : 'OnlineSecurity',
            '온라인 백업 서비스' : 'OnlineBackup',
            '기기 보호 서비스' : 'DeviceProtection',
            '기술 지원 서비스' : 'TechSupport',
            '스마트 TV 서비스' : 'StreamingTV',
            '스마트 영화 서비스' : 'StreamingMovies'
        }
        payment_method_dict = {
            '전자 결제' : 'Electronic check',
            '지로 납부' : 'Mailed check',
            '자동 이체' : 'Bank transfer (automatic)',
            '자동 카드 결제' : 'Credit card (automatic)'
        }
        # 입력 폼 생성
        with st.container(border=True):
            pred_col1, pred_col2 = st.columns(2)
            with pred_col1:
                # 연령
                st.write("#### 연령")
                age = st.number_input("", key="age", step = 1)
                # 배우자 여부
                st.write('### 배우자 여부')
                st.selectbox("", binary_dict_ver2.keys(), key="partner")
            
            with pred_col2:
                # 게약 기간
                st.write("#### 계약 기간")
                st.selectbox("", contract_dict.keys(), key="contract")
                # 부양가족 여부
                st.write('### 부양가족 여부')
                st.selectbox("", binary_dict_ver2.keys(), key="dependents")
            
            # 휴대폰 가입 여부
            st.write("#### 휴대폰 가입 여부")
            st.selectbox("", binary_dict_ver1.keys(), key="phone")
            # 이용중인 휴대폰 서비스 선택
            if binary_dict_ver1[st.session_state.phone] == "Yes":
                st.write("#### 다중회선 사용 여부")
                st.radio("", binary_dict_ver1.keys(), key="multiple_lines")
            
            # 인터넷 가입 여부
            st.write("#### 인터넷 가입 여부")
            st.selectbox("", internet_dict.keys(), key="internetservice")
        
            # 이용하고 있는 서비스 선택
            if internet_dict[st.session_state.internetservice] != "No":
                st.write("#### 이용하고 있는 서비스 선택")
                st.multiselect(
                    "",
                    internet_service_dict.keys(), key = "internet_service")
            
            pred_col3, pred_col4 = st.columns(2)
            with pred_col3:
                # 결제 수단
                st.write("#### 결제 수단")
                st.selectbox("", payment_method_dict.keys(), key="payment_method")
            with pred_col4:
                # 온라인 청구서 여부
                st.write("#### 온라인 청구서 신청 여부")
                st.selectbox("", binary_dict_ver1.keys(), key="paperlessbilling")
            
            # 예측하기 버튼
            predict_button = st.button("예측하기", type="primary", use_container_width=True)
            
            if predict_button:
                # input_pred.py의 tenure_predict 함수에 전달할 입력 데이터 구성
                n = 61
                input_data = {
                    'customerID' : [''] * n,
                    'gender' : [''] * n,
                    'SeniorCitizen' : [int(age <= 60)] * n,
                    'Partner' : [binary_dict_ver2[st.session_state.partner]] * n,
                    'Dependents' : [binary_dict_ver2[st.session_state.dependents]] * n,
                    'tenure' : range(0, n, 1),
                    'PhoneService' : [binary_dict_ver1[st.session_state.phone]] * n,
                    'MultipleLines' : ['No phone service' if binary_dict_ver1[st.session_state.phone] == 'No' else binary_dict_ver1[st.session_state.multiple_lines]] * n,
                    'InternetService' : [internet_dict[st.session_state.internetservice]] * n,
                    'OnlineSecurity' : ['No internet service' if internet_dict[st.session_state.internetservice] == 'No' else 'Yes' if '온라인 보안 서비스' in st.session_state.internet_service else 'No'] * n,
                    'OnlineBackup' : ['No internet service' if internet_dict[st.session_state.internetservice] == 'No' else 'Yes' if '온라인 백업 서비스' in st.session_state.internet_service else 'No'] * n,
                    'DeviceProtection' : ['No internet service' if internet_dict[st.session_state.internetservice] == 'No' else 'Yes' if '기기 보호 서비스' in st.session_state.internet_service else 'No'] * n,
                    'TechSupport' : ['No internet service' if internet_dict[st.session_state.internetservice] == 'No' else 'Yes' if '기술 지원 서비스' in st.session_state.internet_service else 'No'] * n,
                    'StreamingTV' : ['No internet service' if internet_dict[st.session_state.internetservice] == 'No' else 'Yes' if '스마트 TV 서비스' in st.session_state.internet_service else 'No'] * n,
                    'StreamingMovies' : ['No internet service' if internet_dict[st.session_state.internetservice] == 'No' else 'Yes' if '스마트 영화 서비스' in st.session_state.internet_service else 'No'] * n,
                    'Contract' : [contract_dict[st.session_state.contract]] * n,
                    'PaperlessBilling' : [binary_dict_ver1[st.session_state.paperlessbilling]] * n,
                    'PaymentMethod' : [payment_method_dict[st.session_state.payment_method]] * n
                }
                df = pd.DataFrame(input_data)
                # MonthlyChargets, TotalCharges 생성
                df = input_preprocessing(df)
                # 전처리
                df = input_mode(df)
                # 컬럼 정리
                df = fit_columns(df)
                # input_mode(df)
                # 예측 수행
                result = tenure_predict(data=df, root = Path('models'), model_name = 'rf.pkl')
                
                # 예측 결과 표시
                st.success("예측이 완료되었습니다!")
                
                # 이탈 예측 결과 메시지 - 더 크고 눈에 띄게 표시
                st.markdown(f"""
                <div style="background-color: #FF5252; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h2 style="color: white; text-align: center; margin: 0;">{result}</h2>
                </div>
                """, unsafe_allow_html=True)

                # 추가 정보 표시
                # col1, col2 = st.columns(2)
                # with col1:
                #     st.metric("이탈 확률", "78%")
                # with col2:
                #     st.metric("위험도", "높음")
                
                # todo: 주요 이탈 위험 요인 추가
                st.info("""
                #### 주요 이탈 위험 요인:
                - 다중회선 미사용
                - 보안 서비스 미가입
                - 기술 지원 서비스 미사용
                """)

def show_data_analysis():
    st.header("📊 데이터 분석")
    
    # 데이터 로드
    df = data_loader()
    
    # 기본 통계
    st.subheader("기본 통계")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 고객 수", f"{len(df):,}명")
    with col2:
        st.metric("이탈률", f"{(df['Churn'] == 'Yes').mean():.1%}")
    with col3:
        st.metric("평균 이용 기간", f"{df['tenure'].mean():.1f}개월")
    with col4:
        st.metric("평균 월 요금", f"${df['MonthlyCharges'].mean():.2f}")

    # 이탈 요인 분석
    st.subheader("주요 이탈 요인 분석")
    
    # 계약 유형별 이탈률
    fig, ax = plt.subplots(figsize=(10, 6))
    contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
    contract_churn['Yes'].plot(kind='bar', ax=ax)
    plt.title("계약 유형별 이탈률")
    plt.ylabel("이탈률")
    st.pyplot(fig)
    plt.close()

    # 상관관계 히트맵
    st.subheader("특성 간 상관관계")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    st.pyplot(fig)
    plt.close()

def show_model_performance():
    st.header("🎯 모델 성능")
    
    # 모델 성능 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("정확도", "0.75")
    with col2:
        st.metric("ROC-AUC", "0.8276")
    with col3:
        st.metric("이탈고객 Recall", "0.73")
    with col4:
        st.metric("F1-Score", "0.61")

    # 모델 발전 과정
    st.subheader("모델 발전 과정")
    progress_data = {
        '모델': ['Base', 'SMOTE', 'Tuned', 'Tuned V2'],
        '정확도': [0.80, 0.75, 0.74, 0.75],
        'ROC-AUC': [0.8336, 0.8279, 0.8017, 0.8276],
        '이탈고객 Recall': [0.52, 0.74, 0.64, 0.73]
    }
    st.table(pd.DataFrame(progress_data))

def show_prediction():
    st.header("🔮 고객 이탈 예측")
    
    # 입력 폼
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("이용 기간(개월)", 0, 72, 36)
        monthly_charges = st.slider("월 요금($)", 20, 120, 70)
        total_services = st.slider("총 서비스 수", 0, 10, 5)
    
    with col2:
        internet_service = st.selectbox(
            "인터넷 서비스 등급",
            ["Fiber optic", "DSL", "No"]
        )
        contract = st.selectbox(
            "계약 유형",
            ["Month-to-month", "One year", "Two year"]
        )
        payment_method = st.selectbox(
            "결제 방식",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )

    if st.button("예측하기"):
        # 예시 예측 결과
        churn_prob = 0.7  # 실제 모델 예측으로 대체 필요
        
        st.subheader("예측 결과")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("이탈 확률", f"{churn_prob:.1%}")
        
        with col2:
            risk_level = "높음" if churn_prob > 0.5 else "낮음"
            st.metric("위험도", risk_level)
        
        # 위험 요인 설명
        st.info("주요 위험 요인:")
        if churn_prob > 0.5:
            st.write("- Month-to-month 계약")
            st.write("- 높은 월 요금")
            st.write("- Electronic check 결제")

if __name__ == "__main__":
    main()
