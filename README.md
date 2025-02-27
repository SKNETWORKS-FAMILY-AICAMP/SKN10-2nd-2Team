# SKN10-2nd-2Team
# [가입 고객 이탈 예측](https://www.kaggle.com/code/bbksjdd/telco-customer-churn)



# 

## 팀 소개
<table>
  <tr>
    <th>김현수</th>
    <th>박현준</th>
    <th>정소열</th>
    <th>신민주</th>
    <th>조현정</th>
    <th>전서빈</th>

  </tr>
  <tr>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EC%9D%B8%ED%98%95.png" width="175" height="175"></td>
    <td><img src="https://i.namu.wiki/i/yHMdZs8LhKDP0D0XmvNkWe4NplRU5BDyXiZNDk5BTOd9ON5KtykFiDO_Q7SDpQLA-q9Q4fyFKfzM3apcZnPGtg.webp" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EC%8A%A4%EB%85%B8%EC%9A%B0%EB%A7%A8.png" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/baby.jpg" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EB%8F%99%EC%9D%B5.jpg" width="175" height="175"></td>
    <td><img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN07-2nd-3Team/blob/main/images/%EB%8F%99%EC%9D%B5.jpg" width="175" height="175"></td>


  </tr>
  <tr>
    <th>팀장</th>
    <th>팀원</th>
    <th>팀원</th>
    <th>팀원</th>
    <th>팀원</th>
    <th>팀원</th>
 
  </tr>
</table>

---

# 통신사 고객 이탈 예측 모델 개발 보고서
---

## 📊 목차
1. [데이터 분석 및 전처리](#1-데이터-분석-및-전처리)
2. [특성 엔지니어링](#2-특성-엔지니어링)
3. [모델 개발 과정](#3-모델-개발-과정)
4. [결과 및 시사점](#4-결과-및-시사점)
5. [기술 스택 및 향후 계획](#5-기술-스택-및-향후-계획)

---

## 1. 데이터 분석 및 전처리

### 1.1 데이터셋 개요
| 항목 | 내용 |
|------|------|
| 총 고객 수 | 7,043명 |
| 이탈률 | 26.54% |
| 고객 유지 기간 | 0-72개월 |
| 결측치 | TotalCharges 11건 (0.16%) |
| 특성 수 | 원본 21개 → 최종 6개 |


### 1.2 주요 이탈 요인 분석
| 요인 | 세부 내용 | 이탈률 | 전체 비중 |
|------|-----------|---------|------------|
| 계약 유형 | Month-to-month<br>One-year<br>Two-year | 42.7%<br>11.3%<br>2.8% | 55.0%<br>20.9%<br>24.1% |
| 서비스 유형 | Fiber optic<br>DSL<br>미사용 | 41.9%<br>19.0%<br>7.4% | 44.0%<br>34.4%<br>21.6% |
| 결제 방식 | Electronic check<br>Mailed check<br>Bank transfer<br>Credit card | 45.3%<br>19.1%<br>16.7%<br>15.2% | 33.6%<br>23.2%<br>21.8%<br>21.4% |

![image](https://github.com/user-attachments/assets/108532b5-05f8-4a35-95f8-8082c80dcfb8)
![image](https://github.com/user-attachments/assets/69b92809-52ec-4d82-bf04-28c92a629375)
![image](https://github.com/user-attachments/assets/6f56f1f3-e178-46a2-a4f3-e6260d725c49)
> 주요 요인별 이탈률 비교 및 분석

### 1.3 결측치 처리

~~~python
TotalCharges 결측치 처리
df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = \
df.loc[df['TotalCharges'].isna(), 'MonthlyCharges']
~~~
![image](https://github.com/user-attachments/assets/12bf96ab-becb-4964-9dc0-b486dd47a668)
> 결측치 처리 전후의 TotalCharges 분포 비교

---

## 2. 특성 엔지니어링

### 2.1 특성 개발 과정
| 단계 | 개발 특성 | 계산 방법 | 목적 |
|------|------------|------------|------|
| 1차 | 계약기간_수치 | `contract_type_mapping` | 계약 기간 수치화 |
| | 총_서비스_수 | `service_columns.sum()` | 서비스 활용도 측정 |
| | 고객_위험도 | `risk_factors.mean()` | 이탈 위험 정량화 |
| 2차 | 요금_증가율 | `(total - monthly) / tenure` | 요금 변동 추적 |
| | 고객_충성도 | `tenure * monthly_charges` | 고객 가치 산정 |
| | 서비스당_요금 | `monthly / total_services` | 서비스 효율성 |

![image](https://github.com/user-attachments/assets/f7d0244b-1415-466f-aee5-22be149fd66f)
![image](https://github.com/user-attachments/assets/e1f5a55e-3b3c-4b4d-9cd6-d206fc5721fa)

> 개발된 특성들 간의 상관관계 분석

### 2.2 최종 선정 특성
| 특성 | 중요도 | 선정 이유 | 상관관계 |
|------|---------|------------|------------|
| tenure | 0.12 | 고객 충성도 기본 지표 | 독립적 |
| MonthlyCharges | 0.11 | 수익성 직접 지표 | 요금 관련 대표성 |
| 고객_위험도 | 0.09 | 종합 위험 지표 | 독립적 |
| 서비스당_요금 | 0.12 | 효율성 지표 | 요금과 약한 상관관계 |
| 총_서비스_수 | 0.08 | 서비스 활용도 | 독립적 |
| 인터넷_서비스_등급 | 0.07 | 서비스 수준 | 독립적 |

![image](https://github.com/user-attachments/assets/872c29ba-c61c-4954-b17b-35454a5dd47b)

> Random Forest 기반 특성 중요도 시각화

---

## 3. 모델 개발 과정

### 3.1 모델 발전 과정
| 모델 | 정확도 | ROC-AUC | 이탈고객<br>Recall | 이탈고객<br>Precision | F1-score | 과적합<br>(차이) |
|------|---------|----------|-------------------|---------------------|-----------|----------------|
| Base | 0.80 | 0.8336 | 0.52 | 0.64 | 0.57 | - |
| SMOTE | 0.75 | 0.8279 | 0.74 | 0.52 | 0.61 | - |
| Tuned | 0.74 | 0.8017 | 0.64 | 0.51 | 0.57 | 0.0934 |
| Tuned V2 | 0.75 | 0.8276 | 0.73 | 0.52 | 0.61 | 0.0541 |

![모델 성능 비교](./images/model_performance_comparison.png)
> 각 모델별 주요 성능 지표 비교

### 3.2 하이퍼파라미터 최적화
python
최종 모델 파라미터
final_params = {
'n_estimators': 400,
'max_depth': 10,
'min_samples_split': 5,
'min_samples_leaf': 2,
'max_features': 'sqrt',
'class_weight': 'balanced'
}


![하이퍼파라미터 튜닝 과정](./images/hyperparameter_tuning.png)
> 주요 하이퍼파라미터별 성능 영향 분석

### 3.3 교차 검증 결과
| 폴드 | ROC-AUC | 정확도 | Recall | Precision |
|------|---------|---------|---------|------------|
| 1 | 0.8711 | 0.76 | 0.71 | 0.53 |
| 2 | 0.8834 | 0.75 | 0.72 | 0.51 |
| 3 | 0.8983 | 0.77 | 0.74 | 0.54 |
| 4 | 0.8801 | 0.75 | 0.73 | 0.52 |
| 5 | 0.8781 | 0.76 | 0.72 | 0.53 |
| 평균 | 0.8822 | 0.76 | 0.72 | 0.53 |
| 표준편차 | 0.0101 | 0.01 | 0.01 | 0.01 |

![교차 검증 결과](./images/cross_validation_results.png)
> 교차 검증 성능 분포 및 안정성 분석

---

## 4. 결과 및 시사점

### 4.1 모델 성능 최종 평가
| 평가 지표 | 초기 모델 | 최종 모델 | 개선율 | 비고 |
|-----------|------------|------------|---------|------|
| 전체 정확도 | 0.80 | 0.75 | -6.25% | 의도된 trade-off |
| ROC-AUC | 0.8336 | 0.8276 | -0.72% | 안정성 확보 |
| 이탈고객 Recall | 0.52 | 0.73 | +40.38% | 큰 폭 개선 |
| 이탈고객 Precision | 0.64 | 0.52 | -18.75% | 수용 가능 수준 |
| F1-score | 0.57 | 0.61 | +7.02% | 전반적 개선 |
| 과적합(차이) | - | 0.0541 | - | 안정적 수준 |

![최종 성능 평가](./images/final_performance_evaluation.png)
> 주요 성능 지표의 개선 추이 및 trade-off 관계

### 4.2 특성 중요도 최종 분석
| 특성 | 초기 중요도 | 최종 중요도 | 변화 | 시사점 |
|------|-------------|--------------|-------|---------|
| 고객_위험도 | 0.2488 | 0.2848 | ⬆️ | 위험 예측력 향상 |
| tenure | 0.2402 | 0.2258 | ⬇️ | 안정적 영향력 |
| 서비스당_요금 | 0.2278 | 0.1904 | ⬇️ | 과적합 감소 |
| MonthlyCharges | 0.1847 | 0.1664 | ⬇️ | 영향력 조정 |
| 인터넷_서비스_등급 | 0.0684 | 0.0934 | ⬆️ | 중요성 부각 |
| 총_서비스_수 | 0.0300 | 0.0392 | ⬆️ | 보조 지표화 |

![특성 중요도 변화](./images/feature_importance_changes.png)
> 모델 발전 과정에서의 특성 중요도 변화 추이

### 4.3 비즈니스 인사이트
1. 고객 유지 전략
   ```python
   high_risk_segments = {
       'contract': 'Month-to-month',
       'payment': 'Electronic check',
       'service': 'Fiber optic',
       'tenure': '< 12 months'
   }
   ```

2. 서비스 개선 우선순위
   | 서비스 영역 | 현재 이탈률 | 개선 목표 | 주요 조치 |
   |-------------|-------------|------------|-----------|
   | Fiber optic | 41.9% | 30% | 품질 개선 |
   | 전자 결제 | 45.3% | 25% | 시스템 안정화 |
   | 부가 서비스 | 41.8% | 20% | 혜택 강화 |

![서비스 개선 영향 예측](./images/service_improvement_impact.png)
> 서비스 개선에 따른 이탈률 감소 시뮬레이션

---

## 5. 기술 스택 및 향후 계획

### 5.1 개발 환경
| 분야 | 도구 | 버전 | 용도 |
|------|------|------|------|
| 언어 | Python | 3.8+ | 전체 개발 |
| ML | scikit-learn | 0.24.2 | 모델 개발 |
| | imbalanced-learn | 0.8.1 | SMOTE 적용 |
| 데이터 처리 | pandas | 1.3.3 | 데이터 전처리 |
| | numpy | 1.21.2 | 수치 연산 |
| 시각화 | matplotlib | 3.4.3 | 기본 시각화 |
| | seaborn | 0.11.2 | 고급 시각화 |

![개발 파이프라인](./images/development_pipeline.png)
> 전체 개발 프로세스 및 도구 활용 흐름도

### 5.2 향후 개선 로드맵
| 단계 | 계획 | 우선순위 | 예상 효과 |
|------|------|----------|------------|
| 1단계 | 앙상블 모델 도입 | 높음 | 성능 향상 |
| | 시계열 특성 추가 | 높음 | 예측력 개선 |
| 2단계 | 실시간 예측 시스템 | 중간 | 운영 효율화 |
| | 자동 모니터링 | 중간 | 안정성 확보 |
| 3단계 | 딥러닝 모델 검토 | 낮음 | 잠재력 탐색 |

![개선 로드맵](./images/improvement_roadmap.png)
> 단계별 개선 계획 및 기대 효과

---

## 📝 참고 자료
- [데이터 전처리 상세 보고서](data_preprocessing_report.md)
- [특성 선택 보고서](feature_selection_report.md)
- [모델 평가 보고서](model_evaluation_report.md)
- [코드 저장소](https://github.com/username/project)

## 📊 시각화 자료
- [./images/](./images/) 디렉토리에 모든 시각화 자료 저장
- 주요 시각화:
  * data_distribution.png
  * feature_correlation.png
  * model_performance_comparison.png
  * feature_importance_changes.png
  * service_improvement_impact.png



--- 

## 한 줄 회고
- 김현수:
- 박현준:
- 전서빈:
- 정소열:
- 조현정:
- 신민주:
