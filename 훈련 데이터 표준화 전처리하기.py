#https://machfam.com/16451
from sklearn.datasets import load_breast_cancer
import numpy as np

# 사이킷런에서 유방암 데이터 가져오기
cancer = load_breast_cancer()

# x축에 input 데이터 나열
x=cancer.data
# y축에 타겟 데이터 나열
y=cancer.target

# 훈련 데이터와 테스트 데이터 분류
x_train_all, x_test, y_train_all, y_test = \
  train_test_split(x,y,stratify=y,test_size=0.2,\
                   random_state=42)
# 훈련 데이터와 검증 데이터 분류
x_train, x_val, y_train, y_val = \
  train_test_split(x_train_all,y_train_all,stratify=y_train_all, \
                   test_size=0.2,\
                   random_state=42)  

# 객체 만들기
scaler = StandardScaler()

# 변환 규칙을 익히기
scaler.fit(x_train)

# 데이터를 표준화 전처리
x_train_scaled = scaler.transform(x_train)  
x_val_scaled = scaler.transform(x_val)     
