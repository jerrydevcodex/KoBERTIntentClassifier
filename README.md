# KoBERTIntentClassifier
KoreanIntentClassifier is a system that uses a BERT-based deep learning model to classify the intent of Korean text automatically.
# 한국어 문장 의도 예측 모델

이 프로젝트는 BERT를 사용하여 한국어 문장의 의도를 자동으로 예측하는 모델을 구축하는 것을 목표로 합니다. 다양한 사용자 입력 문장을 미리 정의된 범주로 분류하여, 응용 프로그램에서 활용할 수 있는 기반을 제공합니다.

## 프로젝트 구조

- `data/`: 훈련, 테스트, 증강 데이터셋이 포함된 디렉토리입니다.
  - `traindataset.csv`
  - `test_data.csv`
  - `augmented_data.csv`
- `notebooks/`: 데이터 전처리 및 모델 학습을 위한 Jupyter 노트북 파일이 포함되어 있습니다.
- `models/`: 학습된 BERT 모델이 저장되는 디렉토리입니다.
- `src/`: 주요 코드 파일들이 포함되어 있는 디렉토리입니다.
  - `preprocessing.py`: 데이터 전처리 관련 코드
  - `train.py`: 모델 학습 및 저장 코드
  - `predict.py`: 의도 예측 및 결과 저장 코드
  - `evaluate.py`: 모델 평가 코드
- `README.md`: 프로젝트에 대한 설명을 제공하는 문서입니다.

## 기능

- **데이터 전처리**: Okt 형태소 분석기를 사용하여 입력 문장을 형태소 단위로 분해하고 불필요한 문자와 공백을 제거합니다.
- **모델 학습**: BERT 모델을 활용하여 입력 문장의 의도를 학습합니다.
- **의도 예측**: 학습된 모델을 사용하여 입력된 문장의 의도를 예측합니다.
- **모델 평가**: 예측 결과를 평가하여 정확도, 정밀도, 재현율, F1 점수를 제공합니다.

## 설치 및 실행

### 요구사항

- Python 3.7 이상
- Hugging Face Transformers
- TensorFlow
- KoNLPy
- 기타 패키지: pandas, numpy, scikit-learn 등

### 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows의 경우 `venv\Scripts\activate`

# 필요한 패키지 설치
pip install -r requirements.txt
