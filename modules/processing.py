# modules/data_processing.py
import re
import pandas as pd
from konlpy.tag import Okt

# 형태소 분석기 초기화
okt = Okt()

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # NaN 값을 빈 문자열로 대체
    data['text'] = data['text'].fillna('')

    # 모든 값을 문자열로 변환
    data['text'] = data['text'].astype(str)

    # 숫자 및 특수문자 제거
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^가-힣a-zA-Z\s]', '', x))

    # 형태소 분석 및 어간 추출
    data['text'] = data['text'].apply(lambda x: ' '.join(okt.morphs(x, stem=True)))

    return data
