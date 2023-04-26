import pandas as pd
import re

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

ps = PorterStemmer()
nltk.download('stopwords')


def read_dataframe(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.fillna('')

    return data


def load_corpus(data: pd.DataFrame) -> list[str]:
    data['total'] = data['title'] + ' ' + data['author']
    stopwords_eng = set(stopwords.words('english'))

    corpus = []
    for value in data['total']:
        review = re.sub('[^a-zA-Z]', ' ', value)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords_eng]
        review = ' '.join(review)
        corpus.append(review)

    return corpus
