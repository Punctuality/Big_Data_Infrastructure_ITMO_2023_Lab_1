# Лабораторная работа №1

**Дисциплина:** "Инфраструктура больших данных"

**Выполнил:** Федоров Сергей, M4150 

**Вариант:** Fake-News Classification (#22)

**Репозиторий с исходным кодом:** [Репозиторий](https://github.com/Punctuality/Big_Data_Infrastructure_ITMO_2023_Lab_1)

**Статус автоматической проверки:** [![CI/CD Pipeline](https://github.com/Punctuality/Big_Data_Infrastructure_ITMO_2023_Lab_1/actions/workflows/ci-cd.yml/badge.svg?branch=main)](https://github.com/Punctuality/Big_Data_Infrastructure_ITMO_2023_Lab_1/actions/workflows/ci-cd.yml)

## Описания выполнения:

1. Выполнение ML задачи
2. Конвертация исходного кода из iPython Notebook в отдельные скрипт-файлы
3. Написание тестов
4. Настройка DVC для исходных данных и обученных весов модели
5. Контейнеризация программы
6. Добавление параметров программы через конфиг файл
7. Создание CI/CD pipeline'a

## Выполнение ML задачи

По варианту задания необходимо было выполнить классификацию того является некоторая новость фейковой или нет. Исходные данные находятся в данном [Kaggle датасете](https://www.kaggle.com/c/fake-news/overview). 

После препроцессеинга и стемминга использовалась следующая модель:

```python
class FakeNewsClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)#, padding_idx=0)
        self.dropout_1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.dropout_2 = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_dim, 64)
        self.dropout_3 = nn.Dropout(dropout)
        self.out = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout_1(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout_2(x)
        x = t.relu(self.dense(x))
        x = self.dropout_3(x)
        x = t.sigmoid(self.out(x))
        return x
```

Корпус создавался следующим образом:

```python
#We will be using Stemming here
#Stemming map words to their root forms
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import time

def measure_time(callable, name):
    def wrapper():
        start = time.time()
        result = callable()
        end = time.time()
        print(f"Time elapsed ({name}): {end - start}")
        return result

    return wrapper

def construct_corpus(data):
    corpus = []
    for i in tqdm(range(len(data))):
        review = re.sub('[^a-zA-Z]',' ', data['total'].iloc[i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

from typing import Union, Iterable
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

corpus = construct_corpus(X_train)
corpus_test = construct_corpus(X_test)
corpus_val = construct_corpus(X_val)

tokens = [tokenizer(doc) for doc in corpus]
tokens_val = [tokenizer(doc) for doc in corpus_val]
tokens_test = [tokenizer(doc) for doc in corpus_test]
voc = build_vocab_from_iterator(tokens + tokens_val, max_tokens=voc_size, specials=["<unk>"])
voc.set_default_index(voc["<unk>"])

voc_tokens = [t.tensor(voc(token), dtype=t.int64) for token in tokens]
voc_tokens_val = [t.tensor(voc(token), dtype=t.int64) for token in tokens_val]
voc_tokens_test = [t.tensor(voc(token), dtype=t.int64) for token in tokens_test]

from tqdm.auto import tqdm

max_len = 25

def padding_indexes(tokens):
    embedding = []
    for token in tqdm(tokens):
        embedding.append(nn.ConstantPad1d((max_len - len(token), 0), 0)(t.tensor(token, dtype=t.int64)))
    return t.stack(embedding)
    
padded_tokens = padding_indexes(voc_tokens)
padded_tokens_val = padding_indexes(voc_tokens_val)
padded_tokens_test = padding_indexes(voc_tokens_test)
```

Training Loop в данном случае выглядит так:

```python
def accuracy(model, test_dataloader):
    model.eval()
    with t.no_grad():
        sum_acc = 0
        for batch in test_dataloader:
            X, y = batch
            y_pred = model(X)
            y_pred = y_pred.squeeze()
            sum_acc += t.sum((y_pred > 0.5) == y)
        return sum_acc / (len(test_dataloader) * test_dataloader.batch_size)

def train(model, dataloader, test_dataloader, loss, optimizer, epochs):
    print(f"Staring acc_train: {accuracy(model, dataloader)} acc_val: {accuracy(model, test_dataloader)}")

    for epoch in range(epochs):
        model.train()
        l = 0
        for batch in dataloader:
            X, y = batch
            optimizer.zero_grad()
            y_pred = model(X)
            y_pred = y_pred.squeeze()
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()

        train_acc = accuracy(model, dataloader)
        val_acc = accuracy(model, test_dataloader)

        print(f"Epoch {epoch+1} loss: {l.item()} acc_train: {train_acc} acc_val: {val_acc}")
```

Модель обучалась на MPS device. Код для выбора оптимального устройства:

```python
def optimal_device():
    log.debug("Checking for CUDA availability")
    if t.cuda.is_available():
        log.debug("CUDA is available")
        return t.device('cuda')
    else:
        log.debug("CUDA is NOT available")
        try:
            dev = t.device('mps')
            log.debug("Fallbacking to to use MPS")
            return dev
        except Exception as _:
            log.debug("Fallbacking to CPU")
            return t.device('cpu')
```

Логи обучения:

```
Staring acc_train: 0.50390625 acc_val: 0.48750001192092896
Epoch 1 loss: 0.19220766425132751 acc_train: 0.9737379550933838 acc_val: 0.96875
Epoch 2 loss: 0.0719328299164772 acc_train: 0.9912259578704834 acc_val: 0.9872596263885498
Epoch 3 loss: 0.006107015535235405 acc_train: 0.9958533644676208 acc_val: 0.9908654093742371
Epoch 4 loss: 0.009487172588706017 acc_train: 0.9964542984962463 acc_val: 0.9911057949066162
Epoch 5 loss: 0.008560552261769772 acc_train: 0.997776448726654 acc_val: 0.9923076629638672
```

## Конвертация исходного кода из iPython Notebook в отдельные скрипт-файлы

После успешного создание и обучения модели, код данного notebook'a был распределен по следующим файлам:

1. `corpus.py` - препроцессинг датасета и загрузка корпуса 
2. `dataset.py` - определение Torch датасета для данной задачи
3. `device_config.py` - конфигурация оптимального вычислительного устройства
4. `eval.py` - запуск модели на тестовых данных
5. `main.py` - агрегатор остальных файлов, входная точка проекта
6. `model.py` - определение Torch модели для данной задачи
7. `preprocessing.py` - дополнительный препроцессинг данных
8. `train.py` - запуск обучения модели, в случае необходимости

## Написание тестов

Поскольку тестировать конкретные входные и выходные данные модели - выглядит как сомнительная идея, было решено написать следующие тесты:

```python
class TestPreprocessingSpec:

    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one."
    ]

    def test_tokenize(self):
        tokens = tokenize(self.corpus)
        assert len(tokens) == 3
        assert tokens[0] == ['this', 'is', 'the', 'first', 'document', '.']
        assert tokens[1] == ['this', 'document', 'is', 'the', 'second', 'document', '.']
        assert tokens[2] == ['and', 'this', 'is', 'the', 'third', 'one', '.']

    def test_build_vocab(self):
        tokens = tokenize(self.corpus)
        vocab = build_vocab(tokens, 10)
        assert len(vocab) == 10
        assert [vocab[token] for token in [
            '<unk>', '.', 'document', 'is', 'the', 'this', 'and', 'first', 'one', 'second'
            ]] == list(range(10))

    def test_vocab_tokens(self):
        tokens = tokenize(self.corpus)
        vocab = build_vocab(tokens, 10)
        vb_tokens = vocab_tokens(vocab, tokens)
        assert len(vb_tokens) == 3
        assert vb_tokens[0].tolist() == [5, 3, 4, 7, 2, 1]
        assert vb_tokens[1].tolist() == [5, 2, 3, 4, 9, 2, 1]
        assert vb_tokens[2].tolist() == [6, 5, 3, 4, 0, 8, 1]

    def test_padding_indexes(self):
        tokens = tokenize(self.corpus)
        vocab = build_vocab(tokens, 10)
        vb_tokens = vocab_tokens(vocab, tokens)
        padded = padding_indexes(vb_tokens, 10)
        assert len(padded) == 3
        assert padded[0].tolist() == [0, 0, 0, 0, 5, 3, 4, 7, 2, 1]
        assert padded[1].tolist() == [0, 0, 0, 5, 2, 3, 4, 9, 2, 1]
        assert padded[2].tolist() == [0, 0, 0, 6, 5, 3, 4, 0, 8, 1]

class TestModelSpec:

    model = FakeNewsClassifier(5000, 40, 100, 2, 0.2)

    def test_inference_shape(self):
        n = np.random.randint(1, 100)
        x = t.randint(0, 5000, (n, 25))
        y = self.model(x)
        assert y.shape == (n, 1)

    def test_nondet_train(self):
        x = t.randint(0, 5000, (10, 25))

        self.model.train()
        y_1 = self.model(x)
        y_2 = self.model(x)
        assert y_1.shape == y_2.shape
        assert not t.equal(y_1, y_2)

    def test_det_eval(self):
        x = t.randint(0, 5000, (10, 25))

        self.model.eval()
        y_1 = self.model(x)
        y_2 = self.model(x)
        assert y_1.shape == y_2.shape
        assert t.equal(y_1, y_2)
```

## Настройка DVC для исходных данных и обученных весов модели

В данном случае я использую DVC с remote хостом в виде S3, которое я в моем случае развернул в ввиде minio. Такой код позволяет настроить удаленный репозиторий и спулить все необходимые данные.

```bash
dvc remote modify --local remote_s3 endpointurl ${{ secrets.DVC_REMOTE_URL }}
dvc remote modify --local remote_s3 access_key_id ${{ secrets.DVC_REMOTE_ACCESS_KEY_ID }}
dvc remote modify --local remote_s3 secret_access_key ${{ secrets.DVC_REMOTE_SECRET_ACCESS_KEY }}

dvc pull -r remote_s3
```

Стоит отметить, что так как это первая лабораторная работа, много других следовали за ней, а следовательно и сетап множества дополнительных сервисом на моем VPS - гарантия того что S3 работает до сих пор, не предоставляется. Однако, можно проверить последний build статус, который использовал это S3.

## Контейнеризация программы

Программу я контейнеризоровал с помощью Docker и ниже предоставлен Dockerfile:

```Dockerfile
FROM python:3.11

WORKDIR /app

COPY *requirements.txt ./

RUN mkdir /app/data/
RUN mkdir /app/result/

RUN pip install -r requirements.txt

RUN mkdir /app/src/
RUN mkdir /app/test/

COPY src/* /app/src/
COPY test/* /app/test/
COPY docker_config.ini ./config.ini

VOLUME /app/data
VOLUME /app/result

CMD ["python", "src/main.py", "--config_path", "/app/config.ini"]
```

Таким docker-compose, я поднимал локальную среду:

```yaml
version: '3'

services:
  minio:
    container_name: minio
    image: minio/minio
    expose:
      - "9000"
      - "9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_storage:/data
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    command: server --console-address ":9001" --address ":9000" /data

  fake-news-classifier:
    image: fake_news_classificator:latest
    container_name: fake-news-classfier
    volumes:
      - ./data:/app/data
      - ./tmp/result:/app/result

volumes:
  minio_storage:
```

## Добавление параметров программы через конфиг файл

Все параметры передаются в модель посредством `.ini` файла. Для Docker image и для локального запуска параметры естественно разные.

## Создание CI/CD pipeline'a

CI/CD пайплайн настраивался на платформе Github (Github Actions), в виду своей достаточности, бесплатного сервиса и возможности использования интегрированных инструментов платформы на которой хостится репозиторий кода данного проекта. Pipeline выглядит следующим образом:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  run_tests:
    runs-on: ubuntu-latest

    env:
      PYTHONPATH: ${{ github.workspace }}/

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.11

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run tests
      run: pytest --cov=src

  build_docker_image:
    needs: run_tests
    runs-on: ubuntu-latest
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Docker
      uses: docker/setup-qemu-action@v2

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_HUB_USERNAME }}
        password: ${{ secrets.DOCKER_HUB_PASSWORD }}

    - name: Get tag or commit
      id: tag_or_commit
      run: |
        if [[ $GITHUB_REF == refs/tags/* ]]; then
          echo "::set-output name=version::${GITHUB_REF#refs/tags/}"
        else
          echo "::set-output name=version::${GITHUB_SHA}"
        fi

    - name: Build and push Docker image
      uses: docker/build-push-action@v3
      with:
        context: .
        push: true
        tags: |
          ${{ secrets.DOCKER_HUB_USERNAME }}/fake_news_classificator:${{ steps.tag_or_commit.outputs.version }}
          ${{ secrets.DOCKER_HUB_USERNAME }}/fake_news_classificator:latest

  run_docker_image:
    needs: build_docker_image
    runs-on: ubuntu-latest
    env:
      LOG_LEVEL: DEBUG
      
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: 3.11

      - name: Install DVC
        run: pip install dvc dvc-s3

      - name: Prepare DVC for remote storage
        run: |
          dvc remote modify --local remote_s3 endpointurl ${{ secrets.DVC_REMOTE_URL }}
          dvc remote modify --local remote_s3 access_key_id ${{ secrets.DVC_REMOTE_ACCESS_KEY_ID }}
          dvc remote modify --local remote_s3 secret_access_key ${{ secrets.DVC_REMOTE_SECRET_ACCESS_KEY }}

      - name: Pull data and model from DVC remote storage
        run: dvc pull -r remote_s3

      - name: Set up Docker
        uses: docker/setup-qemu-action@v2

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_HUB_USERNAME }}
          password: ${{ secrets.DOCKER_HUB_PASSWORD }}

      - name: Docker pull image
        run: |
          docker pull ${{ secrets.DOCKER_HUB_USERNAME }}/fake_news_classificator:latest
          docker tag ${{ secrets.DOCKER_HUB_USERNAME }}/fake_news_classificator:latest fake_news_classificator:latest 

      - name: Run Docker-Compose service
        run: docker-compose up fake-news-classifier
```
