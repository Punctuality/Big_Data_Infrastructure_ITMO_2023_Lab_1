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