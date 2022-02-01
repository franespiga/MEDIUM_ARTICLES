FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /embeddings

COPY requirements.txt /embeddings

WORKDIR /embeddings

RUN pip install -r requirements.txt 

COPY . /embeddings

EXPOSE 8502

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8502"]
