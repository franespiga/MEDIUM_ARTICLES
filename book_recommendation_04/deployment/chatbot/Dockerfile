FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /chatbot

COPY requirements.txt /chatbot

WORKDIR /chatbot

RUN pip install -r requirements.txt 

COPY . /chatbot

EXPOSE 8505

CMD ["python", "app.py"]