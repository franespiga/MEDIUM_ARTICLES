FROM python:3.7-slim

RUN mkdir /streamlit

COPY requirements.txt /streamlit
COPY books_info.csv /streamlit

WORKDIR /streamlit

RUN pip install -r requirements.txt

COPY . /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "ui.py"]