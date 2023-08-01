FROM python:3.8.15

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD gunicorn --bind 0.0.0.0:5005 --timeout=150 app:app

EXPOSE 5005