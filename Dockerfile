# Base environment
FROM python:3.8.15

# Setting working dir as container dir
WORKDIR /app

# Copy files to container dir
COPY . /app

# Upgrade pip and install reqs
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run Flask app
CMD gunicorn --bind 0.0.0.0:5005 --timeout=150 app:app

EXPOSE 5005