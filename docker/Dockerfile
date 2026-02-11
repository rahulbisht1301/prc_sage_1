FROM python:3.9-slim

RUN pip install --no-cache-dir pandas scikit-learn joblib

COPY src /opt/program
WORKDIR /opt/program

ENV PYTHONUNBUFFERED=TRUE
