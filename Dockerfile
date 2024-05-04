FROM python:3.10-bookworm

WORKDIR /app

RUN pip install -U pip &&\
    pip install torch torchvision  --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

COPY ./requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir 

COPY . .

CMD ["python", "app.py"]

