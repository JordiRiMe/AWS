FROM public.ecr.aws/lambda/python:3.7

COPY requirements.txt .

RUN pip install -r requirements.txt -t .

COPY word_embedding.py .

CMD ['word_embedding.word_embedding']
