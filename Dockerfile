FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Define un argumento de compilación para el token.
ARG HUGGINGFACE_TOKEN
# Asigna el argumento a una variable de entorno.
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

WORKDIR /app

RUN apt update && apt upgrade -y && apt install git -y
RUN pip install llm-swarm
#RUN git clone https://github.com/huggingface/llm-swarm.git && mv llm-swarm/* /app/

COPY . .

RUN mkdir /.cache
RUN chmod -R 777 /.cache

CMD ["python", "examples/hello_world_vllm.py"]
