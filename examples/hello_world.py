import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient, login
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio

# Cargar las variables de entorno del archivo .env
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("No se encontró HUGGINGFACE_TOKEN en las variables de entorno.")

# Iniciar sesión en Hugging Face
login(HUGGINGFACE_TOKEN)

tasks = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the formula for water?"
]

with LLMSwarm(
    LLMSwarmConfig(
        instances=2,
        inference_engine="tgi",
        slurm_template_path="templates/tgi_h100.template.slurm",
        load_balancer_template_path="templates/nginx.template.conf",
    )
) as llm_swarm:
    # Se pasa el token al cliente para autenticación
    client = AsyncInferenceClient(model=llm_swarm.endpoint, token=HUGGINGFACE_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.add_special_tokens({
        "sep_token": "",
        "cls_token": "",
        "mask_token": "",
        "pad_token": "[PAD]"
    })

    async def process_text(task):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task},
            ],
            tokenize=False,
        )
        return await client.text_generation(
            prompt=prompt,
            max_new_tokens=200,
        )

    async def main():
        results = await tqdm_asyncio.gather(*(process_text(task) for task in tasks))
        df = pd.DataFrame({"Task": tasks, "Completion": results})
        print(df)

    asyncio.run(main())
