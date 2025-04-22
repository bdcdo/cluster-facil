"""
Funções utilitárias para rotulação automática de clusters usando modelos de linguagem (ex: GPT-4.1-nano via OpenAI API).
"""
import os
from typing import List, Optional, Dict
from openai import OpenAI
import json

def gerar_rotulo_cluster(
    cluster_texts: List[str],
    model: str = "gpt-4.1-nano",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """
    Gera um rótulo (nome/tema) para um cluster de textos usando a OpenAI Responses API.
    Args:
        cluster_texts (List[str]): Lista de textos pertencentes ao cluster.
        model (str): Nome do modelo OpenAI a ser utilizado. Padrão: 'gpt-4.1-nano'.
        api_key (Optional[str]): Chave da API OpenAI. Se não fornecida, busca em OPENAI_API_KEY.
        temperature (float): Temperatura do modelo (criatividade).
    Returns:
        str: Rótulo sugerido para o cluster.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("É necessário fornecer uma chave de API OpenAI via argumento ou variável de ambiente OPENAI_API_KEY.")

    client_kwargs = {"api_key": api_key}
    client = OpenAI(**client_kwargs)

    sample_texts = cluster_texts[:10]
    # Delimita cada texto com <sampleN>...</sampleN>
    joined_texts = "\n\n".join(f"<sample{i+1}>\n{t}\n</sample{i+1}>" for i, t in enumerate(sample_texts))
    prompt = (
        "Dado o seguinte conjunto de textos, gere um rótulo curto (tema) que represente o cluster. "
        "O rótulo deve ser claro, conciso e descritivo."
    )
    # Utiliza a Responses API mais recente com formato estruturado
    resp_format = {
        "format": {
            "type": "json_schema",
            "name": "rotulo",
            "schema": {
                "type": "object",
                "properties": {
                    "rotulo": {
                        "type": "string",
                        "description": "Rótulo curto, claro e descritivo para o cluster de textos."
                    }
                },
                "required": ["rotulo"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    response = client.responses.create(
        model=model,
        instructions=prompt,
        temperature=temperature,
        input=f"Textos do cluster:\n{joined_texts}\n\nRótulo:",
        text=resp_format
    )
    label_obj = json.loads(response.output_text)
    return label_obj["rotulo"]

def refinar_rotulos_clusters(
    cluster_samples: Dict,
    model: str = "gpt-4.1-nano",
    api_key: str = None,
    temperature: float = 0.0,
) -> Dict:
    """
    Usa o LLM para revisar, padronizar e possivelmente agrupar rótulos de clusters, a partir de amostras e rótulos iniciais.
    Args:
        cluster_samples (dict): Dict {cluster_id: {"label": rótulo, "examples": [str, ...]}}
        model (str): Nome do modelo OpenAI.
        api_key (str): Chave da API OpenAI.
        temperature (float): Temperatura do modelo.
    Returns:
        dict: Dicionário {cluster_id: rótulo_final}
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("É necessário fornecer uma chave de API OpenAI via argumento ou variável de ambiente OPENAI_API_KEY.")

    client_kwargs = {"api_key": api_key}
    client = OpenAI(**client_kwargs)

    # Monta prompt e input estruturado
    prompt = (
        "Você receberá exemplos de clusters, cada um com um rótulo sugerido e algumas amostras de textos.\n"
        "Sua tarefa é:\n"
        "- Unificar rótulos semelhantes se fizer sentido,\n"
        "- Sugerir nomes mais claros e concisos para cada grupo,\n"
        "- Retornar um dicionário JSON com o id do cluster e o novo rótulo.\n\n"
        "Exemplo de entrada:\n"
        "Cluster 0 - Rótulo inicial: 'Esportes'\n<sample1>Texto exemplo</sample1>\n<sample2>Texto exemplo</sample2>\n\n"
        "Cluster 1 - Rótulo inicial: 'Futebol'\n<sample1>Texto exemplo</sample1>\n<sample2>Texto exemplo</sample2>\n\n"
        "Agora, siga o mesmo padrão para os clusters abaixo:\n"
    )
    clusters_str = ""
    for cid, info in cluster_samples.items():
        label = info["label"]
        examples = "\n\n".join(f"<sample{i+1}>\n{t}\n</sample{i+1}>" for i, t in enumerate(info["examples"]))
        clusters_str += f"Cluster {cid} - Rótulo inicial: '{label}'\n{examples}\n\n"
    input_text = clusters_str + "Retorne APENAS um JSON no formato: {\"cluster_id\": \"novo rótulo\", ...}"
    resp_format = {
        "format": {
            "type": "json_schema",
            "name": "refino_rotulos",
            "schema": {
                "type": "object",
                "properties": {
                    "clusters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "cluster_id": {"type": "integer"},
                                "label": {"type": "string"}
                            },
                            "required": ["cluster_id", "label"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["clusters"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    response = client.responses.create(
        model=model,
        instructions=prompt,
        temperature=temperature,
        input=input_text,
        text=resp_format
    )
    refined_labels = json.loads(response.output_text)
    return refined_labels
