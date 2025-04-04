import logging
import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from typing import List, Optional
import os

# Configuração básica de logging (pode ser compartilhada ou configurada por quem usa o utils)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Carregamento de Stopwords ---
stop_words_pt: List[str] = []
try:
    stop_words_pt_set = set(stopwords.words('portuguese'))
    # Garantir minúsculas, embora stopwords do NLTK geralmente já estejam
    stop_words_pt = [word.lower() for word in stop_words_pt_set]
    logging.info("Stopwords em português carregadas do NLTK.")
except LookupError:
    logging.error("Recurso 'stopwords' do NLTK não encontrado. Tentando baixar...")
    try:
        nltk.download('stopwords')
        stop_words_pt_set = set(stopwords.words('portuguese'))
        stop_words_pt = [word.lower() for word in stop_words_pt_set]
        logging.info("Download de 'stopwords' concluído e stopwords carregadas.")
    except Exception as e:
        logging.error(
            f"Falha ao baixar 'stopwords' do NLTK: {e}. "
            "Verifique sua conexão com a internet ou configurações de firewall. "
            "As stopwords em português não serão usadas, o que pode afetar a qualidade da clusterização."
        )
        # Mantém stop_words_pt como lista vazia
except Exception as e:
    logging.error(
        f"Erro inesperado ao carregar stopwords: {e}. "
        "As stopwords em português não serão usadas, o que pode afetar a qualidade da clusterização."
    )
    # Mantém stop_words_pt como lista vazia

# --- Funções Auxiliares ---

def calcular_e_plotar_cotovelo(X: csr_matrix, limite_k: int, n_init: int = 1) -> Optional[List[float]]:
    """
    Calcula as inércias para diferentes valores de K e plota o gráfico do método do cotovelo.

    Args:
        X (csr_matrix): Matriz TF-IDF dos dados.
        limite_k (int): Número máximo de clusters (K) a testar.
        n_init (int): Número de inicializações do K-Means.

    Returns:
        Optional[List[float]]: Lista de inércias calculadas, ou None se não houver dados.
    """
    logging.info("Calculando inércias para o método do cotovelo...")
    inercias = []
    # Garante que limite_k não seja maior que o número de amostras
    k_max = min(limite_k, X.shape[0])
    if k_max < limite_k:
        logging.warning(f"Limite K ({limite_k}) é maior que o número de amostras ({X.shape[0]}). Usando K máximo = {k_max}.")
    if k_max == 0:
        logging.error("Não há amostras para calcular o método do cotovelo.")
        plt.figure(figsize=(10, 6))
        plt.title('Método do Cotovelo - Nenhuma amostra encontrada')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Inércia (WCSS)')
        plt.text(0.5, 0.5, 'Não há dados para processar', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.grid(True)
        plt.show()
        return None # Retorna None se não há amostras

    k_range = range(1, k_max + 1)
    for k in k_range:
        # Adiciona tratamento para n_init > número de amostras se k=1
        current_n_init = n_init
        if k == 1 and n_init > X.shape[0]:
            logging.warning(f"n_init ({n_init}) é maior que o número de amostras ({X.shape[0]}) para K=1. Usando n_init=1.")
            current_n_init = 1
        elif n_init <= 0:
            logging.warning(f"n_init ({n_init}) deve ser positivo. Usando n_init=1.")
            current_n_init = 1

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=current_n_init)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)
        logging.debug(f"Inércia para K={k}: {kmeans.inertia_}")

    logging.info("Gerando gráfico do método do cotovelo...")
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inercias, marker='o')
    plt.title('Método do Cotovelo para Escolha de K')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia (WCSS)')
    # Garante que xticks funcione mesmo com k_max=1
    if k_max > 0:
        plt.xticks(k_range)
    plt.grid(True)
    plt.show()
    return inercias


def salvar_dataframe_csv(df: pd.DataFrame, nome_arquivo: str) -> bool:
    """
    Salva um DataFrame em um arquivo CSV.

    Args:
        df (pd.DataFrame): O DataFrame a ser salvo.
        nome_arquivo (str): O caminho do arquivo CSV de saída.

    Returns:
        bool: True se o salvamento for bem-sucedido, False caso contrário.
    """
    logging.info(f"Tentando salvar DataFrame em '{nome_arquivo}'...")
    try:
        df.to_csv(nome_arquivo, index=False, encoding='utf-8-sig')
        logging.info(f"DataFrame salvo com sucesso em '{nome_arquivo}'.")
        return True
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo CSV '{nome_arquivo}': {e}")
        return False


def salvar_amostras_excel(df: pd.DataFrame, nome_coluna_cluster: str, num_clusters: int, nome_arquivo: str) -> bool:
    """
    Gera e salva amostras (até 10 por cluster) de um DataFrame em um arquivo Excel.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados e a coluna de cluster.
        nome_coluna_cluster (str): O nome da coluna que identifica o cluster.
        num_clusters (int): O número de clusters esperado (usado para iterar).
        nome_arquivo (str): O caminho do arquivo Excel de saída.

    Returns:
        bool: True se o salvamento for bem-sucedido (ou se não houver amostras), False em caso de erro.
    """
    logging.info(f"Tentando gerar e salvar amostras (até 10 por cluster) em '{nome_arquivo}'...")
    resultados = pd.DataFrame()
    if nome_coluna_cluster not in df.columns:
        logging.error(f"Coluna de cluster '{nome_coluna_cluster}' não encontrada no DataFrame ao tentar salvar amostras.")
        return False

    try:
        # Tenta importar openpyxl ANTES de gerar os resultados
        try:
            import openpyxl
        except ImportError:
            logging.error("A biblioteca 'openpyxl' é necessária para salvar arquivos .xlsx. Instale-a com 'pip install openpyxl'")
            return False

        actual_clusters = df[nome_coluna_cluster].unique()
        valid_num_clusters = len(actual_clusters) # Apenas para informação, não usado diretamente abaixo

        for cluster_id in range(num_clusters): # Itera até o K solicitado
            if cluster_id not in actual_clusters:
                logging.warning(f"Cluster ID {cluster_id} não foi gerado. Nenhuma amostra será retirada.")
                continue

            df_cluster = df[df[nome_coluna_cluster] == cluster_id]
            tamanho_amostra = min(10, len(df_cluster))
            if tamanho_amostra > 0:
                amostra_cluster = df_cluster.sample(tamanho_amostra, random_state=42)
                amostra_cluster.insert(0, 'cluster_original_id', cluster_id)
                resultados = pd.concat([resultados, amostra_cluster], ignore_index=True)
            else:
                logging.warning(f"Cluster {cluster_id} está vazio, nenhuma amostra será retirada.")

        if not resultados.empty:
            resultados.to_excel(nome_arquivo, index=False)
            logging.info(f"Amostras salvas com sucesso em '{nome_arquivo}'.")
        else:
            logging.warning("Nenhuma amostra foi gerada. Arquivo Excel não será criado.")
            # Consideramos sucesso, pois não houve erro de IO.

    except Exception as e:
        logging.error(f"Falha ao gerar ou salvar o arquivo Excel de amostras '{nome_arquivo}': {e}")
        return False
    return True
