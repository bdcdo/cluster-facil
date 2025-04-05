# -*- coding: utf-8 -*-
"""Módulo de funções utilitárias para o pacote cluster_facil."""

# --- Importações ---
import logging
import os
from typing import List, Optional
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

from .validations import (
    validar_arquivo_existe,
    validar_dependencia_leitura,
    validar_formato_suportado,
    validar_coluna_existe,
    validar_inteiro_positivo,
    validar_dependencia
)

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Carregamento de Stopwords (Executado na importação do módulo) ---
stop_words_pt: List[str] = []
try:
    stop_words_pt_set = set(stopwords.words('portuguese'))
    stop_words_pt = [word.lower() for word in stop_words_pt_set]
    logging.info("Stopwords em português carregadas do NLTK.")
except LookupError:
    logging.warning("Recurso 'stopwords' do NLTK não encontrado. Tentando baixar...")
    try:
        nltk.download('stopwords')
        stop_words_pt_set = set(stopwords.words('portuguese'))
        stop_words_pt = [word.lower() for word in stop_words_pt_set]
        logging.info("Download de 'stopwords' concluído e stopwords carregadas.")
    except Exception as e:
        logging.error(
            f"Falha ao baixar 'stopwords' do NLTK: {e}. "
            "Verifique sua conexão ou firewall. Stopwords não serão usadas."
        )
except Exception as e:
    logging.error(
        f"Erro inesperado ao carregar stopwords: {e}. Stopwords não serão usadas."
    )

# --- Funções de Carregamento de Dados ---
def carregar_dados(caminho_arquivo: str, aba: Optional[str] = None) -> pd.DataFrame:
    """
    Carrega dados de um arquivo usando Pandas.

    Suporta CSV, Excel (.xlsx), Parquet e JSON.

    Args:
        caminho_arquivo (str): O caminho para o arquivo de dados.
        aba (Optional[str], optional): O nome ou índice da aba a ser lida se for um arquivo Excel.
                                       Se None, lê a primeira aba. Default é None.

    Returns:
        pd.DataFrame: O DataFrame carregado.

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado.
        ImportError: Se uma dependência necessária (ex: openpyxl, pyarrow) não estiver instalada.
        ValueError: Se o formato do arquivo não for suportado ou houver erro na leitura.
        Exception: Para outros erros inesperados durante o carregamento.
    """
    logging.info(f"Tentando carregar dados do arquivo: {caminho_arquivo}")
    validar_arquivo_existe(caminho_arquivo)
    _, extensao = os.path.splitext(caminho_arquivo)
    extensao = extensao.lower()
    validar_formato_suportado(extensao)
    validar_dependencia_leitura(extensao)

    try:
        if extensao == '.csv':
            df = pd.read_csv(caminho_arquivo)
        elif extensao == '.xlsx':
            df = pd.read_excel(caminho_arquivo, sheet_name=aba)
        elif extensao == '.parquet':
            df = pd.read_parquet(caminho_arquivo)
        elif extensao == '.json':
            df = pd.read_json(caminho_arquivo)

        logging.info(f"Arquivo {caminho_arquivo} carregado com sucesso. Shape: {df.shape}")
        return df
    except Exception as e:
        # Captura outros erros de leitura (ex: arquivo corrompido, JSON mal formatado)
        logging.error(f"Erro ao ler o arquivo {caminho_arquivo} (formato {extensao}): {e}")
        raise ValueError(f"Erro ao processar o arquivo {caminho_arquivo}: {e}")

# --- Funções de Análise e Plotagem ---
def calcular_e_plotar_cotovelo(X: csr_matrix, limite_k: int, n_init: int = 1, plotar: bool = True) -> Optional[List[float]]:
    """
    Calcula as inércias para diferentes valores de K e opcionalmente plota o gráfico do método do cotovelo.

    Args:
        X (csr_matrix): Matriz TF-IDF dos dados.
        limite_k (int): Número máximo de clusters (K) a testar.
        n_init (int): Número de inicializações do K-Means.
        plotar (bool, optional): Se True (padrão), exibe o gráfico do cotovelo usando `plt.show()`. Default True.

    Returns:
        Optional[List[float]]: Lista de inércias calculadas, ou None se não houver dados.
    """
    logging.info("Calculando inércias para o método do cotovelo...")
    # Valida n_init antes de começar usando a função genérica
    validar_inteiro_positivo('n_init', n_init)

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
        if plotar:
            plt.show()
        return None # Retorna None se não há amostras

    k_range = range(1, k_max + 1)
    for k in k_range:
        # Adiciona tratamento para n_init
        # Ajusta n_init apenas se for maior que as amostras para K=1 (caso especial do KMeans)
        # A validação de n_init > 0 já foi feita
        current_n_init = n_init
        if k == 1 and n_init > X.shape[0]:
            logging.warning(f"n_init ({n_init}) > amostras ({X.shape[0]}) para K=1. Usando n_init=1.")
            current_n_init = 1
        # Não precisa mais do elif n_init <= 0

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
    if k_max > 0:
        plt.xticks(k_range)
    plt.grid(True)
    if plotar:
        logging.info("Exibindo gráfico do método do cotovelo...")
        plt.show()
    else:
        logging.info("Gráfico do método do cotovelo não será exibido (plotar=False).")
    return inercias

# --- Funções de Preparação de Caminhos ---
def preparar_caminhos_saida(diretorio_saida: Optional[str], prefixo_saida: str, rodada_clusterizacao: int) -> dict[str, str]:
    """
    Prepara os caminhos completos para os arquivos de saída CSV e Excel.

    Cria o diretório de saída se ele não existir.

    Args:
        diretorio_saida (Optional[str]): Caminho da pasta onde salvar os arquivos. Se None, usa o diretório atual.
        prefixo_saida (str): Prefixo para os nomes dos arquivos de saída.
        rodada_clusterizacao (int): Número da rodada de clusterização atual (usado no nome do arquivo).

    Returns:
        dict[str, str]: Dicionário contendo 'caminho_csv' e 'caminho_excel' com os caminhos completos.

    Raises:
        OSError: Se houver um erro ao tentar criar o diretório de saída.
    """
    logging.info(f"Preparando caminhos de saída para a rodada {rodada_clusterizacao} com prefixo '{prefixo_saida}' e diretório '{diretorio_saida or '.'}'")

    prefixo_fmt = f"{prefixo_saida}_" if prefixo_saida else ""
    nome_base_csv = f"{prefixo_fmt}clusters_{rodada_clusterizacao}.csv"
    nome_base_excel = f"{prefixo_fmt}amostras_por_cluster_{rodada_clusterizacao}.xlsx"

    if diretorio_saida:
        try:
            os.makedirs(diretorio_saida, exist_ok=True)
            logging.info(f"Diretório de saída '{diretorio_saida}' verificado/criado.")
            caminho_csv = os.path.join(diretorio_saida, nome_base_csv)
            caminho_excel = os.path.join(diretorio_saida, nome_base_excel)
        except OSError as e:
            logging.error(f"Não foi possível criar ou acessar o diretório '{diretorio_saida}': {e}.")
            raise
    else:
        caminho_csv = nome_base_csv
        caminho_excel = nome_base_excel

    logging.info(f"Caminho final definido para CSV: {caminho_csv}")
    logging.info(f"Caminho final definido para Excel: {caminho_excel}")
    return {'caminho_csv': caminho_csv, 'caminho_excel': caminho_excel}

# --- Funções de Salvamento de Dados ---
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
        df.to_csv(nome_arquivo, index=False, encoding='utf-8-sig') # utf-8-sig para melhor compatibilidade Excel
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
    try:
        validar_coluna_existe(df, nome_coluna_cluster)
        validar_dependencia(
            'openpyxl',
            "A biblioteca 'openpyxl' é necessária para salvar arquivos .xlsx. Instale-a com 'pip install openpyxl'"
        )
    except (KeyError, ImportError) as e:
        return False

    resultados = pd.DataFrame()
    try:
        actual_clusters = df[nome_coluna_cluster].unique()
        for cluster_id in range(num_clusters):
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
            # Consideramos sucesso, pois não houve erro de IO, apenas não havia o que salvar.

    except Exception as e:
        logging.error(f"Falha ao gerar ou salvar o arquivo Excel de amostras '{nome_arquivo}': {e}")
        return False
    return True
