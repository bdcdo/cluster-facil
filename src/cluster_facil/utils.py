# -*- coding: utf-8 -*-
"""Módulo de funções utilitárias para o pacote cluster_facil."""

# --- Importações ---
import logging
import os
from typing import List, Optional, Union # Adicionado Union
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
    validar_dependencia,
    validar_dependencia_leitura # Adicionado para uso em salvar_dataframe
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
        aba (Optional[str], optional): O nome ou índice da aba a ser lida caso a entrada
                                       seja um caminho para um arquivo Excel (.xlsx).
                                       Se None (padrão), lê a primeira aba. Padrão é None.

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

def calcular_inercias_kmeans(X: csr_matrix, limite_k: int, n_init: int = 1) -> Optional[List[float]]:
    """
    Calcula as inércias do K-Means para diferentes valores de K.

    Args:
        X (csr_matrix): Matriz TF-IDF dos dados.
        limite_k (int): Número máximo de clusters (K) a testar.
        n_init (int): Número de inicializações do K-Means. Padrão é 1.

    Returns:
        Optional[List[float]]: Lista de inércias calculadas, ou None se não houver amostras.
    """
    logging.info("Calculando inércias para o método do cotovelo...")
    validar_inteiro_positivo('n_init', n_init) # Valida o n_init recebido

    k_max = min(limite_k, X.shape[0])
    if k_max < limite_k:
        logging.warning(f"Limite K ({limite_k}) é maior que o número de amostras ({X.shape[0]}). Usando K máximo = {k_max}.")
    if k_max == 0:
        logging.error("Não há amostras para calcular inércias.")
        return None

    inercias = []
    k_range = range(1, k_max + 1)
    for k in k_range:
        current_n_init = n_init
        # Ajusta n_init apenas se for maior que as amostras para K=1 (caso especial do KMeans)
        if k == 1 and n_init > X.shape[0]:
            logging.warning(f"n_init ({n_init}) > amostras ({X.shape[0]}) para K=1. Usando n_init=1.")
            current_n_init = 1

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=current_n_init)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)
        logging.debug(f"Inércia para K={k}: {kmeans.inertia_}")

    return inercias

def _plotar_cotovelo_sem_dados():
    """Plota um gráfico indicando que não há dados para o método do cotovelo."""
    plt.figure(figsize=(10, 6))
    plt.title('Método do Cotovelo - Nenhuma amostra encontrada')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia (WCSS)')
    plt.text(0.5, 0.5, 'Não há dados para processar', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.show()

def plotar_grafico_cotovelo(k_range: range, inercias: List[float]):
    """
    Plota o gráfico do método do cotovelo.

    Args:
        k_range (range): O range de valores de K utilizados.
        inercias (List[float]): A lista de inércias correspondente a cada K.
    """
    logging.info("Gerando gráfico do método do cotovelo...")
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inercias, marker='o')
    plt.title('Método do Cotovelo para Escolha de K')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia (WCSS)')
    if len(k_range) > 0: # Evita erro se k_range for vazio
        plt.xticks(k_range)
    plt.grid(True)
    logging.info("Exibindo gráfico do método do cotovelo...")
    plt.show()

def calcular_e_plotar_cotovelo(X: csr_matrix, limite_k: int, n_init: int = 1, plotar: bool = True) -> Optional[List[float]]:
    """
    Calcula as inércias para diferentes valores de K e opcionalmente plota o gráfico do método do cotovelo.

    Orquestra o cálculo das inércias e a plotagem do gráfico.

    Args:
        X (csr_matrix): Matriz TF-IDF dos dados.
        limite_k (int): Número máximo de clusters (K) a testar.
        n_init (int, optional): Número de inicializações do K-Means para cálculo da inércia. Padrão é 1.
        plotar (bool, optional): Se True (padrão), exibe o gráfico do cotovelo. Padrão é True.

    Returns:
        Optional[List[float]]: Lista de inércias calculadas, ou None se não houver dados.
    """
    inercias = calcular_inercias_kmeans(X, limite_k, n_init)

    if inercias is None:
        if plotar:
            _plotar_cotovelo_sem_dados()
        return None

    if plotar:
        k_max = min(limite_k, X.shape[0]) # Recalcula k_max para o range correto
        k_range = range(1, k_max + 1)
        plotar_grafico_cotovelo(k_range, inercias)
    else:
        logging.info("Gráfico do método do cotovelo não será exibido (plotar=False).")

    return inercias

# --- Funções de Salvamento de Dados ---

def salvar_dataframe(df: pd.DataFrame, caminho_arquivo: str, formato: str) -> bool:
    """
    Salva um DataFrame em um arquivo no formato especificado.

    Suporta 'csv', 'xlsx', 'parquet', 'json'.

    Args:
        df (pd.DataFrame): O DataFrame a ser salvo.
        caminho_arquivo (str): O caminho completo do arquivo de saída (incluindo nome e extensão).
        formato (str): O formato desejado ('csv', 'xlsx', 'parquet', 'json'). A extensão
                       no `caminho_arquivo` deve ser consistente com este formato.

    Returns:
        bool: True se o salvamento for bem-sucedido, False caso contrário.

    Raises:
        ImportError: Se uma dependência necessária para o formato não estiver instalada.
        ValueError: Se o formato for inválido.
    """
    logging.info(f"Tentando salvar DataFrame em '{caminho_arquivo}' (formato: {formato})...")
    formato = formato.lower()
    extensao = f".{formato}"

    try:
        # Garante que o diretório exista
        diretorio = os.path.dirname(caminho_arquivo)
        if diretorio: # Se não for vazio (salvando no diretório atual)
            os.makedirs(diretorio, exist_ok=True)

        # Valida dependências antes de tentar salvar
        validar_dependencia_leitura(extensao)

        if formato == 'csv':
            df.to_csv(caminho_arquivo, index=False, encoding='utf-8-sig')
        elif formato == 'xlsx':
            df.to_excel(caminho_arquivo, index=False)
        elif formato == 'parquet':
            df.to_parquet(caminho_arquivo, index=False)
        elif formato == 'json':
            df.to_json(caminho_arquivo, orient='records', indent=4, force_ascii=False)
        else:
            # Esta validação já deve ter ocorrido antes, mas por segurança
            raise ValueError(f"Formato de salvamento não suportado: {formato}")

        caminho_abs = os.path.abspath(caminho_arquivo)
        logging.info(f"DataFrame salvo com sucesso em '{caminho_abs}'.")
        return True

    except ImportError as e:
        logging.error(f"Falha ao salvar '{caminho_arquivo}': {e}")
        raise # Re-levanta ImportError para ser tratado pelo chamador
    except Exception as e:
        logging.error(f"Falha ao salvar o arquivo '{caminho_arquivo}' (formato {formato}): {e}")
        return False

def _gerar_dataframe_amostras(df: pd.DataFrame, nome_coluna_cluster: str, num_clusters: int) -> pd.DataFrame:
    """
    Gera um DataFrame contendo amostras aleatórias (até 10 por cluster) do DataFrame original.
    As amostras mantêm todas as colunas originais.
    (Função auxiliar interna)

    Args:
        df (pd.DataFrame): O DataFrame completo contendo os dados e a(s) coluna(s) de cluster.
        nome_coluna_cluster (str): O nome da coluna que identifica os clusters da rodada
                                   da qual se deseja extrair as amostras.
        num_clusters (int): O número de clusters esperado nessa rodada (usado para iterar).

    Returns:
        pd.DataFrame: DataFrame com as amostras concatenadas (até 10 por cluster),
                      ou DataFrame vazio se não houver amostras válidas.
    """
    resultados = pd.DataFrame()
    # Garante que estamos lidando com clusters válidos (números inteiros não nulos)
    actual_clusters = df[nome_coluna_cluster].dropna().astype(int).unique()

    for cluster_id in range(num_clusters):
        if cluster_id not in actual_clusters:
            logging.warning(f"Cluster ID {cluster_id} não foi encontrado nos dados ou não foi gerado. Nenhuma amostra será retirada.")
            continue

        df_cluster = df[df[nome_coluna_cluster] == cluster_id]
        tamanho_amostra = min(10, len(df_cluster))
        if tamanho_amostra > 0:
            try:
                amostra_cluster = df_cluster.sample(tamanho_amostra, random_state=42)
                # A linha abaixo foi removida pois a coluna de cluster já existe no df_cluster
                # amostra_cluster.insert(0, 'cluster_original_id', cluster_id)
                resultados = pd.concat([resultados, amostra_cluster], ignore_index=True)
            except Exception as e:
                logging.error(f"Erro ao amostrar cluster {cluster_id}: {e}")
                # Continua para os próximos clusters
        else:
            logging.warning(f"Cluster {cluster_id} está vazio ou contém apenas NA, nenhuma amostra será retirada.")

    return resultados

def salvar_amostras(df: pd.DataFrame, nome_coluna_cluster: str, num_clusters: int, caminho_arquivo: str, formato: str) -> bool:
    """
    Gera e salva amostras (até 10 por cluster) de um DataFrame em um arquivo no formato especificado.

    Suporta os formatos 'xlsx', 'csv', 'json'.

    Args:
        df (pd.DataFrame): O DataFrame completo contendo os dados e a coluna de cluster.
        nome_coluna_cluster (str): O nome da coluna que identifica os clusters da rodada
                                   da qual se deseja extrair as amostras.
        num_clusters (int): O número de clusters esperado nessa rodada (usado para gerar amostras).
        caminho_arquivo (str): O caminho completo do arquivo de saída para as amostras
                               (incluindo nome e extensão).
        formato (str): O formato desejado ('xlsx', 'csv', 'json'). A extensão no
                       `caminho_arquivo` deve ser consistente com este formato.

    Returns:
        bool: True se o salvamento for bem-sucedido (ou se não houver amostras para salvar),
              False em caso de erro durante o salvamento.

    Raises:
        ImportError: Se uma dependência necessária para o formato não estiver instalada.
        ValueError: Se o formato for inválido.
        KeyError: Se a coluna de cluster não existir.
    """
    logging.info(f"Tentando gerar e salvar amostras (até 10 por cluster) em '{caminho_arquivo}' (formato: {formato})...")
    formato = formato.lower()
    formatos_validos = ['xlsx', 'csv', 'json']
    if formato not in formatos_validos:
        raise ValueError(f"Formato inválido para salvar amostras: '{formato}'. Use um de {formatos_validos}.")

    try:
        # Validações essenciais antes de gerar amostras
        validar_coluna_existe(df, nome_coluna_cluster)
        # A validação de dependência será feita dentro de salvar_dataframe

        # Gera o DataFrame de amostras usando a função auxiliar
        df_amostras = _gerar_dataframe_amostras(df, nome_coluna_cluster, num_clusters)

        if not df_amostras.empty:
            # Tenta salvar o DataFrame de amostras usando a função genérica
            # A função salvar_dataframe cuidará da validação de dependência e do log de sucesso
            sucesso = salvar_dataframe(df_amostras, caminho_arquivo, formato)
            if not sucesso:
                 # Log de erro já foi feito por salvar_dataframe
                 return False
        else:
            logging.warning(f"Nenhuma amostra foi gerada para '{caminho_arquivo}'. Arquivo não será criado.")
            # Consideramos sucesso, pois não houve erro de IO, apenas não havia o que salvar.

    except (KeyError, ValueError, ImportError) as e:
        # Captura erros de validação, formato inválido ou dependência (re-levantados por salvar_dataframe)
        logging.error(f"Pré-requisito ou validação para salvar amostras falhou: {e}")
        raise # Re-levanta a exceção para ser tratada pelo chamador (ClusterFacil.salvar)
    except Exception as e:
        # Captura outros erros inesperados durante a geração das amostras
        logging.error(f"Falha inesperada ao gerar ou preparar para salvar amostras em '{caminho_arquivo}': {e}")
        return False # Retorna False para erros inesperados na geração

    return True # Retorna True se salvou com sucesso ou se não havia amostras
