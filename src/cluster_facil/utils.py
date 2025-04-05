# -*- coding: utf-8 -*-
"""Módulo de funções utilitárias para o pacote cluster_facil."""

# --- Importações ---
import logging
import os
import re # Adicionado re
from typing import List, Optional, Union # Adicionado Union
# import matplotlib.pyplot as plt # Removido do topo
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
    validar_dependencia_leitura, # Adicionado para uso em salvar_dataframe
    validar_formato_salvar # Importado para determinar_caminhos_saida
)

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Funções de Ajuste e Preparação ---

def ajustar_rodada_inicial(colunas: pd.Index, prefixo_cluster: str) -> int:
    """
    Verifica colunas existentes com o prefixo fornecido e determina a próxima rodada.

    Args:
        colunas (pd.Index): As colunas do DataFrame a serem verificadas.
        prefixo_cluster (str): O prefixo usado para nomear as colunas de cluster (ex: 'cluster_', 'subcluster_').

    Returns:
        int: O número da próxima rodada de clusterização (começando em 1).
    """
    max_rodada_existente = 0
    # Usa o prefixo para criar o regex dinamicamente
    regex_coluna_cluster = re.compile(rf'^{re.escape(prefixo_cluster)}(\d+)$')
    logging.debug(f"Procurando colunas com padrão: {regex_coluna_cluster.pattern}")
    for col in colunas:
        match = regex_coluna_cluster.match(str(col)) # Garante que col seja string
        if match:
            rodada_num = int(match.group(1))
            logging.debug(f"Coluna encontrada: {col}, rodada: {rodada_num}")
            if rodada_num > max_rodada_existente:
                max_rodada_existente = rodada_num

    proxima_rodada = max_rodada_existente + 1
    if max_rodada_existente > 0:
        logging.info(f"Colunas com prefixo '{prefixo_cluster}' detectadas. Próxima rodada será: {proxima_rodada}")
    else:
        logging.info(f"Nenhuma coluna com prefixo '{prefixo_cluster}' encontrada. Iniciando na rodada 1.")
        proxima_rodada = 1 # Garante que seja 1 se nenhuma for encontrada
    return proxima_rodada

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

def calcular_inercias_kmeans(X: csr_matrix, limite_k: int, n_init: int = 1, random_state: Optional[int] = 42) -> Optional[List[float]]: # Adicionado random_state
    """
    Calcula as inércias do K-Means para diferentes valores de K.

    Args:
        X (csr_matrix): Matriz TF-IDF dos dados.
        limite_k (int): Número máximo de clusters (K) a testar.
        n_init (int): Número de inicializações do K-Means. Padrão é 1.
        random_state (Optional[int], optional): Semente para o gerador de números aleatórios. Padrão é 42.

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

        # Usa o random_state recebido
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=current_n_init)
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)
        logging.debug(f"Inércia para K={k}: {kmeans.inertia_}")

    return inercias

def _plotar_cotovelo_sem_dados():
    """Plota um gráfico indicando que não há dados para o método do cotovelo."""
    try:
        validar_dependencia('matplotlib', "A biblioteca 'matplotlib' é necessária para plotar gráficos. Instale-a com 'pip install matplotlib'")
        import matplotlib.pyplot as plt
    except ImportError as e:
        logging.error(f"Não foi possível plotar: {e}")
        return # Não plota se a dependência estiver faltando

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
    try:
        validar_dependencia('matplotlib', "A biblioteca 'matplotlib' é necessária para plotar gráficos. Instale-a com 'pip install matplotlib'")
        import matplotlib.pyplot as plt
    except ImportError as e:
        logging.error(f"Não foi possível plotar: {e}")
        return # Não plota se a dependência estiver faltando

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

def calcular_e_plotar_cotovelo(X: csr_matrix, limite_k: int, n_init: int = 1, plotar: bool = True, random_state: Optional[int] = 42) -> Optional[List[float]]: # Adicionado random_state
    """
    Calcula as inércias para diferentes valores de K e opcionalmente plota o gráfico do método do cotovelo.

    Orquestra o cálculo das inércias e a plotagem do gráfico.

    Args:
        X (csr_matrix): Matriz TF-IDF dos dados.
        limite_k (int): Número máximo de clusters (K) a testar.
        n_init (int, optional): Número de inicializações do K-Means para cálculo da inércia. Padrão é 1.
        plotar (bool, optional): Se True (padrão), exibe o gráfico do cotovelo. Padrão é True.
        random_state (Optional[int], optional): Semente para o gerador de números aleatórios. Padrão é 42.

    Returns:
        Optional[List[float]]: Lista de inércias calculadas, ou None se não houver dados.
    """
    # Passa o random_state para a função de cálculo
    inercias = calcular_inercias_kmeans(X, limite_k, n_init, random_state=random_state)

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

def determinar_caminhos_saida(
    o_que_salvar: str,
    formato_tudo: str,
    formato_amostras: str,
    caminho_tudo: Optional[str],
    caminho_amostras: Optional[str],
    diretorio_saida: Optional[str],
    input_path: Optional[str], # Para nome base padrão
    rodada_a_salvar: int,
    prefixo_cluster: str = "cluster_" # Adicionado prefixo
) -> dict[str, Optional[str]]:
    """
    Determina os caminhos e formatos finais para salvar os resultados.

    Encapsula a lógica de decisão de nomes/caminhos baseada nos parâmetros
    fornecidos e nos padrões.

    Args:
        o_que_salvar (str): 'tudo', 'amostras', ou 'ambos'.
        formato_tudo (str): Formato padrão para DataFrame completo ('csv', 'xlsx', etc.).
        formato_amostras (str): Formato padrão para amostras ('xlsx', 'csv', etc.).
        caminho_tudo (Optional[str]): Caminho explícito para DataFrame completo.
        caminho_amostras (Optional[str]): Caminho explícito para amostras.
        diretorio_saida (Optional[str]): Diretório padrão para salvar.
        input_path (Optional[str]): Caminho do arquivo de entrada original (para nome base).
        rodada_a_salvar (int): Número da rodada sendo salva.
        prefixo_cluster (str, optional): Prefixo usado para as colunas de cluster
                                         (ex: 'cluster_', 'subcluster_'). Padrão é 'cluster_'.

    Returns:
        dict[str, Optional[str]]: Dicionário contendo:
            'path_tudo_final': Caminho absoluto final para o DataFrame completo (ou None).
            'fmt_tudo_final': Formato final para o DataFrame completo (ou None).
            'path_amostras_final': Caminho absoluto final para as amostras (ou None).
            'fmt_amostras_final': Formato final para as amostras (ou None).

    Raises:
        ValueError: Se algum formato fornecido (explícito ou padrão) for inválido.
    """
    logging.debug("Determinando caminhos e formatos de saída...")
    path_tudo_final: Optional[str] = None
    path_amostras_final: Optional[str] = None
    fmt_tudo_final = formato_tudo.lower()
    fmt_amostras_final = formato_amostras.lower()

    # Lógica para nome base padrão
    nome_base_padrao = "clusters" # Default se não houver input_path
    if input_path:
        try:
            base = os.path.basename(input_path)
            nome_base_padrao, _ = os.path.splitext(base)
        except Exception:
            logging.warning(f"Não foi possível extrair nome base de '{input_path}'. Usando nome padrão 'clusters'.")

    # Determinar caminho/formato para DataFrame Completo
    if o_que_salvar in ['tudo', 'ambos']:
        if caminho_tudo:
            logging.info(f"Usando caminho explícito para DataFrame completo: {caminho_tudo}")
            # Extrai formato da extensão, se houver, e valida
            _, ext = os.path.splitext(caminho_tudo)
            fmt_detectado = ext[1:].lower() if ext else None
            if fmt_detectado:
                validar_formato_salvar(fmt_detectado, 'tudo') # Valida o formato detectado
                fmt_tudo_final = fmt_detectado
                path_tudo_final = caminho_tudo
            else:
                # Se não há extensão, usa formato_tudo padrão e adiciona extensão
                validar_formato_salvar(fmt_tudo_final, 'tudo') # Valida o formato padrão
                path_tudo_final = f"{caminho_tudo}.{fmt_tudo_final}"
                logging.info(f"Adicionando extensão .{fmt_tudo_final} ao caminho explícito.")
        else:
            # Usa nome padrão, incorporando o prefixo
            validar_formato_salvar(fmt_tudo_final, 'tudo') # Valida o formato padrão
            # Usa o prefixo no nome do arquivo, removendo o trailing '_' se existir
            prefixo_nome = prefixo_cluster.rstrip('_')
            nome_arquivo = f"{nome_base_padrao}_{prefixo_nome}_{rodada_a_salvar}.{fmt_tudo_final}"
            path_tudo_final = os.path.join(diretorio_saida or '.', nome_arquivo)
            logging.info(f"Usando caminho padrão para DataFrame completo: {path_tudo_final}")

    # Determinar caminho/formato para Amostras
    if o_que_salvar in ['amostras', 'ambos']:
         if caminho_amostras:
            logging.info(f"Usando caminho explícito para amostras: {caminho_amostras}")
            _, ext = os.path.splitext(caminho_amostras)
            fmt_detectado = ext[1:].lower() if ext else None
            if fmt_detectado:
                validar_formato_salvar(fmt_detectado, 'amostras') # Valida o formato detectado
                fmt_amostras_final = fmt_detectado
                path_amostras_final = caminho_amostras
            else:
                # Se não há extensão, usa formato_amostras padrão e adiciona extensão
                validar_formato_salvar(fmt_amostras_final, 'amostras') # Valida o formato padrão
                path_amostras_final = f"{caminho_amostras}.{fmt_amostras_final}"
                logging.info(f"Adicionando extensão .{fmt_amostras_final} ao caminho explícito.")
         else:
            # Usa nome padrão, incorporando o prefixo
            validar_formato_salvar(fmt_amostras_final, 'amostras') # Valida o formato padrão
            # Usa o prefixo no nome do arquivo, removendo o trailing '_' se existir
            prefixo_nome = prefixo_cluster.rstrip('_')
            nome_arquivo = f"{nome_base_padrao}_{prefixo_nome}_amostras_{rodada_a_salvar}.{fmt_amostras_final}"
            path_amostras_final = os.path.join(diretorio_saida or '.', nome_arquivo)
            logging.info(f"Usando caminho padrão para amostras: {path_amostras_final}")

    return {
        'path_tudo_final': os.path.abspath(path_tudo_final) if path_tudo_final else None,
        'fmt_tudo_final': fmt_tudo_final if path_tudo_final else None,
        'path_amostras_final': os.path.abspath(path_amostras_final) if path_amostras_final else None,
        'fmt_amostras_final': fmt_amostras_final if path_amostras_final else None
    }


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


# --- Funções de Subcluster ---

def criar_df_subcluster(df: pd.DataFrame, nome_coluna_classificacao: str, classificacao_desejada: str) -> pd.DataFrame:
    """
    Filtra um DataFrame por uma classificação, remove colunas de cluster/subcluster
    e renomeia a coluna de classificação original.

    Args:
        df (pd.DataFrame): O DataFrame original.
        nome_coluna_classificacao (str): Nome da coluna usada para classificar.
        classificacao_desejada (str): A classificação específica para filtrar.

    Returns:
        pd.DataFrame: Um novo DataFrame contendo apenas os dados filtrados e limpos.

    Raises:
        KeyError: Se a coluna de classificação não existir no DataFrame.
        ValueError: Se a classificação desejada não for encontrada.
    """
    logging.info(f"Criando DataFrame de subcluster para a classificação: '{classificacao_desejada}'")

    # Validação: Coluna de classificação existe? (Reutiliza validação)
    validar_coluna_existe(df, nome_coluna_classificacao)

    # Validação: Classificação desejada existe?
    if classificacao_desejada not in df[nome_coluna_classificacao].unique():
        msg = f"A classificação '{classificacao_desejada}' não foi encontrada na coluna '{nome_coluna_classificacao}'."
        logging.error(msg)
        raise ValueError(msg)

    # Filtragem
    df_sub = df[df[nome_coluna_classificacao] == classificacao_desejada].copy()
    logging.info(f"Subcluster DataFrame criado com {len(df_sub)} linhas.")

    # Limpeza de colunas de cluster/subcluster existentes
    colunas_cluster_para_remover = []
    # Procura por colunas que comecem com 'cluster_' ou 'subcluster_' seguido por números
    regex_qualquer_cluster = re.compile(r'^(cluster_|subcluster_)\d+$')
    for col in df_sub.columns:
        if regex_qualquer_cluster.match(str(col)): # Garante que col seja string
            colunas_cluster_para_remover.append(col)

    if colunas_cluster_para_remover:
        df_sub.drop(columns=colunas_cluster_para_remover, inplace=True, errors='ignore')
        logging.info(f"Colunas de cluster/subcluster removidas do subcluster: {colunas_cluster_para_remover}")

    # Renomear coluna de classificação original
    coluna_origem = f"{nome_coluna_classificacao}_origem"
    df_sub.rename(columns={nome_coluna_classificacao: coluna_origem}, inplace=True)
    logging.info(f"Coluna de classificação original renomeada para '{coluna_origem}'.")

    return df_sub
