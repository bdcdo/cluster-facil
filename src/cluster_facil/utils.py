# -*- coding: utf-8 -*-
"""Módulo de funções utilitárias para o pacote cluster_facil."""

# --- Importações ---
import logging
import os
import re
from zipfile import BadZipFile
import pandas as pd
from tqdm import tqdm
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
    validar_dependencia_leitura,
    validar_formato_salvar
)

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
        logging.debug(f"Colunas com prefixo '{prefixo_cluster}' detectadas. Próxima rodada será: {proxima_rodada}") # Movido para DEBUG
    else:
        logging.debug(f"Nenhuma coluna com prefixo '{prefixo_cluster}' encontrada. Iniciando na rodada 1.") # Movido para DEBUG
        proxima_rodada = 1 # Garante que seja 1 se nenhuma for encontrada
    return proxima_rodada

# --- Constante de Stopwords (Internalizada para Reprodutibilidade) ---
# Lista de stopwords em português (baseada no NLTK 3.9.1)
STOPWORDS_PT: list[str] = [ # Alterado de tuple para list
    'a', 'à', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as',
    'às', 'até', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles',
    'depois', 'do', 'dos', 'e', 'é', 'ela', 'elas', 'ele', 'eles', 'em', 'entre',
    'era', 'eram', 'éramos', 'essa', 'essas', 'esse', 'esses', 'esta', 'está',
    'estamos', 'estão', 'estar', 'estas', 'estava', 'estavam', 'estávamos', 'este',
    'esteja', 'estejam', 'estejamos', 'estes', 'esteve', 'estive', 'estivemos',
    'estiver', 'estivera', 'estiveram', 'estivéramos', 'estiverem', 'estivermos',
    'estivesse', 'estivessem', 'estivéssemos', 'estou', 'eu', 'foi', 'fomos', 'for',
    'fora', 'foram', 'fôramos', 'forem', 'formos', 'fosse', 'fossem', 'fôssemos',
    'fui', 'há', 'haja', 'hajam', 'hajamos', 'hão', 'havemos', 'haver', 'hei',
    'houve', 'houvemos', 'houver', 'houvera', 'houverá', 'houveram', 'houvéramos',
    'houverão', 'houverei', 'houverem', 'houveremos', 'houveria', 'houveriam',
    'houveríamos', 'houvermos', 'houvesse', 'houvessem', 'houvéssemos', 'isso',
    'isto', 'já', 'lhe', 'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus',
    'minha', 'minhas', 'muito', 'na', 'não', 'nas', 'nem', 'no', 'nos', 'nós',
    'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 'ou', 'para',
    'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'são',
    'se', 'seja', 'sejam', 'sejamos', 'sem', 'ser', 'será', 'serão', 'serei',
    'seremos', 'seria', 'seriam', 'seríamos', 'seu', 'seus', 'só', 'somos', 'sou',
    'sua', 'suas', 'também', 'te', 'tem', 'tém', 'temos', 'tenha', 'tenham',
    'tenhamos', 'tenho', 'terá', 'terão', 'terei', 'teremos', 'teria', 'teriam',
    'teríamos', 'teu', 'teus', 'teve', 'tinha', 'tinham', 'tínhamos', 'tive',
    'tivemos', 'tiver', 'tivera', 'tiveram', 'tivéramos', 'tiverem', 'tivermos',
    'tivesse', 'tivessem', 'tivéssemos', 'tu', 'tua', 'tuas', 'um', 'uma', 'você',
    'vocês', 'vos'
]

# --- Funções de Carregamento de Dados ---
def _ler_com_fallback_encoding(
    fn_leitura,
    encoding_principal: str | None,
    encodings_fallback: list[str],
    caminho_arquivo: str
) -> pd.DataFrame:
    """
    Tenta ler um arquivo com o encoding principal e, em caso de falha por
    UnicodeDecodeError, tenta encodings alternativos automaticamente.

    Isso evita que usuários precisem lidar manualmente com erros de codificação,
    comuns em arquivos CSV gerados por Excel no Brasil/Portugal.

    Args:
        fn_leitura: Função que recebe um encoding (str) e retorna um DataFrame.
        encoding_principal (str | None): Encoding a ser tentado primeiro.
        encodings_fallback (list[str]): Lista de encodings alternativos para tentar.
        caminho_arquivo (str): Caminho do arquivo (usado apenas para mensagens de log).

    Returns:
        pd.DataFrame: Os dados lidos com sucesso.

    Raises:
        UnicodeDecodeError: Se nenhum dos encodings conseguir ler o arquivo.
    """
    try:
        return fn_leitura(encoding_principal)
    except UnicodeDecodeError:
        nome_arquivo = os.path.basename(caminho_arquivo)
        logging.warning(
            f"Não foi possível ler '{nome_arquivo}' com encoding '{encoding_principal}'. "
            f"Tentando encodings alternativos: {encodings_fallback}"
        )
        for enc in encodings_fallback:
            try:
                df = fn_leitura(enc)
                logging.info(
                    f"Arquivo '{nome_arquivo}' lido com sucesso usando encoding alternativo '{enc}'. "
                    f"Dica: para evitar este aviso, salve o arquivo em formato UTF-8."
                )
                return df
            except UnicodeDecodeError:
                logging.debug(f"Encoding '{enc}' também falhou para '{nome_arquivo}'.")
                continue
        raise UnicodeDecodeError(
            'múltiplos', b'', 0, 1,
            f"Não foi possível ler o arquivo '{nome_arquivo}' com nenhum dos encodings "
            f"tentados ({encoding_principal}, {', '.join(encodings_fallback)}). "
            f"Verifique a codificação do arquivo ou tente especificar o encoding correto "
            f"no parâmetro 'encoding' da função carregar_dados()."
        )


def _ler_excel_com_fallback(
    caminho_arquivo: str,
    aba: str | None,
    dtype: dict | None,
    encoding: str | None,
    encodings_fallback: list[str]
) -> pd.DataFrame:
    """
    Tenta ler um arquivo .xlsx, primeiro com openpyxl (padrão), depois com
    engines alternativas (calamine), e por último como CSV caso o arquivo
    não seja um Excel válido (ex: CSV renomeado para .xlsx).

    Args:
        caminho_arquivo (str): Caminho do arquivo.
        aba (str | None): Aba do Excel a ser lida.
        dtype (dict | None): Tipos das colunas.
        encoding (str | None): Encoding principal para fallback CSV.
        encodings_fallback (list[str]): Encodings alternativos para fallback CSV.

    Returns:
        pd.DataFrame: Os dados lidos com sucesso.

    Raises:
        BadZipFile: Se o arquivo não puder ser lido como Excel nem como CSV.
        Exception: Se todas as tentativas falharem.
    """
    nome_arquivo = os.path.basename(caminho_arquivo)

    # Tentativa 1: engine padrão (openpyxl)
    try:
        return pd.read_excel(caminho_arquivo, sheet_name=aba, dtype=dtype)
    except BadZipFile:
        pass

    # Tentativa 2: engine alternativa (calamine — baseada em Rust, lê formatos
    # gerados por outras bibliotecas como o Polars)
    _engines_alternativas = ['calamine']
    for engine in _engines_alternativas:
        try:
            df = pd.read_excel(caminho_arquivo, sheet_name=aba, dtype=dtype, engine=engine)
            logging.info(
                f"Arquivo '{nome_arquivo}' lido com sucesso usando engine alternativa '{engine}'."
            )
            return df
        except ImportError:
            logging.debug(
                f"Engine '{engine}' não disponível (biblioteca não instalada). Pulando."
            )
        except Exception:
            logging.debug(
                f"Engine '{engine}' também não conseguiu ler '{nome_arquivo}'. Pulando."
            )

    # Tentativa 3: talvez seja um CSV renomeado para .xlsx
    logging.warning(
        f"O arquivo '{nome_arquivo}' tem extensão .xlsx mas não pôde ser lido como "
        f"Excel por nenhuma engine disponível. Tentando ler como CSV..."
    )
    try:
        df = _ler_com_fallback_encoding(
            lambda enc: pd.read_csv(caminho_arquivo, dtype=dtype, encoding=enc),
            encoding, encodings_fallback, caminho_arquivo
        )
        logging.info(
            f"Arquivo '{nome_arquivo}' lido com sucesso como CSV. "
            f"Dica: renomeie o arquivo com a extensão .csv para evitar este aviso."
        )
        return df
    except Exception:
        # Se nem como CSV funcionar, re-levanta o erro original (BadZipFile)
        raise BadZipFile(
            f"Não foi possível ler o arquivo '{nome_arquivo}'. "
            f"Ele tem extensão .xlsx mas não é um arquivo Excel válido, "
            f"e também não pôde ser lido como CSV. "
            f"Verifique se o arquivo não está corrompido."
        )


def carregar_dados(
    caminho_arquivo: str,
    aba: str | None = None,
    dtype: dict | None = None,
    encoding: str | None = 'utf-8-sig' # Padrão para CSV/JSON, lida com BOM
) -> pd.DataFrame:
    """
    Carrega dados de um arquivo usando Pandas, com opções para maior reprodutibilidade.

    Suporta CSV, Excel (.xlsx), Parquet e JSON.

    Args:
        caminho_arquivo (str): O caminho completo para o arquivo de dados.
        aba (str | None, optional): O nome ou índice da aba a ser lida caso a entrada
                                    seja um caminho para um arquivo Excel (.xlsx).
                                    Se None (padrão), lê a primeira aba. Padrão é None.
        dtype (dict | None, optional): Dicionário mapeando nomes de colunas para tipos
                                       (ex: {'coluna_id': int, 'texto': str}).
                                       Ajuda a garantir a consistência na leitura.
                                       Padrão é None (Pandas infere os tipos).
        encoding (str | None, optional): Codificação a ser usada ao ler arquivos de texto
                                         (CSV, JSON). Padrão é 'utf-8-sig', que lida
                                         com o BOM (Byte Order Mark) comum em arquivos
                                         UTF-8 gerados no Windows. Se None, o Pandas
                                         tentará detectar a codificação (pode falhar em alguns casos).

    Returns:
        pd.DataFrame: Os dados carregados em formato de tabela (DataFrame).

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado.
        ImportError: Se uma dependência necessária (ex: openpyxl, pyarrow) não estiver instalada.
        ValueError: Se o formato do arquivo não for suportado ou houver erro na leitura.
        Exception: Para outros erros inesperados durante o carregamento do arquivo.
    """
    logging.info(f"Carregando dados do arquivo: {caminho_arquivo}")
    validar_arquivo_existe(caminho_arquivo)
    _, extensao = os.path.splitext(caminho_arquivo)
    extensao = extensao.lower()
    validar_formato_suportado(extensao)
    validar_dependencia_leitura(extensao)

    logging.debug(f"Parâmetros de leitura: aba='{aba}', dtype={'especificado' if dtype else 'inferido'}, encoding='{encoding}' (para CSV/JSON)") # Movido para DEBUG

    # Encodings alternativos para tentar caso o encoding principal falhe.
    # 'latin-1' (iso-8859-1) é muito comum em arquivos CSV gerados por Excel no Brasil/Portugal.
    # 'cp1252' (Windows-1252) é outra variante comum no Windows.
    _encodings_fallback = ['latin-1', 'cp1252']

    try:
        if extensao == '.csv':
            df = _ler_com_fallback_encoding(
                lambda enc: pd.read_csv(caminho_arquivo, dtype=dtype, encoding=enc),
                encoding, _encodings_fallback, caminho_arquivo
            )
        elif extensao == '.xlsx':
            df = _ler_excel_com_fallback(
                caminho_arquivo, aba, dtype, encoding, _encodings_fallback
            )
        elif extensao == '.parquet':
            # read_parquet não usa encoding; dtype pode ser inferido ou especificado via 'columns'
            # Para simplificar, não passamos dtype aqui, mas a opção existe se necessário.
            df = pd.read_parquet(caminho_arquivo)
        elif extensao == '.json':
            df = _ler_com_fallback_encoding(
                lambda enc: pd.read_json(caminho_arquivo, dtype=dtype, encoding=enc, orient='records'),
                encoding, _encodings_fallback, caminho_arquivo
            )

        logging.info(f"Arquivo '{os.path.basename(caminho_arquivo)}' carregado com sucesso ({df.shape[0]} linhas, {df.shape[1]} colunas).")
        logging.debug(f"Caminho completo do arquivo carregado: {caminho_arquivo}") # DEBUG
        return df
    except Exception as e:
        # Captura outros erros de leitura (ex: arquivo corrompido, JSON mal formatado, aba não encontrada)
        logging.error(f"Erro ao ler o arquivo {caminho_arquivo} (formato {extensao}): {e}")
        raise ValueError(f"Erro ao processar o arquivo {caminho_arquivo}: {e}")

# --- Funções de Análise e Plotagem ---
def calcular_inercias_kmeans(X: csr_matrix, limite_k: int, n_init: str | int = 'auto', random_state: int | None = 42) -> list[float] | None: # n_init agora aceita 'auto'
    """
    Calcula as inércias (uma medida de coesão interna dos grupos) do K-Means para diferentes números de grupos (K).

    Args:
        X (csr_matrix): Matriz com as características dos textos (resultado do TF-IDF).
        limite_k (int): Número máximo de grupos (K) a serem testados.
        n_init (str | int): Número de inicializações do K-Means ('auto' ou um inteiro). Padrão é 'auto'.
        random_state (int | None, optional): Semente para garantir resultados reproduzíveis. Padrão é 42.

    Returns:
        list[float] | None: Lista com os valores de inércia calculados para cada K,
                           ou None se não houver textos para analisar.
    """
    # A validação de n_init ('auto' ou int) é feita pelo próprio KMeans.
    # Não precisamos validar aqui explicitamente se for 'auto'.
    if isinstance(n_init, int):
        validar_inteiro_positivo('n_init', n_init)

    k_max = min(limite_k, X.shape[0]) # K não pode ser maior que o número de textos
    if k_max < limite_k:
        logging.warning(f"O limite de grupos K ({limite_k}) é maior que o número de textos disponíveis ({X.shape[0]}). O teste será feito até K={k_max}.")
    if k_max == 0:
        logging.error("Não há textos para calcular as opções de agrupamento (inércias).")
        return None

    logging.info(f"Calculando opções de agrupamento (inércias) para K de 1 a {k_max}...")
    inercias = []
    k_range = range(1, k_max + 1)

    # --- Adiciona barra de progresso com tqdm ---
    for k in tqdm(k_range, desc="Avaliando número de grupos (K)", unit="K"):
        # O KMeans lida internamente com n_init='auto' ou int.
        # Não precisamos mais do ajuste manual para K=1.
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        try:
            kmeans.fit(X)
            inercias.append(kmeans.inertia_)
            logging.debug(f"Inércia calculada para K={k}: {kmeans.inertia_}")
        except Exception as e:
            logging.error(f"Erro ao calcular K-Means para K={k} com n_init='{n_init}': {e}")
            # Decide se continua ou para. Vamos continuar, mas registrar o erro.
            # Poderia adicionar um valor placeholder como float('nan') ou None?
            # Por enquanto, vamos pular este K se der erro.
            # Se muitos erros ocorrerem, a lista de inércias pode ficar incompleta.
            # TODO: Considerar uma estratégia melhor para lidar com falhas em K específico.
            continue # Pula para o próximo K

    if not inercias: # Se a lista ficou vazia devido a erros em todos os K
        logging.error("Não foi possível calcular nenhuma inércia devido a erros.")
        return None

    logging.info("Cálculo das opções de agrupamento (inércias) concluído.")
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
    plt.title('Gráfico do Cotovelo - Nenhum texto encontrado')
    plt.xlabel('Número de Grupos (K)')
    plt.ylabel('Coesão Interna (Inércia / WCSS)')
    plt.text(0.5, 0.5, 'Não há textos para analisar', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.grid(True)
    plt.show()

def plotar_grafico_cotovelo(k_range: range, inercias: list[float]):
    """
    Plota o gráfico do método do cotovelo.

    Args:
        k_range (range): O range de valores de K utilizados.
        inercias (list[float]): A lista de inércias correspondente a cada K.
    """
    try:
        validar_dependencia('matplotlib', "A biblioteca 'matplotlib' é necessária para plotar gráficos. Instale-a com 'pip install matplotlib'")
        import matplotlib.pyplot as plt
    except ImportError as e:
        logging.error(f"Não foi possível gerar o gráfico: {e}")
        return # Não plota se a dependência estiver faltando

    logging.info("Gerando gráfico do cotovelo para visualização...")
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inercias, marker='o')
    plt.title('Gráfico do Cotovelo para Escolha do Número de Grupos (K)')
    plt.xlabel('Número de Grupos (K)')
    plt.ylabel('Coesão Interna (Inércia / WCSS)') # Within-Cluster Sum of Squares
    if len(k_range) > 0: # Evita erro se k_range for vazio
        # Garante que os ticks sejam inteiros, se possível
        tick_values = list(k_range)
        if len(tick_values) <= 20: # Evita muitos ticks se o range for grande
             plt.xticks(tick_values)
        else:
             # Usa os ticks padrão do matplotlib se houver muitos Ks
             pass
    plt.grid(True)
    logging.info("Exibindo gráfico do cotovelo...")
    plt.show()

def calcular_e_plotar_cotovelo(X: csr_matrix, limite_k: int, n_init: str | int = 'auto', plotar: bool = True, random_state: int | None = 42) -> list[float] | None: # n_init aceita 'auto'
    """
    Calcula as opções de agrupamento (inércias) e opcionalmente plota o gráfico do cotovelo.

    Coordena o cálculo das inércias e a exibição do gráfico.

    Args:
        X (csr_matrix): Matriz com as características dos textos (resultado do TF-IDF).
        limite_k (int): Número máximo de grupos (K) a serem testados.
        n_init (str | int, optional): Como inicializar o K-Means ('auto' ou int). Padrão é 'auto'.
        plotar (bool, optional): Se True (padrão), exibe o gráfico do cotovelo. Padrão é True.
        random_state (int | None, optional): Semente para garantir resultados reproduzíveis. Padrão é 42.

    Returns:
        list[float] | None: Lista com os valores de inércia calculados, ou None se não houver textos.
    """
    # Passa n_init e random_state para a função de cálculo
    inercias = calcular_inercias_kmeans(X, limite_k, n_init=n_init, random_state=random_state)

    if inercias is None:
        if plotar:
            _plotar_cotovelo_sem_dados() # Mostra gráfico vazio
        # Mensagem de erro já foi logada por calcular_inercias_kmeans
        return None

    if plotar:
        k_max = min(limite_k, X.shape[0]) # Recalcula k_max para o range correto
        # Ajusta k_range para corresponder ao número de inércias realmente calculadas
        # Isso é importante se houve erro em algum K dentro de calcular_inercias_kmeans
        k_range_plot = range(1, len(inercias) + 1)
        if len(k_range_plot) < k_max +1:
             logging.warning(f"O gráfico do cotovelo pode estar incompleto devido a erros no cálculo de alguns valores de K (exibindo {len(k_range_plot)-1} pontos).")
        if len(k_range_plot) > 1: # Só plota se tiver pelo menos 1 ponto (K=1)
            plotar_grafico_cotovelo(k_range_plot, inercias)
        else:
            logging.warning("Não há dados suficientes (apenas K=1 ou menos) para plotar o gráfico do cotovelo.")

    else:
        logging.info("Gráfico do cotovelo não será exibido (conforme solicitado).")

    return inercias

# --- Funções de Salvamento de Dados ---
def determinar_caminhos_saida(
    o_que_salvar: str,
    formato_tudo: str,
    formato_amostras: str,
    caminho_tudo: str | None,
    caminho_amostras: str | None,
    diretorio_saida: str | None,
    input_path: str | None, # Para nome base padrão
    rodada_a_salvar: int,
    prefixo_cluster: str = "cluster_" # Adicionado prefixo
) -> dict[str, str | None]:
    """
    Determina os caminhos e formatos finais para salvar os arquivos de resultados.

    Define os nomes e locais dos arquivos com base nas opções do usuário e nos padrões.

    Args:
        o_que_salvar (str): O que salvar ('tudo', 'amostras', 'ambos').
        formato_tudo (str): Formato para o arquivo completo ('csv', 'xlsx', etc.).
        formato_amostras (str): Formato para o arquivo de amostras ('xlsx', 'csv', etc.).
        caminho_tudo (str | None): Caminho completo fornecido pelo usuário para o arquivo completo.
        caminho_amostras (str | None): Caminho completo fornecido pelo usuário para o arquivo de amostras.
        diretorio_saida (str | None): Pasta padrão para salvar (se caminhos não forem fornecidos).
        input_path (str | None): Caminho do arquivo original (usado para gerar nome padrão).
        rodada_a_salvar (int): Número da rodada de agrupamento cujos resultados serão salvos.
        prefixo_cluster (str, optional): Prefixo usado nas colunas de resultado ('cluster_', 'subcluster_'). Padrão é 'cluster_'.

    Returns:
        dict[str, str | None]: Dicionário contendo:
            'path_tudo_final': Caminho absoluto final para o DataFrame completo (ou None).
            'fmt_tudo_final': Formato final para o DataFrame completo (ou None).
            'path_amostras_final': Caminho absoluto final para as amostras (ou None).
            'fmt_amostras_final': Formato final para as amostras (ou None).

    Raises:
        ValueError: Se algum formato de arquivo solicitado for inválido.
    """
    logging.debug("Determinando caminhos e formatos para salvar os resultados...")
    path_tudo_final: str | None = None
    path_amostras_final: str | None = None
    fmt_tudo_final = formato_tudo.lower()
    fmt_amostras_final = formato_amostras.lower()

    # Define um nome base para os arquivos padrão
    nome_base_padrao = "resultados_cluster" # Default se não houver input_path
    if input_path:
        try:
            base = os.path.basename(input_path)
            nome_base_padrao, _ = os.path.splitext(base)
            logging.debug(f"Nome base para arquivos padrão derivado do input: '{nome_base_padrao}'")
        except Exception as e:
            logging.warning(f"Não foi possível extrair nome base do caminho de entrada '{input_path}': {e}. Usando nome padrão '{nome_base_padrao}'.")

    # Determinar caminho/formato para o arquivo completo
    if o_que_salvar in ['tudo', 'ambos']:
        # --- Lógica para o arquivo completo ---
        if caminho_tudo:
            # Caso 1: Usuário forneceu um caminho explícito para o arquivo completo.
            logging.debug(f"Caminho explícito fornecido para o arquivo completo: {caminho_tudo}")
            # Verifica se o caminho explícito já inclui uma extensão.
            _, ext = os.path.splitext(caminho_tudo)
            fmt_detectado = ext[1:].lower() if ext else None
            if fmt_detectado:
                # Se há extensão, usa-a como formato final (após validação).
                validar_formato_salvar(fmt_detectado, 'tudo') # Valida o formato detectado da extensão
                fmt_tudo_final = fmt_detectado
                path_tudo_final = caminho_tudo # Usa o caminho como está
                logging.debug(f"Formato '{fmt_tudo_final}' detectado da extensão do caminho explícito.")
            else:
                # Se não há extensão no caminho explícito, usa o formato padrão (`formato_tudo`)
                # e adiciona essa extensão ao caminho fornecido.
                validar_formato_salvar(fmt_tudo_final, 'tudo') # Valida o formato padrão
                path_tudo_final = f"{caminho_tudo}.{fmt_tudo_final}"
                logging.debug(f"Nenhuma extensão no caminho explícito. Usando formato padrão '{fmt_tudo_final}' e adicionando extensão: {path_tudo_final}")
        else:
            # Caso 2: Usuário NÃO forneceu caminho explícito. Usaremos um nome padrão.
            validar_formato_salvar(fmt_tudo_final, 'tudo') # Valida o formato padrão escolhido
            # Constrói o nome do arquivo padrão:
            # - Usa o nome base (derivado do input ou 'resultados_cluster').
            # - Adiciona o prefixo do cluster (ex: 'cluster', 'subcluster').
            # - Adiciona o número da rodada.
            # - Adiciona a extensão do formato padrão.
            prefixo_nome = prefixo_cluster.rstrip('_') # Remove '_' final se houver
            nome_arquivo = f"{nome_base_padrao}_{prefixo_nome}_rodada{rodada_a_salvar}.{fmt_tudo_final}"
            # Junta o nome do arquivo com o diretório de saída (ou o diretório atual se None).
            path_tudo_final = os.path.join(diretorio_saida or '.', nome_arquivo)
            logging.info(f"Usando caminho padrão para o arquivo completo: {path_tudo_final}")

    # Determinar caminho/formato para o arquivo de amostras
    if o_que_salvar in ['amostras', 'ambos']:
         # --- Lógica para o arquivo de amostras (análoga à do arquivo completo) ---
         if caminho_amostras:
            # Caso 1: Usuário forneceu um caminho explícito para as amostras.
            logging.debug(f"Caminho explícito fornecido para o arquivo de amostras: {caminho_amostras}")
            _, ext = os.path.splitext(caminho_amostras)
            fmt_detectado = ext[1:].lower() if ext else None
            if fmt_detectado:
                # Se há extensão, usa-a como formato final (após validação).
                validar_formato_salvar(fmt_detectado, 'amostras') # Valida o formato detectado da extensão
                fmt_amostras_final = fmt_detectado
                path_amostras_final = caminho_amostras # Usa o caminho como está
                logging.debug(f"Formato '{fmt_amostras_final}' detectado da extensão do caminho explícito das amostras.")
            else:
                # Se não há extensão no caminho explícito, usa o formato padrão (`formato_amostras`)
                # e adiciona essa extensão ao caminho fornecido.
                validar_formato_salvar(fmt_amostras_final, 'amostras') # Valida o formato padrão
                path_amostras_final = f"{caminho_amostras}.{fmt_amostras_final}"
                logging.debug(f"Nenhuma extensão no caminho explícito das amostras. Usando formato padrão '{fmt_amostras_final}' e adicionando extensão: {path_amostras_final}")
         else:
            # Caso 2: Usuário NÃO forneceu caminho explícito para amostras. Usaremos um nome padrão.
            validar_formato_salvar(fmt_amostras_final, 'amostras') # Valida o formato padrão escolhido
            # Constrói o nome do arquivo padrão (similar ao arquivo completo, mas com '_amostras').
            prefixo_nome = prefixo_cluster.rstrip('_')
            nome_arquivo = f"{nome_base_padrao}_{prefixo_nome}_amostras_rodada{rodada_a_salvar}.{fmt_amostras_final}"
            # Junta com o diretório de saída.
            path_amostras_final = os.path.join(diretorio_saida or '.', nome_arquivo)
            logging.info(f"Usando caminho padrão para o arquivo de amostras: {path_amostras_final}")

    # Retorna os caminhos e formatos finais determinados (ou None se não aplicável)
    # Os caminhos são convertidos para absolutos para clareza.
    return {
        'path_tudo_final': os.path.abspath(path_tudo_final) if path_tudo_final else None,
        'fmt_tudo_final': fmt_tudo_final if path_tudo_final else None,
        'path_amostras_final': os.path.abspath(path_amostras_final) if path_amostras_final else None,
        'fmt_amostras_final': fmt_amostras_final if path_amostras_final else None
    }


def salvar_dataframe(df: pd.DataFrame, caminho_arquivo: str, formato: str) -> bool:
    """
    Salva um DataFrame em um arquivo no formato especificado.

    Salva uma tabela (DataFrame) em um arquivo no formato especificado.

    Suporta 'csv', 'xlsx', 'parquet', 'json'.

    Args:
        df (pd.DataFrame): A tabela de dados a ser salva.
        caminho_arquivo (str): O caminho completo onde o arquivo será salvo (incluindo nome e extensão).
        formato (str): O formato desejado ('csv', 'xlsx', 'parquet', 'json').

    Returns:
        bool: True se o arquivo foi salvo com sucesso, False caso contrário.

    Raises:
        ImportError: Se uma biblioteca necessária para o formato não estiver instalada (ex: openpyxl para xlsx).
        ValueError: Se o formato for inválido ou não suportado.
        OSError: Se houver problemas de permissão ou outros erros ao escrever o arquivo.
    """
    logging.info(f"Salvando dados em '{os.path.basename(caminho_arquivo)}' (Formato: {formato})...")
    logging.debug(f"Caminho completo para salvar: {caminho_arquivo}")
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
            # orient='records' é bom para listas de objetos JSON
            # force_ascii=False preserva caracteres acentuados
            df.to_json(caminho_arquivo, orient='records', indent=4, force_ascii=False, default_handler=str) # default_handler=str ajuda com tipos não serializáveis
        else:
            # Esta validação já deve ter ocorrido antes, mas por segurança
            raise ValueError(f"Formato de salvamento não suportado: {formato}")

        caminho_abs = os.path.abspath(caminho_arquivo)
        logging.info(f"Dados salvos com sucesso em: '{caminho_abs}'.")
        return True

    except ImportError as e:
        logging.error(f"Falha ao salvar '{caminho_arquivo}': {e}")
        raise # Re-levanta ImportError para ser tratado pelo chamador
    except Exception as e:
        logging.error(f"Erro ao salvar o arquivo '{os.path.basename(caminho_arquivo)}' (formato {formato}): {e}")
        return False

def _gerar_dataframe_amostras(df: pd.DataFrame, nome_coluna_cluster: str, num_clusters: int) -> pd.DataFrame:
    """
    Cria uma tabela (DataFrame) contendo exemplos aleatórios (até 10 por grupo) da tabela original.
    Os exemplos mantêm todas as colunas originais.
    (Função auxiliar interna para `salvar_amostras`)

    Args:
        df (pd.DataFrame): A tabela completa com os dados e a coluna de resultados do agrupamento.
        nome_coluna_cluster (str): O nome da coluna que identifica os grupos da rodada desejada.
        num_clusters (int): O número de grupos esperado nessa rodada (usado para iterar).

    Returns:
        pd.DataFrame: Tabela com os exemplos concatenados (até 10 por grupo),
                      ou uma tabela vazia se não houver exemplos válidos.
    """
    logging.debug(f"Gerando amostras para {num_clusters} grupos da coluna '{nome_coluna_cluster}'...")
    resultados = pd.DataFrame()
    # Garante que estamos lidando com grupos válidos (números inteiros não nulos)
    # Usa .unique() para obter os IDs reais presentes nos dados, em vez de range(num_clusters)
    # Isso evita avisos desnecessários se algum ID de cluster não foi gerado (ex: K muito alto)
    actual_clusters = sorted(df[nome_coluna_cluster].dropna().astype(int).unique())
    logging.debug(f"IDs de grupos encontrados nos dados: {actual_clusters}")

    if not actual_clusters:
        logging.warning(f"Nenhum ID de grupo válido encontrado na coluna '{nome_coluna_cluster}'. Não é possível gerar amostras.")
        return resultados

    for cluster_id in actual_clusters:
        df_cluster = df[df[nome_coluna_cluster] == cluster_id]
        tamanho_amostra = min(10, len(df_cluster)) # Pega no máximo 10 ou o tamanho do grupo
        if tamanho_amostra > 0:
            try:
                # Usa random_state para reprodutibilidade das amostras
                amostra_cluster = df_cluster.sample(tamanho_amostra, random_state=42)
                # A coluna de cluster já está presente, não precisa adicionar
                resultados = pd.concat([resultados, amostra_cluster], ignore_index=True)
                logging.debug(f"Retiradas {tamanho_amostra} amostras do grupo {cluster_id}.")
            except Exception as e:
                logging.error(f"Erro ao retirar amostras do grupo {cluster_id}: {e}")
                # Continua para os próximos grupos
        else:
            # Isso não deve acontecer se iterarmos por actual_clusters, mas por segurança:
            logging.warning(f"Grupo {cluster_id} está vazio ou contém apenas valores nulos. Nenhuma amostra será retirada.")

    logging.debug(f"Geração de amostras concluída. Total de {len(resultados)} amostras coletadas.")
    return resultados

def salvar_amostras(df: pd.DataFrame, nome_coluna_cluster: str, num_clusters: int, caminho_arquivo: str, formato: str) -> bool:
    """
    Gera e salva exemplos (até 10 por grupo) de uma tabela em um arquivo no formato especificado.

    Suporta os formatos 'xlsx', 'csv', 'json'.

    Args:
        df (pd.DataFrame): A tabela completa com os dados e a coluna de resultados do agrupamento.
        nome_coluna_cluster (str): O nome da coluna que identifica os grupos da rodada desejada.
        num_clusters (int): O número de grupos esperado nessa rodada (usado para gerar os exemplos).
        caminho_arquivo (str): O caminho completo onde o arquivo de exemplos será salvo.
        formato (str): O formato desejado ('xlsx', 'csv', 'json').

    Returns:
        bool: True se o arquivo foi salvo com sucesso (ou se não havia exemplos para salvar),
              False em caso de erro durante o salvamento.

    Raises:
        ImportError: Se uma biblioteca necessária para o formato não estiver instalada.
        ValueError: Se o formato for inválido.
        KeyError: Se a coluna de resultados do agrupamento não existir.
        OSError: Se houver problemas de permissão ou outros erros ao escrever o arquivo.
    """
    logging.info(f"Gerando e salvando arquivo de exemplos (até 10 por grupo) em '{os.path.basename(caminho_arquivo)}' (Formato: {formato})...")
    logging.debug(f"Caminho completo para salvar amostras: {caminho_arquivo}")
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
            logging.warning(f"Nenhum exemplo foi gerado para '{os.path.basename(caminho_arquivo)}'. O arquivo não será criado.")
            # Consideramos sucesso, pois não houve erro, apenas não havia o que salvar.
            return True # Retorna True explicitamente

    except (KeyError, ValueError, ImportError, OSError) as e:
        # Captura erros de validação, formato inválido, dependência ou IO (re-levantados por salvar_dataframe)
        logging.error(f"Erro ao preparar ou salvar o arquivo de exemplos: {e}")
        raise # Re-levanta a exceção para ser tratada pelo chamador (ClusterFacil.salvar)
    except Exception as e:
        # Captura outros erros inesperados durante a geração dos exemplos
        logging.error(f"Falha inesperada ao gerar ou salvar exemplos em '{os.path.basename(caminho_arquivo)}': {e}")
        return False # Retorna False para erros inesperados na geração

    return True # Retorna True se salvou com sucesso ou se não havia amostras

# --- Funções de Subcluster ---
def criar_df_subcluster(df: pd.DataFrame, nome_coluna_classificacao: str, classificacao_desejada: str) -> pd.DataFrame:
    """
    Filtra uma tabela (DataFrame) por uma classificação manual específica, remove colunas
    de resultados de agrupamentos anteriores e renomeia a coluna de classificação original.

    Args:
        df (pd.DataFrame): A tabela original.
        nome_coluna_classificacao (str): Nome da coluna usada para a classificação manual.
        classificacao_desejada (str): A classificação específica que você deseja usar para filtrar.

    Returns:
        pd.DataFrame: Uma nova tabela contendo apenas os dados filtrados e preparados para subcluster.

    Raises:
        KeyError: Se a coluna de classificação não existir na tabela.
        ValueError: Se a classificação desejada não for encontrada na coluna.
    """
    logging.info(f"Filtrando dados para criar o subcluster da classificação: '{classificacao_desejada}'")

    # Validação: Coluna de classificação existe?
    validar_coluna_existe(df, nome_coluna_classificacao)

    # Validação: Classificação desejada existe? (Usa dropna para evitar erro com NA)
    if classificacao_desejada not in df[nome_coluna_classificacao].dropna().unique():
        msg = f"A classificação '{classificacao_desejada}' não foi encontrada na coluna '{nome_coluna_classificacao}'. Verifique se o nome está correto."
        logging.error(msg)
        raise ValueError(msg)

    # Filtragem
    df_sub = df[df[nome_coluna_classificacao] == classificacao_desejada].copy()
    logging.info(f"Subcluster criado com {len(df_sub)} textos correspondentes à classificação '{classificacao_desejada}'.")

    # Limpeza de colunas de resultados de agrupamentos anteriores
    colunas_cluster_para_remover = []
    # Procura por colunas que comecem com 'cluster_' ou 'subcluster_' seguido por números
    regex_qualquer_cluster = re.compile(r'^(cluster_|subcluster_)\d+$')
    for col in df_sub.columns:
        if regex_qualquer_cluster.match(str(col)): # Garante que col seja string
            colunas_cluster_para_remover.append(col)

    if colunas_cluster_para_remover:
        df_sub.drop(columns=colunas_cluster_para_remover, inplace=True, errors='ignore')
        logging.info(f"Colunas de resultados de agrupamentos anteriores removidas do subcluster: {colunas_cluster_para_remover}")

    # Renomear coluna de classificação original para evitar conflito
    coluna_origem = f"{nome_coluna_classificacao}_origem"
    df_sub.rename(columns={nome_coluna_classificacao: coluna_origem}, inplace=True)
    logging.info(f"Coluna de classificação original ('{nome_coluna_classificacao}') renomeada para '{coluna_origem}' no subcluster.")

    return df_sub
