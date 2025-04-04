import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import Optional, List, Union
from scipy.sparse import csr_matrix
import logging
import os

from .utils import stop_words_pt, calcular_e_plotar_cotovelo, salvar_dataframe_csv, salvar_amostras_excel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClusterFacil():
    """
    Facilita a clusterização de textos em DataFrames do Pandas.

    Esta classe encapsula os passos comuns de pré-processamento de texto (TF-IDF),
    análise do número ideal de clusters (método do cotovelo) e a aplicação
    do algoritmo K-Means para agrupar os textos.

        Atributos:
            df (pd.DataFrame): O DataFrame com os dados e resultados da clusterização.
            rodada_clusterizacao (int): Contador para nomear arquivos de saída e colunas de cluster,
                                       inicializado em 1 e incrementado a cada chamada de `finaliza`.
            coluna_textos (Optional[str]): Nome da coluna contendo os textos a serem clusterizados.
            X (Optional[csr_matrix]): Matriz TF-IDF resultante do pré-processamento.
            inercias (Optional[List[float]]): Lista de inercias calculadas para diferentes K no método do cotovelo.
    """
    def __init__(self, entrada: Union[pd.DataFrame, str], aba: Optional[str] = None):
        """
        Inicializa a classe ClusterFacil.

        Args:
            entrada (Union[pd.DataFrame, str]): Pode ser um DataFrame do Pandas já carregado
                                                 ou uma string contendo o caminho para um arquivo
                                                 de dados (suporta .csv, .xlsx, .parquet, .json).
                                                 O DataFrame ou arquivo deve incluir uma coluna
                                                 com os textos a serem clusterizados.
            aba (Optional[str], optional): O nome ou índice da aba a ser lida caso a entrada
                                           seja um caminho para um arquivo Excel (.xlsx).
                                           Se None (padrão), lê a primeira aba. Default é None.

        Raises:
            TypeError: Se a entrada não for um DataFrame ou uma string.
            FileNotFoundError: Se a entrada for uma string e o arquivo não for encontrado.
            ImportError: Se uma dependência necessária (ex: openpyxl, pyarrow) não estiver instalada.
            ValueError: Se o formato do arquivo não for suportado ou houver erro na leitura.
        """
        if isinstance(entrada, pd.DataFrame):
            self.df: pd.DataFrame = entrada.copy() # Copiar para evitar modificar o original inesperadamente
            logging.info("ClusterFacil inicializado com DataFrame existente.")
        elif isinstance(entrada, str):
            self.df: pd.DataFrame = self._carregar_dados_de_arquivo(entrada, aba=aba) # Passa o parâmetro aba
            logging.info(f"ClusterFacil inicializado com dados do arquivo: {entrada}" + (f" (aba: {aba})" if aba else ""))
        else:
            raise TypeError("A entrada deve ser um DataFrame do Pandas ou o caminho (string) para um arquivo.")

        self.rodada_clusterizacao: int = 1
        self.coluna_textos: Optional[str] = None
        self.X: Optional[csr_matrix] = None
        self.inercias: Optional[List[float]] = None
        self._ultimo_num_clusters: Optional[int] = None
        self._ultima_coluna_cluster: Optional[str] = None

    def _carregar_dados_de_arquivo(self, caminho_arquivo: str, aba: Optional[str] = None) -> pd.DataFrame:
        """
        Método interno para carregar dados de um arquivo usando Pandas.

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
        if not os.path.exists(caminho_arquivo):
            logging.error(f"Arquivo não encontrado: {caminho_arquivo}")
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

        _, extensao = os.path.splitext(caminho_arquivo)
        extensao = extensao.lower()

        try:
            if extensao == '.csv':
                df = pd.read_csv(caminho_arquivo)
            elif extensao == '.xlsx':
                # Tenta importar openpyxl para dar um erro mais informativo se faltar
                try:
                    import openpyxl
                except ImportError:
                     logging.error("A biblioteca 'openpyxl' é necessária para ler arquivos .xlsx. Instale-a com 'pip install openpyxl'")
                     raise ImportError("A biblioteca 'openpyxl' é necessária para ler arquivos .xlsx.")
                # Usa o parâmetro 'aba' (sheet_name) ao ler o Excel
                df = pd.read_excel(caminho_arquivo, sheet_name=aba)
            elif extensao == '.parquet':
                 # Tenta importar pyarrow para dar um erro mais informativo se faltar
                try:
                    import pyarrow
                except ImportError:
                     logging.error("A biblioteca 'pyarrow' é necessária para ler arquivos .parquet. Instale-a com 'pip install pyarrow'")
                     raise ImportError("A biblioteca 'pyarrow' é necessária para ler arquivos .parquet.")
                df = pd.read_parquet(caminho_arquivo)
            elif extensao == '.json':
                df = pd.read_json(caminho_arquivo)
            else:
                logging.error(f"Formato de arquivo não suportado: {extensao}")
                raise ValueError(f"Formato de arquivo não suportado: {extensao}. Suportados: .csv, .xlsx, .parquet, .json")

            logging.info(f"Arquivo {caminho_arquivo} carregado com sucesso. Shape: {df.shape}")
            return df
        except FileNotFoundError: # Redundante devido à verificação inicial, mas seguro
             logging.error(f"Erro: Arquivo não encontrado ao tentar ler: {caminho_arquivo}")
             raise
        except ImportError as e: # Captura especificamente erros de importação de dependências
             logging.error(f"Erro de importação ao tentar ler {caminho_arquivo}: {e}")
             raise
        except Exception as e:
            logging.error(f"Erro ao ler o arquivo {caminho_arquivo} (formato {extensao}): {e}")
            # Pode ser útil levantar um erro mais específico ou apenas o genérico
            raise ValueError(f"Erro ao processar o arquivo {caminho_arquivo}: {e}")

    def preparar(self, coluna_textos: str, limite_k: int = 10, n_init = 1) -> None:
        """
        Prepara os dados para clusterização.

        Realiza o pré-processamento dos textos (lowercase, preenchimento de nulos),
        calcula a matriz TF-IDF e gera o gráfico do método do cotovelo (usando `plt.show()`)
        para ajudar na escolha do número ideal de clusters (K).

        Args:
            coluna_textos (str): O nome da coluna no DataFrame que contém os textos.
            limite_k (int, optional): O número máximo de clusters (K) a ser testado
                                       no método do cotovelo. Default é 10.
            n_init (int, optional): O número de inicializações do K-Means ao calcular
                                    as inércias para o gráfico do cotovelo. Default é 1.

        Raises:
            KeyError: Se o nome da coluna `coluna_textos` não existir no DataFrame.
            ValueError: Se `limite_k` não for um inteiro positivo.
            TypeError: Se a coluna especificada não contiver dados textuais (ou que possam ser convertidos para string).
        """
        logging.info(f"Iniciando preparação com a coluna '{coluna_textos}' e limite K={limite_k}.")
        if coluna_textos not in self.df.columns:
            raise KeyError(f"A coluna '{coluna_textos}' não foi encontrada no DataFrame.")
        if not isinstance(limite_k, int) or limite_k <= 0:
            raise ValueError("O argumento 'limite_k' deve ser um inteiro positivo.")

        self.coluna_textos = coluna_textos

        try:
            # Garante que a coluna seja string e preenche NaNs com string vazia antes do lower()
            textos_processados = self.df[self.coluna_textos].fillna('').astype(str).str.lower()
        except Exception as e:
             raise TypeError(f"Erro ao processar a coluna '{coluna_textos}'. Verifique se ela contém texto. Erro original: {e}")

        logging.info("Calculando TF-IDF...")
        # Usa stop_words_pt importado de utils.py
        vectorizer = TfidfVectorizer(stop_words=stop_words_pt)
        self.X = vectorizer.fit_transform(textos_processados)
        logging.info(f"Matriz TF-IDF calculada com shape: {self.X.shape}")

        # Chama a função auxiliar para calcular inércias e plotar o gráfico
        self.inercias = calcular_e_plotar_cotovelo(self.X, limite_k, n_init)

        if self.inercias is not None:
             logging.info("Preparação concluída. Analise o gráfico do cotovelo para escolher o número de clusters.")
        else:
             # A função auxiliar já logou o erro/aviso
             logging.info("Preparação concluída (sem dados para o método do cotovelo).")

    def clusterizar(self, num_clusters: int) -> str:
        """
        Executa a clusterização K-Means e adiciona a coluna de clusters ao DataFrame.

        Args:
            num_clusters (int): O número de clusters (K) a ser usado.

        Returns:
            str: O nome da coluna de cluster criada (ex: 'cluster_1').

        Raises:
            RuntimeError: Se `preparar` não foi executado antes ou se não há dados.
            ValueError: Se `num_clusters` for inválido.
            Exception: Outros erros durante a execução do K-Means.
        """
        logging.info(f"Iniciando clusterização com K={num_clusters} para a rodada {self.rodada_clusterizacao}.")
        if self.X is None or self.coluna_textos is None:
            raise RuntimeError("O método 'preparar' deve ser executado antes de 'clusterizar'.")
        if self.X.shape[0] == 0:
             raise RuntimeError("Não há dados para clusterizar. Execute 'preparar' com um DataFrame que contenha dados.")
        if not isinstance(num_clusters, int) or num_clusters <= 0:
            raise ValueError("O argumento 'num_clusters' deve ser um inteiro positivo.")
        if num_clusters > self.X.shape[0]:
             raise ValueError(f"O número de clusters ({num_clusters}) não pode ser maior que o número de amostras ({self.X.shape[0]}).")

        logging.info(f"Executando K-Means com {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        try:
            cluster_labels = kmeans.fit_predict(self.X)
        except Exception as e:
             logging.error(f"Erro durante a execução do K-Means: {e}")
             raise # Re-levanta a exceção para indicar falha crítica

        nome_coluna_cluster = f'cluster_{self.rodada_clusterizacao}'
        self.df[nome_coluna_cluster] = cluster_labels
        logging.info(f"Coluna '{nome_coluna_cluster}' adicionada ao DataFrame.")

        # Armazena informações da última rodada
        self._ultimo_num_clusters = num_clusters
        self._ultima_coluna_cluster = nome_coluna_cluster

        self.rodada_clusterizacao += 1 # Incrementa a rodada APÓS sucesso
        logging.info(f"Clusterização da rodada {self.rodada_clusterizacao - 1} concluída.")
        return nome_coluna_cluster

    def salvar(self, prefixo_saida: str = '', diretorio_saida: Optional[str] = None) -> dict[str, bool]:
        """
        Salva os resultados da última clusterização realizada (DataFrame completo em CSV e amostras em Excel).

        Args:
            prefixo_saida (str, optional): Prefixo para os nomes dos arquivos de saída. Default ''.
            diretorio_saida (Optional[str], optional): Caminho da pasta onde salvar os arquivos.
                                                      Se None (padrão), salva no diretório atual. Default None.

        Returns:
            dict[str, bool]: Um dicionário indicando o sucesso do salvamento de cada arquivo.
                             Ex: {'csv_salvo': True, 'excel_salvo': False}
        """
        logging.info("Tentando salvar resultados da última clusterização...")

        if self._ultima_coluna_cluster is None or self._ultimo_num_clusters is None:
            logging.error("Nenhuma clusterização foi realizada ainda. Execute o método 'clusterizar' primeiro.")
            return {'csv_salvo': False, 'excel_salvo': False}

        # Usa a informação da última rodada (rodada_clusterizacao já foi incrementada)
        rodada_a_salvar = self.rodada_clusterizacao - 1
        nome_coluna_cluster = self._ultima_coluna_cluster
        num_clusters = self._ultimo_num_clusters

        if nome_coluna_cluster not in self.df.columns:
            # Verificação de segurança, embora improvável se a lógica estiver correta
            logging.error(f"Erro interno: Coluna '{nome_coluna_cluster}' da última clusterização não encontrada.")
            return {'csv_salvo': False, 'excel_salvo': False}

         # Definir nomes de arquivo base
        prefixo_fmt = f"{prefixo_saida}_" if prefixo_saida else ""
        nome_base_csv = f"{prefixo_fmt}clusters_{rodada_a_salvar}.csv"
        nome_base_excel = f"{prefixo_fmt}amostras_por_cluster_{rodada_a_salvar}.xlsx"

        # Construir caminho final usando o parâmetro diretorio_saida
        if diretorio_saida:
            try:
                os.makedirs(diretorio_saida, exist_ok=True)
                logging.info(f"Diretório de saída '{diretorio_saida}' verificado/criado.")
                nome_csv = os.path.join(diretorio_saida, nome_base_csv)
                nome_excel = os.path.join(diretorio_saida, nome_base_excel)
            except OSError as e:
                logging.error(f"Não foi possível criar ou acessar o diretório de saída '{diretorio_saida}': {e}. Salvando no diretório atual.")
                # Fallback para o diretório atual em caso de erro
                nome_csv = nome_base_csv
                nome_excel = nome_base_excel
        else:
            # Se nenhum diretório foi especificado, usa o diretório atual
            nome_csv = nome_base_csv
            nome_excel = nome_base_excel

        logging.info(f"Caminho final para CSV: {nome_csv}")
        logging.info(f"Caminho final para Excel: {nome_excel}")

        # Chamar funções de utils.py para salvar
        sucesso_csv = salvar_dataframe_csv(self.df, nome_csv)
        sucesso_excel = salvar_amostras_excel(self.df, nome_coluna_cluster, num_clusters, nome_excel)

        # A lógica de warning já está dentro das funções em utils.py, mas podemos manter aqui se quisermos um log adicional
        if not sucesso_csv:
            logging.warning(f"Salvamento do CSV ({nome_csv}) falhou (ver logs anteriores).")
        if not sucesso_excel:
            logging.warning(f"Houve um problema ao salvar o arquivo Excel de amostras: {nome_excel}. Verifique os logs.")

        status_salvamento = {'csv_salvo': sucesso_csv, 'excel_salvo': sucesso_excel}
        logging.info(f"Tentativa de salvamento da rodada {rodada_a_salvar} concluída. Status: {status_salvamento}")
        return status_salvamento

    # --- MÉTODO DE CONVENIÊNCIA ---

    def finaliza(self, num_clusters: int, prefixo_saida: str = '', diretorio_saida: Optional[str] = None) -> dict[str, bool]:
        """
        Método de conveniência que executa a clusterização e depois salva os resultados.

        Equivalente a chamar `clusterizar()` seguido por `salvar()`.
        Erros durante o salvamento dos arquivos são registrados no log e indicados no
        retorno, mas não interrompem a execução.

        Args:
            num_clusters (int): O número de clusters (K) a ser usado.
            prefixo_saida (str, optional): Prefixo para os nomes dos arquivos de saída. Default ''.
            diretorio_saida (Optional[str], optional): Caminho da pasta onde salvar os arquivos.
                                                      Se None (padrão), salva no diretório atual. Default None.

        Returns:
            dict[str, bool]: O status do salvamento dos arquivos (ver método `salvar`).

        Raises:
            RuntimeError: Se `preparar` não foi executado antes ou se não há dados.
            ValueError: Se `num_clusters` for inválido.
            Exception: Outros erros durante a execução do K-Means (vindos de `clusterizar`).
        """
        logging.info(f"Executando 'finaliza' (clusterizar e salvar) com K={num_clusters} e prefixo='{prefixo_saida}'.")
        # 1. Clusterizar (pode levantar exceção)
        self.clusterizar(num_clusters)

        # 2. Salvar (não levanta exceção por falha de IO, apenas loga e retorna status)
        status_salvamento = self.salvar(prefixo_saida=prefixo_saida, diretorio_saida=diretorio_saida)

        logging.info(f"'Finaliza' concluído para a rodada {self.rodada_clusterizacao - 1}. Status de salvamento: {status_salvamento}")
        return status_salvamento # Retorna o status do salvamento
