import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from typing import Optional, List, Union
from scipy.sparse import csr_matrix
import logging
import os

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar stopwords uma vez
try:
    stop_words_pt = set(stopwords.words('portuguese'))
    stop_words_pt = [word.lower() for word in stop_words_pt]
except LookupError:
    logging.error("Recurso 'stopwords' do NLTK não encontrado. Tentando baixar...")
    import nltk
    try:
        nltk.download('stopwords')
        stop_words_pt = set(stopwords.words('portuguese'))
        stop_words_pt = [word.lower() for word in stop_words_pt]
        logging.info("Download de 'stopwords' concluído.")
    except Exception as e:
        logging.error(f"Falha ao baixar 'stopwords': {e}. Stopwords em português não serão usadas.")
        stop_words_pt = []

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
    def __init__(self, entrada: Union[pd.DataFrame, str]):
        """
        Inicializa a classe ClusterFacil.

        Args:
            entrada (Union[pd.DataFrame, str]): Pode ser um DataFrame do Pandas já carregado
                                                 ou uma string contendo o caminho para um arquivo
                                                 de dados (suporta .csv, .xlsx, .parquet, .json).
                                                 O DataFrame ou arquivo deve incluir uma coluna
                                                 com os textos a serem clusterizados.

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
            self.df: pd.DataFrame = self._carregar_dados_de_arquivo(entrada)
            logging.info(f"ClusterFacil inicializado com dados do arquivo: {entrada}")
        else:
            raise TypeError("A entrada deve ser um DataFrame do Pandas ou o caminho (string) para um arquivo.")

        self.rodada_clusterizacao: int = 1
        self.coluna_textos: Optional[str] = None
        self.X: Optional[csr_matrix] = None
        self.inercias: Optional[List[float]] = None


    def _carregar_dados_de_arquivo(self, caminho_arquivo: str) -> pd.DataFrame:
        """
        Método interno para carregar dados de um arquivo usando Pandas.

        Suporta CSV, Excel (.xlsx), Parquet e JSON.

        Args:
            caminho_arquivo (str): O caminho para o arquivo de dados.

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
                df = pd.read_excel(caminho_arquivo) # Pode precisar de sheet_name='nome_da_aba' em alguns casos
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
            textos_processados = self.df[self.coluna_textos].astype(str).apply(lambda x: x.lower())
            textos_processados.fillna('', inplace=True)
        except Exception as e:
             raise TypeError(f"Erro ao processar a coluna '{coluna_textos}'. Verifique se ela contém texto. Erro original: {e}")

        logging.info("Calculando TF-IDF...")
        vectorizer = TfidfVectorizer(stop_words=stop_words_pt)
        self.X = vectorizer.fit_transform(textos_processados)
        logging.info(f"Matriz TF-IDF calculada com shape: {self.X.shape}")

        logging.info("Calculando inércias para o método do cotovelo...")
        self.inercias = []
        k_range = range(1, limite_k + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=n_init)
            kmeans.fit(self.X)
            self.inercias.append(kmeans.inertia_)
            logging.debug(f"Inércia para K={k}: {kmeans.inertia_}")

        logging.info("Gerando gráfico do método do cotovelo...")
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, self.inercias, marker='o')
        plt.title('Método do Cotovelo para Escolha de K')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Inércia (WCSS)')
        plt.xticks(k_range)
        plt.grid(True)
        plt.show()
        logging.info("Preparação concluída. Analise o gráfico do cotovelo para escolher o número de clusters.")

    def _salvar_csv(self, nome_arquivo: str) -> None:
        """
        Método interno para salvar o DataFrame principal em CSV.

        Em caso de erro, registra a falha no log, mas não levanta exceção.
        """
        logging.info(f"Tentando salvar DataFrame completo em '{nome_arquivo}'...")
        try:
            self.df.to_csv(nome_arquivo, index=False, encoding='utf-8-sig')
            logging.info(f"DataFrame salvo com sucesso em '{nome_arquivo}'.")
        except Exception as e:
            logging.error(f"Falha ao salvar o arquivo CSV '{nome_arquivo}': {e}")

    def _salvar_amostras_excel(self, nome_arquivo: str, nome_coluna_cluster: str, num_clusters: int) -> None:
        """
        Método interno para gerar e salvar amostras (até 10 por cluster) em Excel.

        Em caso de erro, registra a falha no log, mas não levanta exceção.
        """
        logging.info(f"Tentando gerar e salvar amostras (até 10 por cluster) em '{nome_arquivo}'...")
        resultados = pd.DataFrame()
        try:
            for cluster_id in range(num_clusters):
                df_cluster = self.df[self.df[nome_coluna_cluster] == cluster_id]
                tamanho_amostra = min(10, len(df_cluster))
                if tamanho_amostra > 0:
                    amostra_cluster = df_cluster.sample(tamanho_amostra, random_state=42)
                    amostra_cluster['cluster_original_id'] = cluster_id
                    resultados = pd.concat([resultados, amostra_cluster], ignore_index=True)
                else:
                     logging.warning(f"Cluster {cluster_id} está vazio, nenhuma amostra será retirada.")

            if not resultados.empty:
                resultados.to_excel(nome_arquivo, index=False)
                logging.info(f"Amostras salvas com sucesso em '{nome_arquivo}'.")
            else:
                 logging.warning("Nenhuma amostra foi gerada (todos os clusters estavam vazios?). Arquivo Excel não será criado.")

        except Exception as e:
            logging.error(f"Falha ao gerar ou salvar o arquivo Excel de amostras '{nome_arquivo}': {e}")

    def finaliza(self, num_clusters: int, prefixo_saida: str = '') -> None:
        """
        Executa a clusterização K-Means, adiciona a coluna de clusters ao DataFrame
        e tenta salvar os resultados (DataFrame completo em CSV e amostras em Excel).

        O K-Means final é executado com n_init=10 (padrão recomendado).
        Erros durante o salvamento dos arquivos são registrados no log, mas não
        interrompem a execução do método.
        Incrementa o contador `self.rodada_clusterizacao` ao final.

        Args:
            num_clusters (int): O número de clusters (K) a ser usado.
            prefixo_saida (str, optional): Prefixo para os nomes dos arquivos de saída. Default ''.

        Raises:
            RuntimeError: Se `preparar` não foi executado antes (self.X ou self.coluna_textos são None).
            ValueError: Se `num_clusters` for inválido (não positivo ou maior que o número de amostras).
        """
        logging.info(f"Iniciando finalização com K={num_clusters} e prefixo='{prefixo_saida}'.")
        if self.X is None or self.coluna_textos is None:
            raise RuntimeError("O método 'preparar' deve ser executado antes de 'finaliza'.")
        if not isinstance(num_clusters, int) or num_clusters <= 0:
            raise ValueError("O argumento 'num_clusters' deve ser um inteiro positivo.")
        if num_clusters > self.X.shape[0]:
             raise ValueError(f"O número de clusters ({num_clusters}) não pode ser maior que o número de amostras ({self.X.shape[0]}).")

        logging.info(f"Executando K-Means com {num_clusters} clusters...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.X)

        nome_coluna_cluster = f'cluster_{self.rodada_clusterizacao}'
        self.df[nome_coluna_cluster] = cluster_labels
        logging.info(f"Coluna '{nome_coluna_cluster}' adicionada ao DataFrame.")

        # Definir nomes de arquivo (garantindo que o prefixo funcione bem)
        prefixo_fmt = f"{prefixo_saida}_" if prefixo_saida else ""
        nome_csv = f"{prefixo_fmt}clusters_{self.rodada_clusterizacao}.csv"
        nome_excel = f"{prefixo_fmt}amostras_por_cluster_{self.rodada_clusterizacao}.xlsx"

        # Chamar métodos internos para salvar
        self._salvar_csv(nome_csv)
        self._salvar_amostras_excel(nome_excel, nome_coluna_cluster, num_clusters)

        logging.info(f"Finalização concluída para a rodada {self.rodada_clusterizacao}.")
        self.rodada_clusterizacao += 1
