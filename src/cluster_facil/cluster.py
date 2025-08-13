import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import Optional, Self
from scipy.sparse import csr_matrix
import logging
import re

from .utils import (
    STOPWORDS_PT,
    calcular_e_plotar_cotovelo,
    salvar_dataframe,
    salvar_amostras,
    carregar_dados,
    determinar_caminhos_saida,
    ajustar_rodada_inicial,
    criar_df_subcluster
)
from .validations import (
    validar_entrada_inicial,
    validar_coluna_existe,
    validar_inteiro_positivo,
    validar_tipo_coluna_texto,
    validar_estado_preparado,
    validar_parametro_num_clusters,
    validar_estado_clusterizado,
    validar_coluna_cluster_existe,
    validar_rodada_valida,
    validar_cluster_ids_presentes,
    validar_tipo_classificacao,
    validar_opcao_salvar
)

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
            inercias (Optional[list[float]]): Lista de inercias calculadas para diferentes K no método do cotovelo.
            prefixo_cluster (str): Prefixo usado para nomear as colunas de cluster (ex: 'cluster_', 'subcluster_').
            nome_coluna_classificacao (str): Nome da coluna usada para armazenar as classificações manuais.
            random_state (Optional[int]): Semente para geradores de números aleatórios usada internamente.
    """
    def __init__(self,
                 entrada: pd.DataFrame | str,
                 aba: Optional[str] = None,
                 prefixo_cluster: str = "cluster_",
                 nome_coluna_classificacao: str = "classificacao",
                 random_state: Optional[int] = 42):
        """
        Inicializa a classe ClusterFacil.

        Args:
            entrada (pd.DataFrame | str): Pode ser um DataFrame do Pandas já carregado
                                          ou uma string contendo o caminho para um arquivo
                                          de dados (suporta .csv, .xlsx, .parquet, .json).
                                          O DataFrame ou arquivo deve incluir uma coluna
                                                 com os textos a serem clusterizados.
            aba (Optional[str], optional): O nome ou índice da aba a ser lida caso a entrada
                                           seja um caminho para um arquivo Excel (.xlsx).
                                           Se None (padrão), lê a primeira aba. Padrão é None.
            prefixo_cluster (str, optional): Prefixo para as colunas de cluster. Padrão é "cluster_".
            nome_coluna_classificacao (str, optional): Nome da coluna de classificação. Padrão é "classificacao".
            random_state (Optional[int], optional): Semente para geradores de números aleatórios,
                                                    garantindo reprodutibilidade. Padrão é 42.


        Raises:
            TypeError: Se a entrada não for um DataFrame ou uma string.
            FileNotFoundError: Se a entrada for uma string e o arquivo não for encontrado.
            ImportError: Se uma dependência necessária (ex: openpyxl, pyarrow) não estiver instalada.
            ValueError: Se o formato do arquivo não for suportado ou houver erro na leitura.
        """
        validar_entrada_inicial(entrada)

        if isinstance(entrada, pd.DataFrame):
            self.df: pd.DataFrame = entrada.copy() # Copiar para evitar modificar o original inesperadamente
            self._input_path: Optional[str] = None
            logging.info("ClusterFacil iniciado com um DataFrame já existente.")
        elif isinstance(entrada, str): # Sabemos que é string por causa da validação
            self.df: pd.DataFrame = carregar_dados(entrada, aba=aba)
            logging.info(f"ClusterFacil iniciado com dados do arquivo: {entrada}" + (f" (aba: {aba})" if aba else ""))
            self._input_path: Optional[str] = entrada # Guarda o caminho de entrada

        self.prefixo_cluster: str = prefixo_cluster
        self.nome_coluna_classificacao: str = nome_coluna_classificacao
        self.rodada_clusterizacao: int = 1
        self.coluna_textos: Optional[str] = None
        self.X: Optional[csr_matrix] = None # Matriz TF-IDF da última operação relevante
        self.inercias: Optional[list[float]] = None
        self._ultimo_num_clusters: Optional[int] = None
        self._ultima_coluna_cluster: Optional[str] = None
        self._vectorizer: Optional[TfidfVectorizer] = None # Guardar o vectorizer para reuso
        self._tfidf_kwargs: Optional[dict] = None # Guardar kwargs do TF-IDF para referência
        self._indices_preparados_na_rodada: Optional[pd.Index] = None # Índices usados na última preparação
        self.random_state = random_state # Armazena o random_state

        # Ajustar rodada_clusterizacao com base nas colunas existentes
        self.rodada_clusterizacao = ajustar_rodada_inicial(self.df.columns, self.prefixo_cluster)
        logging.info(f"Próxima rodada de clusterização definida como: {self.rodada_clusterizacao}")

    # --- Métodos Privados Auxiliares ---
    def _atribuir_labels_cluster(self, cluster_labels: list[int], indices_alvo: pd.Index, nome_coluna_cluster: str) -> None:
        """
        Atribui os rótulos de cluster ao DataFrame principal.

        Cria ou atualiza a coluna de cluster especificada, atribuindo os labels
        apenas às linhas correspondentes aos índices fornecidos. Garante que a
        coluna tenha o tipo Int64Dtype (inteiro nullable).

        Args:
            cluster_labels (list[int]): Lista de rótulos de cluster retornados pelo K-Means.
            indices_alvo (pd.Index): Índices do DataFrame original onde os labels devem ser atribuídos.
            nome_coluna_cluster (str): Nome da coluna de cluster a ser criada/atualizada.
        """
        logging.debug(f"Atribuindo {len(cluster_labels)} labels à coluna '{nome_coluna_cluster}' nos índices alvo.")
        # Inicializa a coluna com NA para garantir que linhas não clusterizadas (se houver) fiquem NA
        # ou para limpar valores de uma rodada anterior se a coluna já existir (embora não devesse pelo nome)
        if nome_coluna_cluster not in self.df.columns:
            self.df[nome_coluna_cluster] = pd.NA
        else:
            # Se a coluna já existe (improvável, mas seguro), preenche com NA antes de atribuir
            self.df[nome_coluna_cluster] = pd.NA

        # Atribui os labels apenas às linhas que foram clusterizadas, usando os índices originais
        self.df.loc[indices_alvo, nome_coluna_cluster] = cluster_labels
        # Converte para inteiro nullable para consistência
        try:
            self.df[nome_coluna_cluster] = self.df[nome_coluna_cluster].astype(pd.Int64Dtype())
            logging.debug(f"Coluna '{nome_coluna_cluster}' convertida para Int64Dtype.")
        except Exception as e:
            logging.error(f"Falha ao converter a coluna de resultados '{nome_coluna_cluster}' para o tipo numérico ideal: {e}.")

        logging.info(f"Coluna de resultados '{nome_coluna_cluster}' adicionada/atualizada.")

    def _garantir_coluna_classificacao(self) -> None:
        """
        Garante que a coluna de classificação (definida em self.nome_coluna_classificacao)
        exista no DataFrame e tenha um tipo adequado (StringDtype).

        Se a coluna não existir, ela é criada com pd.NA e tipo StringDtype.
        Se existir mas não for string/object, tenta convertê-la para StringDtype.
        """
        col_classif = self.nome_coluna_classificacao # Nome da coluna para guardar classificações manuais
        if col_classif not in self.df.columns:
            logging.info(f"Coluna de classificação '{col_classif}' não encontrada. Criando coluna para futuras classificações.")
            # Criar diretamente com tipo que aceita nulos e strings
            self.df[col_classif] = pd.Series(pd.NA, index=self.df.index, dtype=pd.StringDtype())
        else:
            # Se existe, garante que seja um tipo adequado (string ou object)
            if not pd.api.types.is_string_dtype(self.df[col_classif]) and not pd.api.types.is_object_dtype(self.df[col_classif]):
                 logging.warning(f"Coluna de classificação '{col_classif}' existe, mas não é do tipo texto ({self.df[col_classif].dtype}). Tentando converter.")
                 try:
                      # Tenta converter preservando NAs
                      self.df[col_classif] = self.df[col_classif].astype(pd.StringDtype())
                      logging.info(f"Coluna de classificação '{col_classif}' convertida com sucesso para o tipo texto.")
                 except Exception as e:
                      # Se a conversão falhar (tipos muito mistos), loga erro mas continua.
                      # A atribuição na função classificar ainda pode funcionar dependendo do caso.
                      logging.error(f"Falha ao converter coluna de classificação '{col_classif}' existente para o tipo texto: {e}. A classificação pode não funcionar como esperado.")
            else:
                logging.debug(f"Coluna de classificação '{col_classif}' já existe e possui tipo texto adequado.")

    # --- Métodos Públicos ---
    def preparar(self, coluna_textos: str, limite_k: int = 10, n_init: str | int = 'auto', plotar_cotovelo: bool = True, **tfidf_kwargs) -> None: # n_init default agora é 'auto'
        """
        Prepara os dados para a próxima rodada de agrupamento (clusterização).

        Realiza a análise inicial dos textos (TF-IDF) e calcula as opções de agrupamento
        (método do cotovelo) para ajudar na escolha do número ideal de grupos (K).
        Opcionalmente, exibe um gráfico (cotovelo) para visualizar essas opções.

        **Como funciona em múltiplas rodadas:**
        *   **Primeira rodada:** Analisa todos os textos do conjunto de dados.
        *   **Rodadas seguintes:** Se você já classificou alguns grupos manualmente na coluna
            de classificação (padrão: 'classificacao'), este método analisará **apenas**
            os textos **ainda não classificados**. A análise e o gráfico do cotovelo
            serão baseados somente nesses textos restantes.

        Nota: A exibição automática do gráfico (`plotar_cotovelo=True`) funciona melhor
        em ambientes interativos como Jupyter Notebooks. Se estiver usando em um script,
        pode ser melhor definir `plotar_cotovelo=False`.

        Args:
            coluna_textos (str): O nome da coluna no seu DataFrame que contém os textos a serem agrupados.
            limite_k (int, optional): O número máximo de grupos (K) a serem testados
                                      para o gráfico do cotovelo. Padrão é 10.
            n_init (str | int, optional): Define como o K-Means é inicializado para o cálculo
                                          do cotovelo. 'auto' (padrão) geralmente executa 10 vezes
                                          e escolhe o melhor resultado, buscando mais robustez.
                                          Um número inteiro (ex: 1) executa um número fixo de vezes.
            plotar_cotovelo (bool, optional): Se True (padrão), mostra o gráfico do cotovelo
                                              após a análise. Padrão é True.
            **tfidf_kwargs: Outras configurações avançadas para a análise de texto (TF-IDF).
                            Permite ajustar parâmetros como `min_df`, `max_df`, `ngram_range`, etc.
                            Permite configurar parâmetros como `min_df`, `max_df`, `ngram_range`, etc.
                            Ex: `preparar(..., min_df=5, ngram_range=(1, 2))`

        Raises:
            KeyError: Se a coluna de texto informada (`coluna_textos`) não for encontrada.
            ValueError: Se `limite_k` não for um número positivo, ou se não houver textos
                        não classificados para analisar em rodadas posteriores à primeira.
            TypeError: Se a coluna de texto não contiver dados do tipo texto.
            ImportError: Se a biblioteca 'matplotlib' for necessária (`plotar_cotovelo=True`) e não estiver instalada.
        """
        logging.info(f"Iniciando preparação dos textos para agrupamento (rodada {self.rodada_clusterizacao}). Coluna de texto: '{coluna_textos}', limite de grupos para teste: {limite_k}.")

        validar_coluna_existe(self.df, coluna_textos)
        validar_inteiro_positivo('limite_k', limite_k)
        validar_tipo_coluna_texto(self.df, coluna_textos)

        self.coluna_textos = coluna_textos # Armazena para referência futura
        self._tfidf_kwargs = tfidf_kwargs # Armazena os kwargs passados

        # --- Lógica de Filtragem para Rodadas > 1 ---
        df_para_preparar = self.df
        indices_para_preparar = self.df.index

        if self.rodada_clusterizacao > 1 and self.nome_coluna_classificacao in self.df.columns:
            linhas_nao_classificadas = self.df[self.nome_coluna_classificacao].isna()
            if not linhas_nao_classificadas.all(): # Se houver alguma linha classificada
                logging.info(f"Rodada {self.rodada_clusterizacao}: Filtrando apenas os textos ainda não classificados na coluna '{self.nome_coluna_classificacao}'.")
                df_para_preparar = self.df.loc[linhas_nao_classificadas].copy()
                indices_para_preparar = df_para_preparar.index
                logging.info(f"Encontrados {len(df_para_preparar)} textos não classificados para esta rodada de análise.")

                if df_para_preparar.empty:
                    msg = f"Nenhum texto não classificado encontrado para analisar na rodada {self.rodada_clusterizacao}."
                    logging.warning(msg + " A preparação será interrompida.")
                    # Limpa X e índices para indicar que não há o que clusterizar
                    self.X = None
                    self.inercias = None
                    self._indices_preparados_na_rodada = None
                    # Poderia levantar um erro ou apenas avisar. Vamos avisar e limpar.
                    # raise ValueError(msg) # Alternativa: falhar se não houver dados
                    return # Termina a preparação aqui
            else:
                logging.info(f"Rodada {self.rodada_clusterizacao}: Coluna de classificação '{self.nome_coluna_classificacao}' existe, mas todos os textos estão sem classificação. Analisando todos os textos.")
        elif self.rodada_clusterizacao > 1:
             logging.info(f"Rodada {self.rodada_clusterizacao}: Coluna de classificação '{self.nome_coluna_classificacao}' não encontrada. Analisando todos os textos.")

        # --- Processamento e TF-IDF (no DataFrame selecionado) ---
        textos_processados = df_para_preparar[self.coluna_textos].fillna('').astype(str).str.lower()

        logging.info(f"Analisando características de {len(df_para_preparar)} textos (TF-IDF)...")
        # Define parâmetros padrão que podem ser sobrescritos pelos kwargs
        default_tfidf_params = {'stop_words': STOPWORDS_PT}
        final_tfidf_kwargs = {**default_tfidf_params, **self._tfidf_kwargs}
        logging.debug(f"Parâmetros finais para TfidfVectorizer: {final_tfidf_kwargs}") # Movido para DEBUG

        # Cria um novo vectorizer a cada chamada de 'preparar'.
        # Porquê? Para garantir que o vocabulário (features) do TF-IDF seja aprendido
        # *especificamente* a partir dos textos que estão sendo processados *nesta rodada*.
        # Isso é crucial em rodadas > 1, onde textos já classificados são filtrados.
        # Reutilizar um vectorizer antigo poderia incluir features de textos que não
        # estão mais sendo considerados, distorcendo a análise dos textos restantes.
        self._vectorizer = TfidfVectorizer(**final_tfidf_kwargs)
        self.X = self._vectorizer.fit_transform(textos_processados)
        self._indices_preparados_na_rodada = indices_para_preparar # Armazena os índices correspondentes a X
        logging.info(f"Análise TF-IDF concluída. {self.X.shape[0]} textos processados, {self.X.shape[1]} características (palavras/termos) identificadas.")
        logging.debug(f"Shape da matriz TF-IDF: {self.X.shape}") # Detalhe técnico para DEBUG

        # --- Método do Cotovelo (no subset preparado) ---
        if self.X.shape[0] > 0: # Só calcula se houver dados
            logging.info("Avaliando diferentes números de grupos (método do cotovelo)...")
            # Passa n_init e random_state para a função do cotovelo
            self.inercias = calcular_e_plotar_cotovelo(self.X, limite_k, n_init=n_init, plotar=plotar_cotovelo, random_state=self.random_state) # Passa n_init explicitamente
            if self.inercias is not None:
                if plotar_cotovelo:
                    logging.info(f"Preparação da rodada {self.rodada_clusterizacao} concluída. Analise o gráfico do cotovelo (baseado em {self.X.shape[0]} textos) para escolher o número de grupos (K).")
                else:
                    logging.info(f"Preparação da rodada {self.rodada_clusterizacao} concluída. Opções de agrupamento calculadas (baseado em {self.X.shape[0]} textos), gráfico não exibido.")
            else:
                 # A função calcular_e_plotar_cotovelo já loga erro se não houver amostras
                 logging.warning(f"Preparação da rodada {self.rodada_clusterizacao} concluída, mas não foi possível calcular as opções de agrupamento (método do cotovelo).")
        else:
            # Caso df_para_preparar não estivesse vazio mas textos_processados sim (improvável)
            logging.warning(f"Preparação da rodada {self.rodada_clusterizacao} concluída, mas sem textos válidos para analisar ou calcular opções de agrupamento.")
            self.inercias = None
            self.X = None # Garante que X esteja None
            self._indices_preparados_na_rodada = None # Garante que índices estejam None

    def clusterizar(self, num_clusters: int, **kmeans_kwargs) -> str:
        """
        Executa a clusterização K-Means nos dados preparados pela última chamada a `preparar`.

        Adiciona a coluna de clusters ao DataFrame original, apenas para as linhas
        que foram incluídas na última preparação.

        Args:
            num_clusters (int): O número de clusters (K) a ser usado.
            **kmeans_kwargs: Argumentos de palavra-chave adicionais a serem passados
                             diretamente para o construtor `sklearn.cluster.KMeans`.
                             Permite configurar parâmetros como `max_iter`, `tol`, etc.
                             O `n_clusters` é definido pelo argumento obrigatório, e
                             `random_state` e `n_init` têm padrões internos, mas todos
                             podem ser sobrescritos se incluídos nos `kmeans_kwargs`.
                             Ex: `clusterizar(..., max_iter=500, tol=1e-5)`

        Returns:
            str: O nome da coluna de cluster criada (ex: 'cluster_1', 'subcluster_2').

        Raises:
            RuntimeError: Se `preparar` não foi executado com sucesso antes (self.X ou
                          self._indices_preparados_na_rodada estão None), ou se não há dados
                          preparados para clusterizar.
            ValueError: Se `num_clusters` for inválido para o número de amostras preparadas.
            Exception: Outros erros durante a execução do K-Means.
        """
        logging.info(f"Iniciando agrupamento (clusterização) em {num_clusters} grupos para a rodada {self.rodada_clusterizacao} (prefixo: '{self.prefixo_cluster}').")
        validar_estado_preparado(self) # Garante que self.X, self.coluna_textos e self._vectorizer existem (após preparar)

        # Validação adicional: Os dados foram realmente preparados nesta rodada?
        if self.X is None or self._indices_preparados_na_rodada is None:
             raise RuntimeError(f"A etapa 'preparar' não foi executada com sucesso ou não encontrou textos para analisar na rodada {self.rodada_clusterizacao}. Execute 'preparar' novamente antes de 'clusterizar'.")

        if self.X.shape[0] == 0:
            logging.warning(f"Nenhum texto foi preparado para agrupar na rodada {self.rodada_clusterizacao}. O agrupamento será pulado.")
            # Não incrementa rodada, retorna o último sucesso
            if self._ultima_coluna_cluster is None:
                 raise RuntimeError("Nenhum agrupamento anterior foi bem-sucedido e o atual não pode ser executado (sem textos preparados).")
            return self._ultima_coluna_cluster

        # 1. Valida K e executa K-Means nos dados preparados (self.X)
        num_textos_preparados = self.X.shape[0]
        validar_parametro_num_clusters(num_clusters, num_textos_preparados)

        logging.info(f"Agrupando {num_textos_preparados} textos em {num_clusters} grupos (K-Means)...")
        # Define parâmetros padrão que podem ser sobrescritos pelos kwargs
        default_kmeans_params = {'n_clusters': num_clusters, 'random_state': self.random_state, 'n_init': 'auto'}
        final_kmeans_kwargs = {**default_kmeans_params, **kmeans_kwargs}
        logging.debug(f"Parâmetros finais para KMeans: {final_kmeans_kwargs}") # Movido para DEBUG
        kmeans = KMeans(**final_kmeans_kwargs)
        try:
            cluster_labels = kmeans.fit_predict(self.X)
            logging.info(f"Agrupamento K-Means concluído. {len(cluster_labels)} textos foram atribuídos a grupos.")
        except Exception as e:
             logging.error(f"Erro durante a execução do algoritmo de agrupamento K-Means: {e}")
             raise

        # 2. Atribui os labels ao DataFrame original usando os índices preparados
        nome_coluna_cluster = f'{self.prefixo_cluster}{self.rodada_clusterizacao}'
        self._atribuir_labels_cluster(cluster_labels, self._indices_preparados_na_rodada, nome_coluna_cluster)

        # 3. Atualiza estado interno
        self._ultimo_num_clusters = num_clusters
        self._ultima_coluna_cluster = nome_coluna_cluster
        # Limpa os dados preparados após o uso para evitar inconsistências se preparar não for chamado de novo
        # self.X = None # Opcional: Forçar chamar preparar de novo? Pode ser confuso. Melhor deixar.
        # self._indices_preparados_na_rodada = None # Opcional: Forçar chamar preparar de novo?
        self.rodada_clusterizacao += 1 # Incrementa a rodada APÓS sucesso
        logging.info(f"Agrupamento da rodada {self.rodada_clusterizacao - 1} (prefixo '{self.prefixo_cluster}') concluído com sucesso. Resultados na coluna '{nome_coluna_cluster}'.")
        return nome_coluna_cluster

    def salvar(self,
               o_que_salvar: str = 'ambos',
               formato_tudo: str = 'csv',
               formato_amostras: str = 'xlsx',
               caminho_tudo: Optional[str] = None,
               caminho_amostras: Optional[str] = None,
               diretorio_saida: Optional[str] = None
               ) -> dict[str, bool | str | None]:
        """
        Salva os resultados do último agrupamento realizado, com opções flexíveis.

        Permite salvar o conjunto de dados completo com os resultados, amostras de cada grupo,
        ou ambos, em diferentes formatos e locais.

        Args:
            o_que_salvar (str, optional): Define o que será salvo:
                                          'tudo': Salva o conjunto de dados completo com a nova coluna de grupo.
                                          'amostras': Salva um arquivo separado com exemplos de textos de cada grupo.
                                          'ambos': Salva ambos os arquivos (padrão).
            formato_tudo (str, optional): Formato para salvar o conjunto completo ('csv', 'xlsx', 'parquet', 'json'). Padrão é 'csv'.
            formato_amostras (str, optional): Formato para salvar as amostras ('xlsx', 'csv', 'json'). Padrão é 'xlsx'.
            caminho_tudo (Optional[str], optional): Caminho completo (incluindo nome do arquivo) para salvar o conjunto completo.
                                                    Se fornecido, ignora `diretorio_saida`. Se a extensão for omitida,
                                                    será adicionada com base em `formato_tudo`. Padrão é None (usa nome padrão).
            caminho_amostras (Optional[str], optional): Caminho completo para salvar as amostras. Se fornecido, ignora
                                                       `diretorio_saida`. Se a extensão for omitida, será adicionada
                                                       com base em `formato_amostras`. Padrão é None (usa nome padrão).
            diretorio_saida (Optional[str], optional): Pasta onde salvar os arquivos caso `caminho_tudo` ou `caminho_amostras`
                                                      não sejam especificados. Se None (padrão), salva na pasta atual.

        Returns:
            dict[str, bool | str | None]: Um dicionário indicando o sucesso e o caminho de cada arquivo salvo.
                                          Ex: `{'tudo_salvo': True, 'caminho_tudo': '/caminho/abs/dados_com_grupos.csv', 'amostras_salvas': True, 'caminho_amostras': '/caminho/abs/amostras_grupos.xlsx'}`

        Raises:
            RuntimeError: Se o método `clusterizar` (agrupamento) não foi executado antes.
            KeyError: Se a coluna de resultados do último agrupamento não for encontrada.
            ValueError: Se alguma opção (`o_que_salvar`, `formato_tudo`, `formato_amostras`) for inválida.
            ImportError: Se uma biblioteca necessária para o formato de arquivo escolhido não estiver instalada
                         (ex: `openpyxl` para `.xlsx`, `pyarrow` para `.parquet`).
            OSError: Se houver um erro ao criar a pasta de saída (ex: permissão negada).
        """
        logging.info(f"Iniciando processo para salvar resultados (Opção: '{o_que_salvar}', Prefixo: '{self.prefixo_cluster}')...")
        resultados_salvamento = {
            'tudo_salvo': False, 'caminho_tudo': None,
            'amostras_salvas': False, 'caminho_amostras': None
        }

        try:
            # --- Validações Iniciais ---
            validar_estado_clusterizado(self)
            rodada_a_salvar = self.rodada_clusterizacao - 1
            nome_coluna_cluster = self._ultima_coluna_cluster # Já contém o prefixo correto
            num_clusters = self._ultimo_num_clusters
            validar_coluna_cluster_existe(self.df, nome_coluna_cluster)
            validar_opcao_salvar(o_que_salvar)

            # --- Determinar Caminhos e Formatos Finais (usando a função de utils) ---
            # A validação de formato (validar_formato_salvar) agora ocorre dentro de determinar_caminhos_saida
            # Passa o prefixo para gerar nomes padrão corretos
            caminhos_formatos = determinar_caminhos_saida(
                o_que_salvar=o_que_salvar,
                formato_tudo=formato_tudo,
                formato_amostras=formato_amostras,
                caminho_tudo=caminho_tudo,
                caminho_amostras=caminho_amostras,
                diretorio_saida=diretorio_saida,
                input_path=self._input_path,
                rodada_a_salvar=rodada_a_salvar,
                prefixo_cluster=self.prefixo_cluster # Passa o prefixo atual
            )

            path_tudo_final = caminhos_formatos['path_tudo_final']
            fmt_tudo_final = caminhos_formatos['fmt_tudo_final']
            path_amostras_final = caminhos_formatos['path_amostras_final']
            fmt_amostras_final = caminhos_formatos['fmt_amostras_final']

            # --- Executar Salvamento ---
            if path_tudo_final and fmt_tudo_final:
                sucesso_tudo = salvar_dataframe(self.df, path_tudo_final, fmt_tudo_final)
                resultados_salvamento['tudo_salvo'] = sucesso_tudo
                # O caminho já vem absoluto de determinar_caminhos_saida
                resultados_salvamento['caminho_tudo'] = path_tudo_final if sucesso_tudo else None

            if path_amostras_final and fmt_amostras_final:
                # salvar_amostras internamente chama salvar_dataframe
                sucesso_amostras = salvar_amostras(self.df, nome_coluna_cluster, num_clusters, path_amostras_final, fmt_amostras_final)
                resultados_salvamento['amostras_salvas'] = sucesso_amostras
                # O caminho já vem absoluto de determinar_caminhos_saida
                resultados_salvamento['caminho_amostras'] = path_amostras_final if sucesso_amostras else None

        except (RuntimeError, KeyError, ValueError, ImportError, OSError) as e:
             # Captura erros de validação (incluindo formato), estado, IO, dependência
             logging.error(f"Falha no processo de salvamento: {e}")
             # Retorna o dicionário com False/None (estado inicial)
             return resultados_salvamento
        except Exception as e:
             # Captura outros erros inesperados
             logging.error(f"Erro inesperado durante o salvamento: {e}")
             return resultados_salvamento

        logging.info(f"Processo de salvamento concluído. Status: {resultados_salvamento}")
        return resultados_salvamento

    def classificar(self, cluster_ids: int | list[int], classificacao: str, rodada: Optional[int] = None) -> None:
        """
        Atribui uma classificação a um ou mais clusters de uma rodada específica.

        Preenche a coluna de classificação (self.nome_coluna_classificacao) do DataFrame
        para todas as linhas pertencentes aos clusters especificados na rodada indicada.
        Se a coluna de classificação não existir, ela será criada. Classificações posteriores
        para as mesmas linhas sobrescreverão as anteriores.

        Args:
            cluster_ids (int | list[int]): O ID do cluster ou uma lista de IDs
                                           de clusters a serem classificados.
            classificacao (str): O rótulo (string) de classificação a ser atribuído.
                                 Não pode ser uma string vazia.
            rodada (Optional[int], optional): O número da rodada de clusterização
                                              cujos clusters serão classificados.
                                              Se None (padrão), usa a última rodada
                                              de clusterização concluída. Padrão é None.

        Raises:
            RuntimeError: Se nenhuma clusterização foi realizada ainda (`clusterizar` não foi chamado).
            TypeError: Se `classificacao` não for string ou `cluster_ids` não for int/list de ints,
                       ou se `rodada` não for int (se fornecido).
            ValueError: Se `classificacao` for vazia, `rodada` for inválida (fora do range),
                        ou se algum `cluster_id` não existir na rodada especificada,
                        ou se a lista `cluster_ids` estiver vazia.
            KeyError: Se a coluna de cluster correspondente à `rodada` não for encontrada (erro interno).
        """
        logging.info(f"Iniciando atribuição da classificação '{classificacao}' para o(s) grupo(s) {cluster_ids}" + (f" da rodada {rodada}." if rodada else " da última rodada."))

        # --- Validações Iniciais ---
        validar_estado_clusterizado(self) # Garante que houve ao menos um agrupamento
        validar_tipo_classificacao(classificacao)

        # --- Determina e Valida a Rodada Alvo ---
        rodada_alvo = rodada if rodada is not None else self.rodada_clusterizacao - 1
        # Passa o prefixo para a validação encontrar a coluna correta
        validar_rodada_valida(rodada_alvo, self.rodada_clusterizacao, self.prefixo_cluster)

        # --- Valida Coluna e IDs de Cluster ---
        # Usa o prefixo para construir o nome da coluna alvo
        coluna_cluster_alvo = f'{self.prefixo_cluster}{rodada_alvo}'
        validar_coluna_existe(self.df, coluna_cluster_alvo) # Garante que a coluna da rodada existe

        # Garante que cluster_ids seja uma lista para validação e uso
        lista_ids = cluster_ids if isinstance(cluster_ids, list) else [cluster_ids]
        # A validação de tipo e não vazio é feita dentro de validar_cluster_ids_presentes
        # Passa o prefixo para a validação
        validar_cluster_ids_presentes(self.df, coluna_cluster_alvo, lista_ids, self.prefixo_cluster)

        # --- Garante a Existência e Tipo Adequado da Coluna de Classificação ---
        self._garantir_coluna_classificacao() # Já usa self.nome_coluna_classificacao

        # --- Aplica a Classificação ---
        linhas_a_classificar = self.df[coluna_cluster_alvo].isin(lista_ids)
        num_linhas = linhas_a_classificar.sum()
        logging.info(f"Aplicando classificação '{classificacao}' a {num_linhas} texto(s) na coluna '{self.nome_coluna_classificacao}' correspondente(s) ao(s) grupo(s) {lista_ids} da coluna '{coluna_cluster_alvo}'.")
        # Usa o nome da coluna de classificação da instância
        self.df.loc[linhas_a_classificar, self.nome_coluna_classificacao] = classificacao

        logging.info("Atribuição de classificação concluída.")

    def resetar(self) -> None:
        """
        Reinicia o estado da instância ClusterFacil.

        Remove todas as colunas de resultados de agrupamentos anteriores (identificadas pelo `prefixo_cluster`)
        e a coluna de classificação manual (`self.nome_coluna_classificacao`), se existirem.
        Redefine o contador de rodadas para 1 e limpa as configurações e resultados
        anteriores (análise TF-IDF, opções de agrupamento, etc.).

        Após chamar `resetar`, você precisará chamar `preparar` novamente antes
        de poder `clusterizar` ou `salvar`.
        """
        logging.warning(f"Iniciando reinicialização do estado do ClusterFacil (Prefixo: '{self.prefixo_cluster}', Coluna Classificação: '{self.nome_coluna_classificacao}')...")

        colunas_para_remover = []
        # Usa o prefixo da instância para encontrar as colunas de cluster
        regex_coluna_cluster = re.compile(rf'^{re.escape(self.prefixo_cluster)}\d+$')

        for col in self.df.columns:
            if regex_coluna_cluster.match(col):
                colunas_para_remover.append(col)

        # Usa o nome da coluna de classificação da instância
        if self.nome_coluna_classificacao in self.df.columns:
             colunas_para_remover.append(self.nome_coluna_classificacao)

        if colunas_para_remover:
            self.df.drop(columns=colunas_para_remover, inplace=True, errors='ignore')
            logging.info(f"Colunas de resultados anteriores removidas: {colunas_para_remover}")
        else:
            logging.info("Nenhuma coluna de resultados de agrupamentos ou classificação encontrada para remover.")

        # Resetar atributos de estado
        self.rodada_clusterizacao = 1
        self.coluna_textos = None
        self.X = None
        self.inercias = None
        self._ultimo_num_clusters = None
        self._ultima_coluna_cluster = None
        self._vectorizer = None
        self._tfidf_kwargs = None # Resetar também as configurações do TF-IDF
        self._indices_preparados_na_rodada = None # Resetar índices preparados

        logging.info("Estado do ClusterFacil reiniciado. Rodada definida para 1. Chame 'preparar' novamente para começar uma nova análise.")

    def subcluster(self, classificacao_desejada: str) -> Self:
        """
        Cria uma nova instância de ClusterFacil contendo apenas os textos de uma classificação específica.

        Útil para realizar um novo agrupamento (subcluster) dentro de um grupo já classificado.

        A nova instância terá os resultados de agrupamentos anteriores removidos e usará 'subcluster_'
        como prefixo para os novos resultados e 'subclassificacao' como nome da coluna para
        classificações manuais dentro deste subcluster. A coluna de classificação original
        é mantida, mas renomeada para '{nome_original}_origem' para referência.

        Args:
            classificacao_desejada (str): A classificação manual atribuída anteriormente que você deseja usar para filtrar os textos.

        Returns:
            ClusterFacil: Uma nova instância de ClusterFacil, pronta para analisar e agrupar os textos do subcluster.

        Raises:
            KeyError: Se a coluna de classificação original não for encontrada.
            ValueError: Se a `classificacao_desejada` não existir na coluna de classificação.
            RuntimeError: Se `preparar` não foi chamado na instância original (necessário para saber qual era a coluna de texto).
        """
        logging.info(f"Criando um novo objeto ClusterFacil para analisar o subcluster da classificação: '{classificacao_desejada}'")

        # Validação: Coluna de classificação existe?
        validar_coluna_existe(self.df, self.nome_coluna_classificacao)

        # Validação: Classificação desejada existe?
        if classificacao_desejada not in self.df[self.nome_coluna_classificacao].dropna().unique(): # Adicionado dropna() para segurança
            msg = f"A classificação '{classificacao_desejada}' não foi encontrada na coluna '{self.nome_coluna_classificacao}'."
            logging.error(msg)
            raise ValueError(msg)

        # Validação: Coluna de texto foi definida?
        if not self.coluna_textos:
             msg = "A etapa 'preparar' deve ser executada na instância original para definir qual coluna contém os textos antes de criar um subcluster."
             logging.error(msg)
             raise RuntimeError(msg)

        # Cria o DataFrame do subcluster usando a função utilitária
        # A função já faz a filtragem, limpeza de colunas e renomeação
        df_sub = criar_df_subcluster(self.df, self.nome_coluna_classificacao, classificacao_desejada)

        # Criar e retornar nova instância
        subcluster_instance = ClusterFacil(
            entrada=df_sub, # Usa o DataFrame já processado
            prefixo_cluster="subcluster_",
            nome_coluna_classificacao="subclassificacao"
        )
        # Propagar a coluna de texto e as configurações do TF-IDF para a nova instância
        # Isso permite chamar preparar/clusterizar diretamente no subcluster sem re-especificar
        subcluster_instance.coluna_textos = self.coluna_textos
        subcluster_instance._tfidf_kwargs = self._tfidf_kwargs.copy() if self._tfidf_kwargs else None
        logging.info("Novo objeto ClusterFacil para o subcluster criado e configurado.")

        return subcluster_instance

    def obter_subcluster_df(self, classificacao_desejada: str) -> pd.DataFrame:
        """
        Retorna um DataFrame contendo apenas os textos de uma classificação específica,
        com as colunas de resultados de agrupamentos anteriores removidas e a coluna de
        classificação original renomeada.

        Args:
            classificacao_desejada (str): A classificação manual que você deseja usar para filtrar os textos.

        Returns:
            pd.DataFrame: O DataFrame filtrado e preparado para análise de subcluster.

        Raises:
            KeyError: Se a coluna de classificação original não for encontrada.
            ValueError: Se a `classificacao_desejada` não existir na coluna de classificação.
        """
        logging.info(f"Extraindo o conjunto de dados do subcluster para a classificação: '{classificacao_desejada}'")
        # Validações de coluna e classificação, filtragem e limpeza são feitas dentro de criar_df_subcluster
        return criar_df_subcluster(self.df, self.nome_coluna_classificacao, classificacao_desejada)

    def listar_classificacoes(self) -> list[str]:
        """
        Retorna uma lista das classificações manuais únicas (não nulas) presentes na coluna de classificação.

        Returns:
            list[str]: Lista das classificações únicas encontradas. Retorna lista vazia se
                       a coluna não existir ou não houver classificações atribuídas.
        """
        if self.nome_coluna_classificacao not in self.df.columns:
            logging.warning(f"Coluna de classificação '{self.nome_coluna_classificacao}' não encontrada. Não há classificações para listar.")
            return []

        classificacoes_unicas = self.df[self.nome_coluna_classificacao].dropna().unique().tolist()
        # Garante que sejam strings (embora _garantir_coluna_classificacao tente fazer isso)
        classificacoes_unicas = [str(c) for c in classificacoes_unicas]
        logging.info(f"Classificações manuais únicas encontradas: {classificacoes_unicas}")
        return classificacoes_unicas

    def contar_classificacoes(self, inclui_na=False) -> pd.Series:
        """
        Loga a contagem de quantos textos pertencem a cada classificação manual atribuída.
        """
        if self.nome_coluna_classificacao not in self.df.columns:
            logging.warning(f"Coluna de classificação '{self.nome_coluna_classificacao}' não encontrada. Não há contagem para retornar.")
            return pd.Series(dtype=int) # Retorna Series vazia com tipo int

        contagem = self.df[self.nome_coluna_classificacao].value_counts(dropna=inclui_na)
        logging.info(f"Contagem de textos por classificação manual na coluna '{self.nome_coluna_classificacao}':\n{contagem}")
        return None
