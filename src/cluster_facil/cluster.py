import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import Optional, Self # Adicionado Self para type hint, removido List, Union
from scipy.sparse import csr_matrix
import logging
import re
import os

from .utils import (
    stop_words_pt,
    calcular_e_plotar_cotovelo,
    salvar_dataframe,
    salvar_amostras,
    carregar_dados,
    determinar_caminhos_saida,
    ajustar_rodada_inicial, # Adicionado
    criar_df_subcluster # Adicionado
)
from .validations import (
    validar_entrada_inicial,
    validar_arquivo_existe,
    validar_dependencia_leitura,
    validar_formato_suportado,
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
    validar_opcao_salvar,
    validar_formato_salvar
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
                 random_state: Optional[int] = 42): # Adicionado random_state
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
            logging.info("ClusterFacil inicializado com DataFrame existente.")
        elif isinstance(entrada, str): # Sabemos que é string por causa da validação
            self.df: pd.DataFrame = carregar_dados(entrada, aba=aba)
            logging.info(f"ClusterFacil inicializado com dados do arquivo: {entrada}" + (f" (aba: {aba})" if aba else ""))
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
        self.random_state = random_state # Armazena o random_state

        # Ajustar rodada_clusterizacao com base nas colunas existentes usando a função de utils
        self.rodada_clusterizacao = ajustar_rodada_inicial(self.df.columns, self.prefixo_cluster)

    # --- Métodos Privados Auxiliares ---
    def _preparar_dados_rodada_atual(self) -> Optional[pd.Index]:
        """
        Prepara os dados para a rodada de clusterização atual.

        Identifica as linhas a serem clusterizadas (todas ou apenas as não classificadas
        em rodadas > 1), recalcula o TF-IDF se necessário (apenas para o subset não
        classificado) e atualiza self.X.

        Returns:
            Optional[pd.Index]: Os índices do DataFrame original correspondentes às
                                linhas que foram selecionadas para esta rodada de
                                clusterização. Retorna None se nenhuma linha for selecionada.
                                Retorna self.df.index se todas as linhas forem usadas.

        Raises:
            RuntimeError: Se ocorrer um erro interno e a matriz TF-IDF (self.X) não estiver disponível.
        """
        df_para_clusterizar = self.df
        indices_originais = self.df.index # Guarda todos os índices por padrão

        # Lógica de Filtragem para Rodadas > 1
        # Usa o nome da coluna de classificação definido na instância
        if self.rodada_clusterizacao > 1 and self.nome_coluna_classificacao in self.df.columns:
            linhas_nao_classificadas = self.df[self.nome_coluna_classificacao].isna()
            if not linhas_nao_classificadas.all(): # Se houver alguma linha classificada
                logging.info(f"Identificando linhas não classificadas na coluna '{self.nome_coluna_classificacao}' para a nova rodada.")
                df_para_clusterizar = self.df.loc[linhas_nao_classificadas].copy()
                indices_originais = df_para_clusterizar.index # Guarda índices do subset
                logging.info(f"Encontradas {len(df_para_clusterizar)} linhas não classificadas.")

                if df_para_clusterizar.empty:
                    logging.warning("Nenhuma linha não classificada encontrada. Clusterização desta rodada será pulada.")
                    self.X = None # Garante que X esteja vazio se não houver dados
                    return None # Indica que não há o que clusterizar

                # Recalcular TF-IDF apenas no subset
                logging.info("Recalculando TF-IDF para as linhas não classificadas...")
                # Garante que coluna_textos não seja None (validado em preparar)
                textos_subset = df_para_clusterizar[self.coluna_textos].fillna('').astype(str).str.lower()
                # Reutiliza o vectorizer configurado, mas ajusta ao novo vocabulário
                # Garante que _vectorizer não seja None (validado em preparar)
                self.X = self._vectorizer.fit_transform(textos_subset)
                logging.info(f"Nova matriz TF-IDF calculada com shape: {self.X.shape}")
                return indices_originais # Retorna os índices do subset
            else:
                logging.info(f"Coluna '{self.nome_coluna_classificacao}' existe, mas todas as linhas estão sem classificação. Usando todos os dados.")
                # Neste caso, self.X já deve ser o da preparação inicial ou da última rodada completa
                # Se self.X for None por algum motivo (ex: rodada anterior vazia), precisa recalcular
                if self.X is None:
                    logging.warning("self.X era None, recalculando TF-IDF para todos os dados.")
                    textos_processados = self.df[self.coluna_textos].fillna('').astype(str).str.lower()
                    self.X = self._vectorizer.fit_transform(textos_processados)

        elif self.rodada_clusterizacao > 1:
             logging.info(f"Coluna '{self.nome_coluna_classificacao}' não encontrada. Usando todos os dados para clusterização.")
             # self.X já deve ser o da preparação inicial ou da última rodada completa
             # Se self.X for None por algum motivo, precisa recalcular
             if self.X is None:
                 logging.warning("self.X era None, recalculando TF-IDF para todos os dados.")
                 textos_processados = self.df[self.coluna_textos].fillna('').astype(str).str.lower()
                 self.X = self._vectorizer.fit_transform(textos_processados)

        # Se chegou aqui, significa que todas as linhas são usadas ou é a primeira rodada
        # Garante que self.X não seja None (deve ter sido calculado em preparar ou recalculado acima)
        if self.X is None:
             # Isso não deveria acontecer se preparar() foi chamado, mas como segurança:
             logging.error("Estado inesperado: self.X é None mesmo após a lógica de preparação da rodada.")
             raise RuntimeError("Erro interno: Matriz TF-IDF (self.X) não está disponível.")

        return indices_originais # Retorna todos os índices

    def _atribuir_labels_cluster(self, cluster_labels: List[int], indices_alvo: pd.Index, nome_coluna_cluster: str) -> None:
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
            logging.error(f"Falha ao converter a coluna '{nome_coluna_cluster}' para Int64Dtype: {e}. A coluna pode não ter o tipo ideal.")

        logging.info(f"Coluna '{nome_coluna_cluster}' adicionada/atualizada no DataFrame.")


    def _garantir_coluna_classificacao(self) -> None:
        """
        Garante que a coluna de classificação (definida em self.nome_coluna_classificacao)
        exista no DataFrame e tenha um tipo adequado (StringDtype).

        Se a coluna não existir, ela é criada com pd.NA e tipo StringDtype.
        Se existir mas não for string/object, tenta convertê-la para StringDtype.
        """
        col_classif = self.nome_coluna_classificacao # Usa o nome da instância
        if col_classif not in self.df.columns:
            logging.info(f"Coluna '{col_classif}' não encontrada. Criando coluna com tipo StringDtype.")
            # Criar diretamente com tipo que aceita nulos e strings
            self.df[col_classif] = pd.Series(pd.NA, index=self.df.index, dtype=pd.StringDtype())
        else:
            # Se existe, garante que seja um tipo adequado (string ou object)
            if not pd.api.types.is_string_dtype(self.df[col_classif]) and not pd.api.types.is_object_dtype(self.df[col_classif]):
                 logging.warning(f"Coluna '{col_classif}' existe mas não é string/object ({self.df[col_classif].dtype}). Convertendo para StringDtype.")
                 try:
                     # Tenta converter preservando NAs
                     self.df[col_classif] = self.df[col_classif].astype(pd.StringDtype())
                     logging.info(f"Coluna '{col_classif}' convertida com sucesso para StringDtype.")
                 except Exception as e:
                     # Se a conversão falhar (tipos muito mistos), loga erro mas continua.
                     # A atribuição na função classificar ainda pode funcionar dependendo do caso.
                     logging.error(f"Falha ao converter coluna '{col_classif}' existente para StringDtype: {e}. A classificação pode não ter o tipo ideal.")
            else:
                logging.debug(f"Coluna '{col_classif}' já existe e possui tipo adequado.")


    # --- Métodos Públicos ---
    def preparar(self, coluna_textos: str, limite_k: int = 10, n_init: int = 1, plotar_cotovelo: bool = True, **tfidf_kwargs) -> None:
        """
        Prepara os dados para clusterização.

        Realiza o pré-processamento dos textos (lowercase, preenchimento de nulos),
        calcula a matriz TF-IDF e calcula as inércias para o método do cotovelo.
        Opcionalmente, exibe o gráfico do cotovelo (usando `plt.show()`) para ajudar
        na escolha do número ideal de clusters (K).

        Nota: A exibição automática do gráfico (`plotar_cotovelo=True`) pressupõe um
        ambiente interativo (ex: Jupyter Notebook). Em scripts, considere usar `plotar_cotovelo=False`.

        Nota 2: O cálculo da inércia para o gráfico do cotovelo usa `n_init=1` por padrão
        para agilidade. Isso pode resultar em um gráfico menos estável que a clusterização
        final, que usa `n_init='auto'`. Veja o Roadmap para futuras melhorias.

        Args:
            coluna_textos (str): O nome da coluna no DataFrame que contém os textos.
            limite_k (int, optional): O número máximo de clusters (K) a ser testado
                                       no método do cotovelo. Padrão é 10.
            n_init (int, optional): O número de inicializações do K-Means ao calcular
                                    as inércias para o gráfico do cotovelo. Padrão é 1.
            plotar_cotovelo (bool, optional): Se True (padrão), exibe o gráfico do método
                                              do cotovelo. Padrão é True.
            **tfidf_kwargs: Argumentos de palavra-chave adicionais a serem passados
                            diretamente para o construtor `TfidfVectorizer`.
                            Permite configurar parâmetros como `min_df`, `max_df`, `ngram_range`, etc.
                            Ex: `preparar(..., min_df=5, ngram_range=(1, 2))`

        Raises:
            KeyError: Se o nome da coluna `coluna_textos` não existir no DataFrame.
            ValueError: Se `limite_k` não for um inteiro positivo.
            TypeError: Se a coluna especificada não contiver dados textuais (ou que possam ser convertidos para string).
            ImportError: Se a biblioteca 'matplotlib' for necessária para plotar o gráfico do cotovelo (`plotar_cotovelo=True`) e não estiver instalada.
        """
        logging.info(f"Iniciando preparação com a coluna '{coluna_textos}' e limite K={limite_k}.")

        validar_coluna_existe(self.df, coluna_textos)
        validar_inteiro_positivo('limite_k', limite_k)
        validar_tipo_coluna_texto(self.df, coluna_textos)

        self.coluna_textos = coluna_textos

        textos_processados = self.df[self.coluna_textos].fillna('').astype(str).str.lower()

        logging.info("Calculando TF-IDF inicial...")
        self._tfidf_kwargs = tfidf_kwargs # Armazena os kwargs passados
        # Define parâmetros padrão que podem ser sobrescritos pelos kwargs
        default_tfidf_params = {'stop_words': stop_words_pt}
        final_tfidf_kwargs = {**default_tfidf_params, **self._tfidf_kwargs} # kwargs do usuário têm precedência
        logging.info(f"Parâmetros finais para TfidfVectorizer: {final_tfidf_kwargs}")
        self._vectorizer = TfidfVectorizer(**final_tfidf_kwargs)
        self.X = self._vectorizer.fit_transform(textos_processados)
        logging.info(f"Matriz TF-IDF inicial calculada com shape: {self.X.shape}")

        # TODO (Roadmap): Avaliar uso de n_init > 1 para o gráfico do cotovelo.
        # TODO (Roadmap): Permitir passar kwargs para o KMeans do cotovelo.
        # Passa self.random_state para a função do cotovelo
        self.inercias = calcular_e_plotar_cotovelo(self.X, limite_k, n_init, plotar=plotar_cotovelo, random_state=self.random_state)

        if self.inercias is not None:
             if plotar_cotovelo:
                 logging.info("Preparação concluída. Analise o gráfico do cotovelo exibido para escolher o número de clusters.")
             else:
                 logging.info("Preparação concluída. Inércias calculadas, mas gráfico não exibido (plotar_cotovelo=False).")
        else:
                 logging.info("Preparação concluída (sem dados para o método do cotovelo).")

    def clusterizar(self, num_clusters: int, **kmeans_kwargs) -> str:
        """
        Executa a clusterização K-Means e adiciona a coluna de clusters ao DataFrame.

        Nota sobre múltiplas rodadas: Se a coluna de classificação (self.nome_coluna_classificacao)
        existir e contiver linhas já classificadas, esta função irá (por padrão) clusterizar
        apenas as linhas *não* classificadas. O TF-IDF será recalculado *apenas* para este
        subset, o que significa que o espaço vetorial (vocabulário, pesos IDF) pode
        mudar entre as rodadas. Este comportamento é intencional para focar a nova
        clusterização nos dados remanescentes.

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
            str: O nome da coluna de cluster criada (ex: 'cluster_1', 'subcluster_1').

        Raises:
            RuntimeError: Se `preparar` não foi executado antes ou se não há dados.
            ValueError: Se `num_clusters` for inválido.
            Exception: Outros erros durante a execução do K-Means.
        """
        logging.info(f"Iniciando clusterização com K={num_clusters} para a rodada {self.rodada_clusterizacao} (prefixo: '{self.prefixo_cluster}').")
        validar_estado_preparado(self) # Garante que self.X, self.coluna_textos e self._vectorizer existem

        # 1. Prepara dados (filtra, recalcula X se necessário) e obtém índices alvo
        indices_para_clusterizar = self._preparar_dados_rodada_atual()

        # Se _preparar_dados_rodada_atual retornou None ou self.X ficou None/vazio, pula a rodada
        if indices_para_clusterizar is None or self.X is None or self.X.shape[0] == 0:
            logging.warning(f"Nenhuma amostra válida para clusterizar na rodada {self.rodada_clusterizacao}. Pulando.")
            # Não incrementa rodada, retorna o último sucesso
            if self._ultima_coluna_cluster is None:
                 raise RuntimeError("Nenhuma clusterização anterior bem-sucedida e a atual não pôde ser executada.")
            return self._ultima_coluna_cluster

        # 2. Valida K e executa K-Means
        num_amostras_atual = self.X.shape[0]
        validar_parametro_num_clusters(num_clusters, num_amostras_atual)

        logging.info(f"Executando K-Means com {num_clusters} clusters em {num_amostras_atual} amostras...")
        # Define parâmetros padrão que podem ser sobrescritos pelos kwargs
        # Usa self.random_state como padrão
        default_kmeans_params = {'n_clusters': num_clusters, 'random_state': self.random_state, 'n_init': 'auto'}
        final_kmeans_kwargs = {**default_kmeans_params, **kmeans_kwargs} # kwargs do usuário têm precedência
        logging.info(f"Parâmetros finais para KMeans: {final_kmeans_kwargs}")
        kmeans = KMeans(**final_kmeans_kwargs)
        try:
            cluster_labels = kmeans.fit_predict(self.X)
            logging.info(f"K-Means concluído. {len(cluster_labels)} labels gerados.")
        except Exception as e:
             logging.error(f"Erro durante a execução do K-Means: {e}")
             raise

        # 3. Atribui os labels ao DataFrame original usando o método auxiliar
        # Usa o prefixo da instância para nomear a coluna
        nome_coluna_cluster = f'{self.prefixo_cluster}{self.rodada_clusterizacao}'
        self._atribuir_labels_cluster(cluster_labels, indices_para_clusterizar, nome_coluna_cluster)

        # 4. Atualiza estado interno
        self._ultimo_num_clusters = num_clusters
        self._ultima_coluna_cluster = nome_coluna_cluster
        self.rodada_clusterizacao += 1 # Incrementa a rodada APÓS sucesso
        logging.info(f"Clusterização da rodada {self.rodada_clusterizacao - 1} (prefixo '{self.prefixo_cluster}') concluída com sucesso.")
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
        Salva os resultados da última clusterização realizada, com opções flexíveis.

        Permite salvar o DataFrame completo, as amostras por cluster, ou ambos,
        em diferentes formatos e com nomes/caminhos de arquivo personalizados ou padrão.

        Args:
            o_que_salvar (str, optional): O que salvar: 'tudo' (DataFrame completo),
                                          'amostras', ou 'ambos'. Padrão é 'ambos'.
            formato_tudo (str, optional): Formato para o DataFrame completo: 'csv',
                                          'xlsx', 'parquet', 'json'. Padrão é 'csv'.
            formato_amostras (str, optional): Formato para as amostras: 'xlsx',
                                             'csv', 'json'. Padrão é 'xlsx'.
            caminho_tudo (Optional[str], optional): Caminho completo (incluindo nome e, opcionalmente,
                                                    extensão) para salvar o DataFrame completo.
                                                    Se fornecido, ignora `diretorio_saida`. Se a extensão
                                                    for omitida, será adicionada com base em `formato_tudo`.
                                                    Padrão é None (usa nome padrão).
            caminho_amostras (Optional[str], optional): Caminho completo para salvar as
                                                       amostras. Se fornecido, ignora
                                                       `diretorio_saida`. Se a extensão for omitida,
                                                       será adicionada com base em `formato_amostras`.
                                                       Padrão é None (usa nome padrão).
            diretorio_saida (Optional[str], optional): Pasta onde salvar os arquivos com
                                                      nomes padrão. Se None (padrão), salva no
                                                      diretório atual. Usado apenas se
                                                      `caminho_tudo` ou `caminho_amostras`
                                                      não forem fornecidos. Padrão é None.

        Returns:
            dict[str, bool | str | None]: Dicionário com o status (True/False) e caminho absoluto
                                          de cada tipo de arquivo que foi salvo (ou tentou ser salvo).
                                          Ex: `{'tudo_salvo': True, 'caminho_tudo': '/caminho/abs/df.csv', 'amostras_salvas': False, 'caminho_amostras': None}`

        Raises:
            RuntimeError: Se nenhuma clusterização foi realizada ainda (`clusterizar` não foi chamado).
            KeyError: Se a coluna de cluster da última rodada não for encontrada no DataFrame.
            ValueError: Se alguma opção (`o_que_salvar`, `formato_tudo`, `formato_amostras`) for inválida,
                        ou se a extensão em `caminho_tudo`/`caminho_amostras` (se fornecida) for
                        incompatível com os formatos permitidos.
            ImportError: Se uma dependência necessária para o formato de arquivo escolhido
                         (ex: `openpyxl` para `.xlsx`, `pyarrow` para `.parquet`) não estiver instalada.
            OSError: Se houver um erro ao tentar criar o `diretorio_saida` (ex: permissão negada).
        """
        logging.info(f"Iniciando processo de salvamento (o_que_salvar='{o_que_salvar}', prefixo='{self.prefixo_cluster}')...")
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
        logging.info(f"Iniciando classificação '{classificacao}' para cluster(s) {cluster_ids}" + (f" da rodada {rodada}." if rodada else " da última rodada."))

        # --- Validações Iniciais ---
        validar_estado_clusterizado(self) # Garante que houve ao menos uma clusterização
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
        logging.info(f"Aplicando classificação '{classificacao}' a {num_linhas} linha(s) na coluna '{self.nome_coluna_classificacao}' correspondente(s) aos clusters {lista_ids} da coluna '{coluna_cluster_alvo}'.")
        # Usa o nome da coluna de classificação da instância
        self.df.loc[linhas_a_classificar, self.nome_coluna_classificacao] = classificacao

        logging.info("Classificação concluída.")

    # --- MÉTODO DE CONVENIÊNCIA ---
    def finalizar(self, num_clusters: int, **kwargs_salvar) -> dict[str, bool | str | None]:
        """
        Método de conveniência que executa a clusterização e depois salva os resultados.

        Equivalente a chamar `clusterizar()` seguido por `salvar()`.
        Aceita os mesmos argumentos de palavra-chave que `salvar()` para controlar o processo.

        Args:
            num_clusters (int): O número de clusters (K) a ser usado na clusterização.
            **kwargs_salvar: Argumentos de palavra-chave a serem passados diretamente
                             para o método `salvar()`. Consulte a documentação de `salvar()`
                             para ver as opções disponíveis (ex: `o_que_salvar`, `formato_tudo`,
                             `caminho_tudo`, `diretorio_saida`, etc.).

        Returns:
            dict[str, bool | str | None]: O dicionário retornado pelo método `salvar`,
                                          indicando o status e os caminhos dos arquivos.

        Raises:
            RuntimeError: Se `preparar` não foi executado antes, se não há dados, ou se
                          nenhuma clusterização foi realizada antes de tentar salvar.
            ValueError: Se `num_clusters` for inválido ou se algum argumento em `kwargs_salvar`
                        for inválido (conforme validações em `salvar`).
            ImportError: Se uma dependência necessária para o formato de salvamento não estiver
                         instalada (conforme validações em `salvar`).
            OSError: Se houver erro ao criar o diretório de saída durante o salvamento
                     (conforme validações em `salvar`).
            Exception: Outros erros durante a execução do K-Means (vindos de `clusterizar`).
        """
        logging.info(f"Executando 'finaliza' (clusterizar e salvar) com K={num_clusters}...")
        logging.debug(f"Argumentos para salvar: {kwargs_salvar}")

        # 1. Clusterizar (pode levantar exceção)
        self.clusterizar(num_clusters) # Não passa kwargs de salvar aqui

        # 2. Salvar (pode levantar exceção de validação, IO, dependência)
        # Passa os kwargs recebidos diretamente para o método salvar
        status_salvamento = self.salvar(**kwargs_salvar)

        logging.info(f"'Finaliza' concluído para a rodada {self.rodada_clusterizacao - 1}. Status de salvamento: {status_salvamento}")
        return status_salvamento

    def resetar(self) -> None:
        """
        Reseta o estado da instância ClusterFacil.

        Remove todas as colunas de cluster (usando self.prefixo_cluster) e
        a coluna de classificação (self.nome_coluna_classificacao), se existirem.
        Redefine o contador de rodadas para 1 e limpa os resultados e configurações
        de clusterizações anteriores (matriz TF-IDF, inércias, vectorizer, coluna de textos).

        Após chamar `resetar`, será necessário chamar `preparar` novamente antes
        de poder `clusterizar` ou `finalizar`.
        """
        logging.warning(f"Iniciando reset do estado do ClusterFacil (prefixo: '{self.prefixo_cluster}', classificacao: '{self.nome_coluna_classificacao}')...")

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
            logging.info(f"Colunas removidas durante o reset: {colunas_para_remover}")
        else:
            logging.info("Nenhuma coluna de cluster ou classificação encontrada para remover.")

        # Resetar atributos de estado
        self.rodada_clusterizacao = 1
        self.coluna_textos = None
        self.X = None
        self.inercias = None
        self._ultimo_num_clusters = None
        self._ultima_coluna_cluster = None
        self._vectorizer = None
        self._tfidf_kwargs = None # Resetar também os kwargs do TF-IDF

        logging.info("Estado do ClusterFacil resetado. Rodada definida para 1. Chame 'preparar' novamente.")

    # --- Novos Métodos ---

    def subcluster(self, classificacao_desejada: str) -> Self:
        """
        Cria uma nova instância de ClusterFacil contendo apenas os dados de uma classificação específica.

        A nova instância terá as colunas de cluster resetadas e usará 'subcluster_' como prefixo
        e 'subclassificacao' como nome da coluna de classificação. A coluna de classificação
        original é mantida e renomeada para '{nome_original}_origem'.

        Args:
            classificacao_desejada (str): A classificação a ser usada para filtrar os dados.

        Returns:
            ClusterFacil: Uma nova instância de ClusterFacil configurada para o subcluster.

        Raises:
            KeyError: Se a coluna de classificação (self.nome_coluna_classificacao) não existir (via `criar_df_subcluster`).
            ValueError: Se a `classificacao_desejada` não for encontrada na coluna de classificação (via `criar_df_subcluster`).
            RuntimeError: Se `preparar` não foi chamado na instância original (necessário para self.coluna_textos).
        """
        logging.info(f"Criando subcluster para a classificação: '{classificacao_desejada}'")

        # Validação: Coluna de classificação existe?
        validar_coluna_existe(self.df, self.nome_coluna_classificacao)

        # Validação: Classificação desejada existe?
        if classificacao_desejada not in self.df[self.nome_coluna_classificacao].unique():
            msg = f"A classificação '{classificacao_desejada}' não foi encontrada na coluna '{self.nome_coluna_classificacao}'."
            logging.error(msg)
            raise ValueError(msg)

        # Validação: Coluna de texto foi definida?
        if not self.coluna_textos:
             msg = "O método 'preparar' deve ser executado na instância original para definir 'coluna_textos' antes de criar um subcluster."
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
        # Propagar a coluna de texto e os kwargs do TF-IDF para a nova instância
        # Isso permite chamar preparar/clusterizar diretamente no subcluster sem re-especificar
        subcluster_instance.coluna_textos = self.coluna_textos
        subcluster_instance._tfidf_kwargs = self._tfidf_kwargs.copy() if self._tfidf_kwargs else None
        logging.info("Nova instância ClusterFacil para o subcluster criada.")

        return subcluster_instance

    def obter_subcluster_df(self, classificacao_desejada: str) -> pd.DataFrame:
        """
        Retorna um DataFrame contendo apenas os dados de uma classificação específica,
        com as colunas de cluster removidas e a coluna de classificação original renomeada.

        Args:
            classificacao_desejada (str): A classificação a ser usada para filtrar os dados.

        Returns:
            pd.DataFrame: O DataFrame filtrado e limpo.

        Raises:
            KeyError: Se a coluna de classificação (self.nome_coluna_classificacao) não existir (via `criar_df_subcluster`).
            ValueError: Se a `classificacao_desejada` não for encontrada na coluna de classificação (via `criar_df_subcluster`).
        """
        logging.info(f"Obtendo DataFrame do subcluster para a classificação: '{classificacao_desejada}'")
        # Validações de coluna e classificação, filtragem e limpeza são feitas dentro de criar_df_subcluster
        return criar_df_subcluster(self.df, self.nome_coluna_classificacao, classificacao_desejada)

    def listar_classificacoes(self) -> list[str]:
        """
        Retorna uma lista das classificações únicas (não nulas) presentes na coluna de classificação.

        Returns:
            list[str]: Lista de strings das classificações únicas. Retorna lista vazia se
                       a coluna não existir ou não houver classificações.
        """
        if self.nome_coluna_classificacao not in self.df.columns:
            logging.warning(f"Coluna de classificação '{self.nome_coluna_classificacao}' não encontrada. Retornando lista vazia.")
            return []
        
        classificacoes_unicas = self.df[self.nome_coluna_classificacao].dropna().unique().tolist()
        # Garante que sejam strings (embora _garantir_coluna_classificacao tente fazer isso)
        classificacoes_unicas = [str(c) for c in classificacoes_unicas]
        logging.info(f"Classificações únicas encontradas: {classificacoes_unicas}")
        return classificacoes_unicas

    def contar_classificacoes(self) -> pd.Series:
        """
        Retorna a contagem de cada classificação presente na coluna de classificação.

        Returns:
            pd.Series: Uma Series do Pandas com as classificações como índice e suas contagens
                       como valores. Retorna uma Series vazia se a coluna não existir.
        """
        if self.nome_coluna_classificacao not in self.df.columns:
            logging.warning(f"Coluna de classificação '{self.nome_coluna_classificacao}' não encontrada. Retornando Series vazia.")
            return pd.Series(dtype=int) # Retorna Series vazia com tipo int

        contagem = self.df[self.nome_coluna_classificacao].value_counts(dropna=True) # dropna=True é o padrão, mas explícito
        logging.info(f"Contagem de classificações na coluna '{self.nome_coluna_classificacao}':\n{contagem}")
        return contagem
