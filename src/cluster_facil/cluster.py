import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import Optional, List, Union
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
    determinar_caminhos_saida # Importa a nova função
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
                                           Se None (padrão), lê a primeira aba. Padrão é None.

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

        self.rodada_clusterizacao: int = 1
        self.coluna_textos: Optional[str] = None
        self.X: Optional[csr_matrix] = None # Matriz TF-IDF da última operação relevante
        self.inercias: Optional[List[float]] = None
        self._ultimo_num_clusters: Optional[int] = None
        self._ultima_coluna_cluster: Optional[str] = None
        self._vectorizer: Optional[TfidfVectorizer] = None # Guardar o vectorizer para reuso
        self._tfidf_kwargs: Optional[dict] = None # Guardar kwargs do TF-IDF para referência

        # Ajustar rodada_clusterizacao com base nas colunas existentes
        self._ajustar_rodada_inicial()

    def _ajustar_rodada_inicial(self):
        """Verifica colunas existentes e ajusta a rodada de clusterização inicial."""
        max_rodada_existente = 0
        regex_coluna_cluster = re.compile(r'^cluster_(\d+)$')
        for col in self.df.columns:
            match = regex_coluna_cluster.match(col)
            if match:
                rodada_num = int(match.group(1))
                if rodada_num > max_rodada_existente:
                    max_rodada_existente = rodada_num
        
        if max_rodada_existente > 0:
            self.rodada_clusterizacao = max_rodada_existente + 1
            logging.info(f"Colunas de cluster existentes detectadas. Próxima rodada será: {self.rodada_clusterizacao}")
        else:
            logging.info("Nenhuma coluna de cluster pré-existente encontrada. Iniciando na rodada 1.")
            self.rodada_clusterizacao = 1 # Garante que seja 1 se nenhuma for encontrada

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
        """
        df_para_clusterizar = self.df
        indices_originais = self.df.index # Guarda todos os índices por padrão -

        # Lógica de Filtragem para Rodadas > 1
        if self.rodada_clusterizacao > 1 and 'classificacao' in self.df.columns:
            linhas_nao_classificadas = self.df['classificacao'].isna()
            if not linhas_nao_classificadas.all(): # Se houver alguma linha classificada
                logging.info("Identificando linhas não classificadas para a nova rodada.")
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
                logging.info("Coluna 'classificacao' existe, mas todas as linhas estão sem classificação. Usando todos os dados.")
                # Neste caso, self.X já deve ser o da preparação inicial ou da última rodada completa
                # Se self.X for None por algum motivo (ex: rodada anterior vazia), precisa recalcular
                if self.X is None:
                    logging.warning("self.X era None, recalculando TF-IDF para todos os dados.")
                    textos_processados = self.df[self.coluna_textos].fillna('').astype(str).str.lower()
                    self.X = self._vectorizer.fit_transform(textos_processados)

        elif self.rodada_clusterizacao > 1:
             logging.info("Coluna 'classificacao' não encontrada. Usando todos os dados para clusterização.")
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
            cluster_labels (List[int]): Lista de rótulos de cluster retornados pelo K-Means.
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
        Garante que a coluna 'classificacao' exista no DataFrame e tenha um tipo adequado (StringDtype).

        Se a coluna não existir, ela é criada com pd.NA e tipo StringDtype.
        Se existir mas não for string/object, tenta convertê-la para StringDtype.
        """
        if 'classificacao' not in self.df.columns:
            logging.info("Coluna 'classificacao' não encontrada. Criando coluna com tipo StringDtype.")
            # Criar diretamente com tipo que aceita nulos e strings
            self.df['classificacao'] = pd.Series(pd.NA, index=self.df.index, dtype=pd.StringDtype())
        else:
            # Se existe, garante que seja um tipo adequado (string ou object)
            if not pd.api.types.is_string_dtype(self.df['classificacao']) and not pd.api.types.is_object_dtype(self.df['classificacao']):
                 logging.warning(f"Coluna 'classificacao' existe mas não é string/object ({self.df['classificacao'].dtype}). Convertendo para StringDtype.")
                 try:
                     # Tenta converter preservando NAs
                     self.df['classificacao'] = self.df['classificacao'].astype(pd.StringDtype())
                     logging.info("Coluna 'classificacao' convertida com sucesso para StringDtype.")
                 except Exception as e:
                     # Se a conversão falhar (tipos muito mistos), loga erro mas continua.
                     # A atribuição na função classificar ainda pode funcionar dependendo do caso.
                     logging.error(f"Falha ao converter coluna 'classificacao' existente para StringDtype: {e}. A classificação pode não ter o tipo ideal.")
            else:
                logging.debug("Coluna 'classificacao' já existe e possui tipo adequado.")


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
        self.inercias = calcular_e_plotar_cotovelo(self.X, limite_k, n_init, plotar=plotar_cotovelo)

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

        Nota sobre múltiplas rodadas: Se a coluna 'classificacao' existir e contiver
        linhas já classificadas, esta função irá (por padrão) clusterizar apenas as
        linhas *não* classificadas. O TF-IDF será recalculado *apenas* para este
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
            str: O nome da coluna de cluster criada (ex: 'cluster_1').

        Raises:
            RuntimeError: Se `preparar` não foi executado antes ou se não há dados.
            ValueError: Se `num_clusters` for inválido.
            Exception: Outros erros durante a execução do K-Means.
        """
        logging.info(f"Iniciando clusterização com K={num_clusters} para a rodada {self.rodada_clusterizacao}.")
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
        default_kmeans_params = {'n_clusters': num_clusters, 'random_state': 42, 'n_init': 'auto'}
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
        nome_coluna_cluster = f'cluster_{self.rodada_clusterizacao}'
        self._atribuir_labels_cluster(cluster_labels, indices_para_clusterizar, nome_coluna_cluster)

        # 4. Atualiza estado interno
        self._ultimo_num_clusters = num_clusters
        self._ultima_coluna_cluster = nome_coluna_cluster
        self.rodada_clusterizacao += 1 # Incrementa a rodada APÓS sucesso
        logging.info(f"Clusterização da rodada {self.rodada_clusterizacao - 1} concluída com sucesso.")
        return nome_coluna_cluster

    def salvar(self,
               o_que_salvar: str = 'ambos',
               formato_tudo: str = 'csv',
               formato_amostras: str = 'xlsx',
               caminho_tudo: Optional[str] = None,
               caminho_amostras: Optional[str] = None,
               diretorio_saida: Optional[str] = None
               ) -> dict[str, Union[bool, Optional[str]]]:
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
            dict[str, Union[bool, Optional[str]]]: Dicionário com o status (True/False) e caminho absoluto
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
        logging.info(f"Iniciando processo de salvamento (o_que_salvar='{o_que_salvar}')...")
        resultados_salvamento = {
            'tudo_salvo': False, 'caminho_tudo': None,
            'amostras_salvas': False, 'caminho_amostras': None
        }

        try:
            # --- Validações Iniciais ---
            validar_estado_clusterizado(self)
            rodada_a_salvar = self.rodada_clusterizacao - 1
            nome_coluna_cluster = self._ultima_coluna_cluster
            num_clusters = self._ultimo_num_clusters
            validar_coluna_cluster_existe(self.df, nome_coluna_cluster)
            validar_opcao_salvar(o_que_salvar)

            # --- Determinar Caminhos e Formatos Finais (usando a função de utils) ---
            # A validação de formato (validar_formato_salvar) agora ocorre dentro de determinar_caminhos_saida
            caminhos_formatos = determinar_caminhos_saida(
                o_que_salvar=o_que_salvar,
                formato_tudo=formato_tudo,
                formato_amostras=formato_amostras,
                caminho_tudo=caminho_tudo,
                caminho_amostras=caminho_amostras,
                diretorio_saida=diretorio_saida,
                input_path=self._input_path,
                rodada_a_salvar=rodada_a_salvar
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


    def classificar(self, cluster_ids: Union[int, List[int]], classificacao: str, rodada: Optional[int] = None) -> None:
        """
        Atribui uma classificação a um ou mais clusters de uma rodada específica.

        Preenche a coluna 'classificacao' do DataFrame para todas as linhas
        pertencentes aos clusters especificados na rodada indicada. Se a coluna
        'classificacao' não existir, ela será criada. Classificações posteriores
        para as mesmas linhas sobrescreverão as anteriores.

        Args:
            cluster_ids (Union[int, List[int]]): O ID do cluster ou uma lista de IDs
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
        validar_rodada_valida(rodada_alvo, self.rodada_clusterizacao)

        # --- Valida Coluna e IDs de Cluster ---
        coluna_cluster_alvo = f'cluster_{rodada_alvo}'
        validar_coluna_existe(self.df, coluna_cluster_alvo) # Garante que a coluna da rodada existe

        lista_ids = cluster_ids if isinstance(cluster_ids, list) else [cluster_ids]
        # A validação de tipo e não vazio é feita dentro de validar_cluster_ids_presentes
        validar_cluster_ids_presentes(self.df, coluna_cluster_alvo, lista_ids)

        # --- Garante a Existência e Tipo Adequado da Coluna 'classificacao' ---
        self._garantir_coluna_classificacao()

        # --- Aplica a Classificação ---
        linhas_a_classificar = self.df[coluna_cluster_alvo].isin(lista_ids)
        num_linhas = linhas_a_classificar.sum()
        logging.info(f"Aplicando classificação '{classificacao}' a {num_linhas} linha(s) correspondente(s) aos clusters {lista_ids} da rodada {rodada_alvo}.")
        self.df.loc[linhas_a_classificar, 'classificacao'] = classificacao

        logging.info("Classificação concluída.")

    # --- MÉTODO DE CONVENIÊNCIA ---
    def finalizar(self, num_clusters: int, **kwargs_salvar) -> dict[str, Union[bool, Optional[str]]]:
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
            dict[str, Union[bool, Optional[str]]]: O dicionário retornado pelo método `salvar`,
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

        Remove todas as colunas de cluster ('cluster_1', 'cluster_2', etc.) e
        a coluna 'classificacao' (se existir) do DataFrame. Redefine o contador
        de rodadas para 1 e limpa os resultados e configurações de clusterizações
        anteriores (matriz TF-IDF, inércias, vectorizer, coluna de textos).

        Após chamar `resetar`, será necessário chamar `preparar` novamente antes
        de poder `clusterizar` ou `finalizar`.
        """
        logging.warning("Iniciando reset do estado do ClusterFacil...")

        colunas_para_remover = []
        regex_coluna_cluster = re.compile(r'^cluster_\d+$')

        for col in self.df.columns:
            if regex_coluna_cluster.match(col):
                colunas_para_remover.append(col)

        if 'classificacao' in self.df.columns:
             colunas_para_remover.append('classificacao') # Também remove a coluna de classificação

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

        logging.info("Estado do ClusterFacil resetado. Rodada definida para 1. Chame 'preparar' novamente.")
