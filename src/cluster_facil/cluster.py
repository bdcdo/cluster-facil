import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import Optional, List, Union
from scipy.sparse import csr_matrix
import logging

from .utils import (
    stop_words_pt,
    calcular_e_plotar_cotovelo,
    salvar_dataframe_csv,
    salvar_amostras_excel,
    preparar_caminhos_saida,
    carregar_dados
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
    validar_tipo_classificacao     
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
                                           Se None (padrão), lê a primeira aba. Default é None.

        Raises:
            TypeError: Se a entrada não for um DataFrame ou uma string.
            FileNotFoundError: Se a entrada for uma string e o arquivo não for encontrado.
            ImportError: Se uma dependência necessária (ex: openpyxl, pyarrow) não estiver instalada.
            ValueError: Se o formato do arquivo não for suportado ou houver erro na leitura.
        """
        validar_entrada_inicial(entrada)

        if isinstance(entrada, pd.DataFrame):
            self.df: pd.DataFrame = entrada.copy() # Copiar para evitar modificar o original inesperadamente
            logging.info("ClusterFacil inicializado com DataFrame existente.")
        elif isinstance(entrada, str): # Sabemos que é string por causa da validação
            self.df: pd.DataFrame = carregar_dados(entrada, aba=aba)
            logging.info(f"ClusterFacil inicializado com dados do arquivo: {entrada}" + (f" (aba: {aba})" if aba else ""))

        self.rodada_clusterizacao: int = 1
        self.coluna_textos: Optional[str] = None
        self.X: Optional[csr_matrix] = None # Matriz TF-IDF da última operação relevante
        self.inercias: Optional[List[float]] = None
        self._ultimo_num_clusters: Optional[int] = None
        self._ultima_coluna_cluster: Optional[str] = None
        self._vectorizer: Optional[TfidfVectorizer] = None # Guardar o vectorizer para reuso

    def preparar(self, coluna_textos: str, limite_k: int = 10, n_init: int = 1, plotar_cotovelo: bool = True) -> None:
        """
        Prepara os dados para clusterização.

        Realiza o pré-processamento dos textos (lowercase, preenchimento de nulos),
        calcula a matriz TF-IDF e calcula as inércias para o método do cotovelo.
        Opcionalmente, exibe o gráfico do cotovelo (usando `plt.show()`) para ajudar
        na escolha do número ideal de clusters (K).

        Nota: A exibição automática do gráfico (`plotar_cotovelo=True`) pressupõe um
        ambiente interativo (ex: Jupyter Notebook). Em scripts, considere usar `plotar_cotovelo=False`.

        Nota: O cálculo da inércia para o gráfico do cotovelo usa `n_init=1` por padrão
        para agilidade. Isso pode resultar em um gráfico menos estável que a clusterização
        final, que usa `n_init='auto'`. Veja o Roadmap para futuras melhorias.

        Args:
            coluna_textos (str): O nome da coluna no DataFrame que contém os textos.
            limite_k (int, optional): O número máximo de clusters (K) a ser testado
                                       no método do cotovelo. Default é 10.
            n_init (int, optional): O número de inicializações do K-Means ao calcular
                                    as inércias para o gráfico do cotovelo. Default é 1.
            plotar_cotovelo (bool, optional): Se True (padrão), exibe o gráfico do método
                                              do cotovelo. Default é True.

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
        # TODO (Roadmap): Permitir configuração de parâmetros do TfidfVectorizer.
        self._vectorizer = TfidfVectorizer(stop_words=stop_words_pt)
        self.X = self._vectorizer.fit_transform(textos_processados)
        logging.info(f"Matriz TF-IDF inicial calculada com shape: {self.X.shape}")

        # TODO (Roadmap): Avaliar uso de n_init > 1 para o gráfico do cotovelo.
        self.inercias = calcular_e_plotar_cotovelo(self.X, limite_k, n_init, plotar=plotar_cotovelo)

        if self.inercias is not None:
             if plotar_cotovelo:
                 logging.info("Preparação concluída. Analise o gráfico do cotovelo exibido para escolher o número de clusters.")
             else:
                 logging.info("Preparação concluída. Inércias calculadas, mas gráfico não exibido (plotar_cotovelo=False).")
        else:
             logging.info("Preparação concluída (sem dados para o método do cotovelo).")

    def clusterizar(self, num_clusters: int) -> str:
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

        Returns:
            str: O nome da coluna de cluster criada (ex: 'cluster_1').

        Raises:
            RuntimeError: Se `preparar` não foi executado antes ou se não há dados.
            ValueError: Se `num_clusters` for inválido.
            Exception: Outros erros durante a execução do K-Means.
        """
        logging.info(f"Iniciando clusterização com K={num_clusters} para a rodada {self.rodada_clusterizacao}.")
        validar_estado_preparado(self) # Garante que self.X e self.coluna_textos existem

        df_para_clusterizar = self.df
        indices_originais = self.df.index # Guarda todos os índices por padrão

        # --- Lógica de Filtragem para Rodadas > 1 ---
        if self.rodada_clusterizacao > 1 and 'classificacao' in self.df.columns:
            linhas_nao_classificadas = self.df['classificacao'].isna()
            if not linhas_nao_classificadas.all(): # Se houver alguma linha classificada
                logging.info("Identificando linhas não classificadas para a nova rodada.")
                df_para_clusterizar = self.df.loc[linhas_nao_classificadas].copy()
                indices_originais = df_para_clusterizar.index # Guarda índices do subset
                logging.info(f"Encontradas {len(df_para_clusterizar)} linhas não classificadas.")

                if df_para_clusterizar.empty:
                    logging.warning("Nenhuma linha não classificada encontrada. Clusterização desta rodada será pulada.")
                    # Não incrementa rodada, não faz nada
                    # Retorna o nome da coluna da *última* clusterização bem-sucedida
                    return self._ultima_coluna_cluster

                # Recalcular TF-IDF apenas no subset
                logging.info("Recalculando TF-IDF para as linhas não classificadas...")
                textos_subset = df_para_clusterizar[self.coluna_textos].fillna('').astype(str).str.lower()
                # Reutiliza o vectorizer configurado, mas ajusta ao novo vocabulário
                self.X = self._vectorizer.fit_transform(textos_subset)
                logging.info(f"Nova matriz TF-IDF calculada com shape: {self.X.shape}")
            else:
                logging.info("Coluna 'classificacao' existe, mas todas as linhas estão sem classificação. Usando todos os dados.")
                # Neste caso, self.X já deve ser o da preparação inicial ou da última rodada completa
        elif self.rodada_clusterizacao > 1:
             logging.info("Coluna 'classificacao' não encontrada. Usando todos os dados para clusterização.")
             # self.X já deve ser o da preparação inicial ou da última rodada completa

        # --- Validação e Execução do K-Means ---
        num_amostras_atual = self.X.shape[0]
        validar_parametro_num_clusters(num_clusters, num_amostras_atual)

        logging.info(f"Executando K-Means com {num_clusters} clusters em {num_amostras_atual} amostras...")
        # TODO (Roadmap): Permitir configuração de parâmetros do KMeans.
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        try:
            cluster_labels = kmeans.fit_predict(self.X)
        except Exception as e:
             logging.error(f"Erro durante a execução do K-Means: {e}")
             raise

        # --- Atribuição dos Clusters ao DataFrame Original ---
        nome_coluna_cluster = f'cluster_{self.rodada_clusterizacao}'
        # Inicializa a coluna com NA para garantir que linhas não clusterizadas (se houver) fiquem NA
        self.df[nome_coluna_cluster] = pd.NA
        # Atribui os labels apenas às linhas que foram clusterizadas, usando os índices originais
        self.df.loc[indices_originais, nome_coluna_cluster] = cluster_labels
        # Converte para inteiro nullable para consistência
        self.df[nome_coluna_cluster] = self.df[nome_coluna_cluster].astype(pd.Int64Dtype())

        logging.info(f"Coluna '{nome_coluna_cluster}' adicionada/atualizada no DataFrame.")

        self._ultimo_num_clusters = num_clusters
        self._ultima_coluna_cluster = nome_coluna_cluster

        self.rodada_clusterizacao += 1 # Incrementa a rodada APÓS sucesso
        logging.info(f"Clusterização da rodada {self.rodada_clusterizacao - 1} concluída.")
        return nome_coluna_cluster

    def salvar(self, prefixo_saida: str = '', diretorio_saida: Optional[str] = None) -> dict[str, bool]:
        """
        Salva os resultados da última clusterização realizada.

        Gera dois arquivos:
        1. Um arquivo CSV contendo o DataFrame completo com a(s) coluna(s) de cluster.
        2. Um arquivo Excel contendo até 10 amostras aleatórias de cada cluster gerado
           na última rodada, facilitando a inspeção rápida dos grupos.

        Args:
            prefixo_saida (str, optional): Prefixo para os nomes dos arquivos de saída. Default ''.
            diretorio_saida (Optional[str], optional): Caminho da pasta onde salvar os arquivos.
                                                      Se None (padrão), salva no diretório atual. Default None.

        Returns:
            dict[str, bool]: Um dicionário indicando o sucesso do salvamento de cada arquivo.
                             Ex: {'csv_salvo': True, 'excel_salvo': False}
        """
        logging.info("Tentando salvar resultados da última clusterização...")

        try:
            validar_estado_clusterizado(self)
            rodada_a_salvar = self.rodada_clusterizacao - 1
            nome_coluna_cluster = self._ultima_coluna_cluster
            num_clusters = self._ultimo_num_clusters
            validar_coluna_cluster_existe(self.df, nome_coluna_cluster) # Verificação de segurança
        except (RuntimeError, KeyError) as e:
             # Se a validação falhar (RuntimeError de estado, KeyError de coluna), retorna falha
             logging.error(f"Não é possível salvar: {e}")
             return {'csv_salvo': False, 'excel_salvo': False}

        try:
            caminhos = preparar_caminhos_saida(diretorio_saida, prefixo_saida, rodada_a_salvar)
            nome_csv = caminhos['caminho_csv']
            nome_excel = caminhos['caminho_excel']
        except OSError as e:
            # Se preparar_caminhos_saida falhar (ex: problema de permissão), retorna falha
            logging.error(f"Falha ao preparar diretório/caminhos de saída: {e}")
            return {'csv_salvo': False, 'excel_salvo': False}

        sucesso_csv = salvar_dataframe_csv(self.df, nome_csv)
        sucesso_excel = salvar_amostras_excel(self.df, nome_coluna_cluster, num_clusters, nome_excel)

        status_salvamento = {'csv_salvo': sucesso_csv, 'excel_salvo': sucesso_excel}
        logging.info(f"Tentativa de salvamento da rodada {rodada_a_salvar} concluída. Status: {status_salvamento}")
        return status_salvamento

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
            classificacao (str): O rótulo/string de classificação a ser atribuído.
                                 Não pode ser uma string vazia.
            rodada (Optional[int], optional): O número da rodada de clusterização
                                              cujos clusters serão classificados.
                                              Se None (padrão), usa a última rodada
                                              de clusterização concluída. Default None.

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
        if 'classificacao' not in self.df.columns:
            logging.info("Coluna 'classificacao' não encontrada. Criando coluna com tipo StringDtype.")
            # Criar diretamente com tipo que aceita nulos e strings
            self.df['classificacao'] = pd.Series(pd.NA, index=self.df.index, dtype=pd.StringDtype())
        else:
            # Se existe, garante que seja um tipo adequado (string ou object)
            # Se for numérico/booleano/etc., a atribuição da string funcionará,
            # mas pode ser menos eficiente ou gerar warnings dependendo da versão do Pandas.
            # A conversão explícita é mais segura se quisermos garantir StringDtype.
            if not pd.api.types.is_string_dtype(self.df['classificacao']) and not pd.api.types.is_object_dtype(self.df['classificacao']):
                 logging.warning(f"Coluna 'classificacao' existe mas não é string/object ({self.df['classificacao'].dtype}). Convertendo para StringDtype.")
                 try:
                     self.df['classificacao'] = self.df['classificacao'].astype(pd.StringDtype())
                 except Exception as e:
                     # Se a conversão falhar (tipos muito mistos), loga erro mas continua.
                     # A atribuição abaixo ainda pode funcionar dependendo do caso.
                     logging.error(f"Falha ao converter coluna 'classificacao' existente para StringDtype: {e}. A classificação pode não ter o tipo ideal.")

        # --- Aplica a Classificação ---
        linhas_a_classificar = self.df[coluna_cluster_alvo].isin(lista_ids)
        num_linhas = linhas_a_classificar.sum()
        logging.info(f"Aplicando classificação '{classificacao}' a {num_linhas} linha(s) correspondente(s) aos clusters {lista_ids} da rodada {rodada_alvo}.")
        self.df.loc[linhas_a_classificar, 'classificacao'] = classificacao

        logging.info("Classificação concluída.")

    # --- MÉTODO DE CONVENIÊNCIA ---
    def finalizar(self, num_clusters: int, prefixo_saida: str = '', diretorio_saida: Optional[str] = None) -> dict[str, bool]:
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
        return status_salvamento
