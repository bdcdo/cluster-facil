import pandas as pd
import os
import logging
from typing import Any, Optional, TYPE_CHECKING
from scipy.sparse import csr_matrix

# Evita importação circular para type hinting
if TYPE_CHECKING:
    from .cluster import ClusterFacil

def validar_entrada_inicial(entrada: Any) -> None:
    """Valida o tipo do argumento de entrada para ClusterFacil."""
    if not isinstance(entrada, (pd.DataFrame, str)):
        logging.error("Tipo de entrada inválido. Deve ser DataFrame ou string (caminho do arquivo).")
        raise TypeError("A entrada deve ser um DataFrame do Pandas ou o caminho (string) para um arquivo.")

def validar_arquivo_existe(caminho_arquivo: str) -> None:
    """Verifica se um arquivo existe no caminho especificado."""
    if not os.path.exists(caminho_arquivo):
        logging.error(f"Arquivo não encontrado: {caminho_arquivo}")
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

def validar_dependencia_leitura(extensao: str) -> None:
    """Verifica se as dependências para ler certos formatos de arquivo estão instaladas."""
    if extensao == '.xlsx':
        try:
            import openpyxl
        except ImportError:
            msg = "A biblioteca 'openpyxl' é necessária para ler arquivos .xlsx. Instale-a com 'pip install openpyxl'"
            logging.error(msg)
            raise ImportError(msg)
    elif extensao == '.parquet':
        try:
            import pyarrow
        except ImportError:
            msg = "A biblioteca 'pyarrow' é necessária para ler arquivos .parquet. Instale-a com 'pip install pyarrow'"
            logging.error(msg)
            raise ImportError(msg)

def validar_formato_suportado(extensao: str) -> None:
    """Verifica se a extensão do arquivo é suportada para leitura."""
    suportados = ['.csv', '.xlsx', '.parquet', '.json']
    if extensao not in suportados:
        logging.error(f"Formato de arquivo não suportado: {extensao}")
        raise ValueError(f"Formato de arquivo não suportado: {extensao}. Suportados: {', '.join(suportados)}")

def validar_coluna_existe(df: pd.DataFrame, coluna: str) -> None:
    """Verifica se uma coluna existe no DataFrame."""
    if coluna not in df.columns:
        logging.error(f"A coluna '{coluna}' não foi encontrada no DataFrame.")
        raise KeyError(f"A coluna '{coluna}' não foi encontrada no DataFrame.")

def validar_parametro_limite_k(limite_k: Any) -> None:
    """Valida o tipo e valor do parâmetro limite_k."""
    if not isinstance(limite_k, int) or limite_k <= 0:
        logging.error("O argumento 'limite_k' deve ser um inteiro positivo.")
        raise ValueError("O argumento 'limite_k' deve ser um inteiro positivo.")

def validar_tipo_coluna_texto(df: pd.DataFrame, coluna: str) -> None:
    """Tenta converter a coluna para string para validar se contém dados textuais."""
    try:
        # Apenas tenta a conversão, não modifica o df original aqui
        df[coluna].fillna('').astype(str)
    except Exception as e:
        logging.error(f"Erro ao processar a coluna '{coluna}'. Verifique se ela contém texto.")
        raise TypeError(f"Erro ao processar a coluna '{coluna}'. Verifique se ela contém texto. Erro original: {e}")

def validar_estado_preparado(instance: 'ClusterFacil') -> None:
    """Verifica se o método 'preparar' foi executado."""
    if instance.X is None or instance.coluna_textos is None:
        logging.error("O método 'preparar' deve ser executado antes desta operação.")
        raise RuntimeError("O método 'preparar' deve ser executado antes desta operação.")
    if isinstance(instance.X, csr_matrix) and instance.X.shape[0] == 0:
         logging.error("Não há dados para processar. Execute 'preparar' com um DataFrame que contenha dados.")
         raise RuntimeError("Não há dados para processar. Execute 'preparar' com um DataFrame que contenha dados.")

def validar_parametro_num_clusters(num_clusters: Any, num_amostras: int) -> None:
    """Valida o tipo e valor do parâmetro num_clusters."""
    if not isinstance(num_clusters, int) or num_clusters <= 0:
        logging.error("O argumento 'num_clusters' deve ser um inteiro positivo.")
        raise ValueError("O argumento 'num_clusters' deve ser um inteiro positivo.")
    if num_clusters > num_amostras:
        logging.error(f"O número de clusters ({num_clusters}) não pode ser maior que o número de amostras ({num_amostras}).")
        raise ValueError(f"O número de clusters ({num_clusters}) não pode ser maior que o número de amostras ({num_amostras}).")

def validar_estado_clusterizado(instance: 'ClusterFacil') -> None:
    """Verifica se o método 'clusterizar' foi executado pelo menos uma vez."""
    if instance._ultima_coluna_cluster is None or instance._ultimo_num_clusters is None:
        logging.error("Nenhuma clusterização foi realizada ainda. Execute o método 'clusterizar' primeiro.")
        # Retorna False em vez de levantar exceção para o método salvar poder lidar com isso
        # raise RuntimeError("Nenhuma clusterização foi realizada ainda. Execute o método 'clusterizar' primeiro.")
        # Decidi manter o raise para consistência, o método salvar fará a checagem antes de chamar utils
        raise RuntimeError("Nenhuma clusterização foi realizada ainda. Execute o método 'clusterizar' primeiro.")

def validar_coluna_cluster_existe(df: pd.DataFrame, nome_coluna_cluster: str) -> None:
    """Verifica se a coluna de cluster esperada existe no DataFrame (verificação de segurança)."""
    if nome_coluna_cluster not in df.columns:
        # Verificação de segurança, embora improvável se a lógica estiver correta
        logging.error(f"Erro interno: Coluna '{nome_coluna_cluster}' da última clusterização não encontrada.")
        raise RuntimeError(f"Erro interno: Coluna '{nome_coluna_cluster}' da última clusterização não encontrada.")
