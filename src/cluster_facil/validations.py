# -*- coding: utf-8 -*-
"""Módulo de funções de validação para o pacote cluster_facil."""

import importlib
import logging
import os
from typing import Any, Optional, TYPE_CHECKING, Union
import pandas as pd
from scipy.sparse import csr_matrix

# Evita importação circular para type hinting
if TYPE_CHECKING:
    from .cluster import ClusterFacil

# --- Funções de Validação Genéricas ---
def validar_inteiro_positivo(nome_parametro: str, valor: Any) -> None:
    """Valida se um valor é um inteiro positivo."""
    if not isinstance(valor, int) or valor <= 0:
        msg = f"O argumento '{nome_parametro}' deve ser um inteiro positivo (recebeu: {valor})."
        logging.error(msg)
        raise ValueError(msg)

def validar_dependencia(biblioteca: str, mensagem_erro: str) -> None:
    """Verifica se uma biblioteca (dependência) pode ser importada."""
    try:
        importlib.import_module(biblioteca)
        logging.debug(f"Dependência '{biblioteca}' encontrada.")
    except ImportError:
        logging.error(mensagem_erro)
        raise ImportError(mensagem_erro)

# --- Validações de Entrada/Tipo Inicial ---
def validar_entrada_inicial(entrada: Any) -> None:
    """Valida o tipo do argumento de entrada para ClusterFacil."""
    if not isinstance(entrada, (pd.DataFrame, str)):
        logging.error("Tipo de entrada inválido. Deve ser DataFrame ou string (caminho do arquivo).")
        raise TypeError("A entrada deve ser um DataFrame do Pandas ou o caminho (string) para um arquivo.")

# --- Validações de Sistema de Arquivos ---
def validar_arquivo_existe(caminho_arquivo: str) -> None:
    """Verifica se um arquivo existe no caminho especificado."""
    if not os.path.exists(caminho_arquivo):
        logging.error(f"Arquivo não encontrado: {caminho_arquivo}")
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

# --- Validações de Formato de Arquivo ---
def validar_formato_suportado(extensao: str) -> None:
    """Verifica se a extensão do arquivo é suportada para leitura."""
    suportados = ['.csv', '.xlsx', '.parquet', '.json']
    if extensao not in suportados:
        logging.error(f"Formato de arquivo não suportado: {extensao}")
        raise ValueError(f"Formato de arquivo não suportado: {extensao}. Suportados: {', '.join(suportados)}")

# --- Validações de Dependências ---
def validar_dependencia_leitura(extensao: str) -> None:
    """Verifica se as dependências para ler certos formatos de arquivo estão instaladas."""
    if extensao == '.xlsx':
        validar_dependencia(
            'openpyxl',
            "A biblioteca 'openpyxl' é necessária para ler arquivos .xlsx. Instale-a com 'pip install openpyxl'"
        )
    elif extensao == '.parquet':
        validar_dependencia(
            'pyarrow',
            "A biblioteca 'pyarrow' é necessária para ler arquivos .parquet. Instale-a com 'pip install pyarrow'"
        )

# --- Validações de Conteúdo do DataFrame ---
def validar_coluna_existe(df: pd.DataFrame, coluna: str) -> None:
    """Verifica se uma coluna existe no DataFrame."""
    if coluna not in df.columns:
        logging.error(f"A coluna '{coluna}' não foi encontrada no DataFrame.")
        raise KeyError(f"A coluna '{coluna}' não foi encontrada no DataFrame.")

def validar_tipo_coluna_texto(df: pd.DataFrame, coluna: str) -> None:
    """Tenta converter a coluna para string para validar se contém dados processáveis como texto."""
    try:
        # Apenas tenta a conversão, não modifica o df original aqui
        df[coluna].fillna('').astype(str)
    except Exception as e:
        logging.error(f"Erro ao processar a coluna '{coluna}'. Verifique se ela contém texto.")
        raise TypeError(f"Erro ao processar a coluna '{coluna}'. Verifique se ela contém texto. Erro original: {e}")

def validar_coluna_cluster_existe(df: pd.DataFrame, nome_coluna_cluster: str) -> None:
    """Verifica se a coluna de cluster esperada existe no DataFrame (verificação de segurança)."""
    if nome_coluna_cluster not in df.columns:
        # Verificação de segurança, embora improvável se a lógica estiver correta
        logging.error(f"Erro interno: Coluna '{nome_coluna_cluster}' da última clusterização não encontrada.")
        raise KeyError(f"Erro interno: Coluna de cluster '{nome_coluna_cluster}' não encontrada no DataFrame.") # Mudado para KeyError

# --- Validações de Parâmetros ---
def validar_parametro_num_clusters(num_clusters: Any, num_amostras: int) -> None:
    """Valida o tipo e valor do parâmetro num_clusters, comparando com o número de amostras."""
    validar_inteiro_positivo('num_clusters', num_clusters)
    if num_clusters > num_amostras:
        msg = f"O número de clusters ({num_clusters}) não pode ser maior que o número de amostras ({num_amostras})."
        logging.error(msg)
        raise ValueError(msg)

# --- Validações de Estado da Instância ---
def validar_estado_preparado(instance: 'ClusterFacil') -> None:
    """Verifica se o método 'preparar' foi executado e se há dados."""
    if instance.X is None or instance.coluna_textos is None:
        msg = "O método 'preparar' deve ser executado antes desta operação."
        logging.error(msg)
        raise RuntimeError(msg)
    # Verifica se a matriz X existe e tem linhas
    if not isinstance(instance.X, csr_matrix) or instance.X.shape[0] == 0:
         msg = "Não há dados para processar (matriz X vazia). Execute 'preparar' com um DataFrame que contenha dados."
         logging.error(msg)
         raise RuntimeError(msg)

# --- Validações para 'classificar' ---
def validar_rodada_valida(rodada_alvo: int, rodada_atual_sistema: int, prefixo_cluster: str = "cluster_") -> None:
    """
    Valida se a rodada alvo é um inteiro válido dentro do histórico de rodadas.
    Considera o prefixo para mensagens de erro mais claras, embora a lógica principal
    dependa apenas dos números das rodadas.
    """
    if not isinstance(rodada_alvo, int):
        msg = f"O argumento 'rodada' deve ser um inteiro (recebeu: {type(rodada_alvo)})."
        logging.error(msg)
        raise TypeError(msg)
    if not (1 <= rodada_alvo < rodada_atual_sistema):
        # Usa o prefixo na mensagem de erro para clareza
        msg = f"Rodada inválida: {rodada_alvo} para o prefixo '{prefixo_cluster}'. Deve ser entre 1 e {rodada_atual_sistema - 1}."
        logging.error(msg)
        raise ValueError(msg)

def validar_cluster_ids_presentes(df: pd.DataFrame, coluna_cluster: str, cluster_ids: list[int], prefixo_cluster: str = "cluster_") -> None:
    """
    Valida se todos os IDs de cluster fornecidos existem na coluna especificada.
    Usa o prefixo para mensagens de erro mais claras.
    """
    if not all(isinstance(cid, int) for cid in cluster_ids):
        msg = "A lista 'cluster_ids' deve conter apenas inteiros."
        logging.error(msg)
        raise TypeError(msg)
    if not cluster_ids:
        msg = "A lista 'cluster_ids' não pode estar vazia."
        logging.error(msg)
        raise ValueError(msg)

    # A coluna_cluster já vem com o prefixo correto do método classificar
    valores_unicos = df[coluna_cluster].dropna().unique() # Adiciona dropna para segurança
    ids_nao_encontrados = [cid for cid in cluster_ids if cid not in valores_unicos]
    if ids_nao_encontrados:
        # Usa o prefixo na mensagem de erro para clareza
        rodada_num_match = re.search(r'(\d+)$', coluna_cluster)
        rodada_str = f" (rodada {rodada_num_match.group(1)})" if rodada_num_match else ""
        msg = f"Os seguintes IDs de cluster não foram encontrados na coluna '{coluna_cluster}'{rodada_str} para o prefixo '{prefixo_cluster}': {ids_nao_encontrados}."
        logging.error(msg)
        raise ValueError(msg)

def validar_tipo_classificacao(classificacao: Any) -> None:
    """Valida se a classificação é uma string não vazia."""
    if not isinstance(classificacao, str):
        msg = f"A 'classificacao' deve ser uma string (recebeu: {type(classificacao)})."
        logging.error(msg)
        raise TypeError(msg)
    if not classificacao: # Verifica se a string não está vazia
        msg = "A 'classificacao' não pode ser uma string vazia."
        logging.error(msg)
        raise ValueError(msg)

def validar_estado_clusterizado(instance: 'ClusterFacil') -> None:
    """Verifica se o método 'clusterizar' foi executado pelo menos uma vez."""
    if instance._ultima_coluna_cluster is None or instance._ultimo_num_clusters is None:
        msg = "Nenhuma clusterização foi realizada ainda. Execute o método 'clusterizar' primeiro."
        logging.error(msg)
        raise RuntimeError(msg)

# --- Validações para 'salvar' ---
def validar_opcao_salvar(opcao: str) -> None:
    """Valida se a opção para o parâmetro 'o_que_salvar' é válida."""
    opcoes_validas = ['tudo', 'amostras', 'ambos']
    if opcao not in opcoes_validas:
        msg = f"Opção inválida para 'o_que_salvar': '{opcao}'. Use uma das opções: {opcoes_validas}."
        logging.error(msg)
        raise ValueError(msg)

def validar_formato_salvar(formato: str, tipo: str) -> None:
    """Valida se o formato de arquivo para salvar é suportado para o tipo especificado."""
    formatos_tudo = ['csv', 'xlsx', 'parquet', 'json']
    formatos_amostras = ['xlsx', 'csv', 'json']

    if tipo == 'tudo':
        if formato not in formatos_tudo:
            msg = f"Formato inválido para 'formato_tudo': '{formato}'. Use um dos formatos: {formatos_tudo}."
            logging.error(msg)
            raise ValueError(msg)
        # Valida dependências para formatos específicos (ex: pyarrow para parquet)
        validar_dependencia_leitura(f".{formato}") # Reutiliza validação de dependência de leitura
    elif tipo == 'amostras':
        if formato not in formatos_amostras:
            msg = f"Formato inválido para 'formato_amostras': '{formato}'. Use um dos formatos: {formatos_amostras}."
            logging.error(msg)
            raise ValueError(msg)
        # Valida dependências para formatos específicos (ex: openpyxl para xlsx)
        validar_dependencia_leitura(f".{formato}") # Reutiliza validação de dependência de leitura
    else:
        # Erro interno se o tipo de validação for inesperado
        msg = f"Tipo de validação de formato desconhecido: '{tipo}'."
        logging.error(msg)
        raise ValueError(msg)
