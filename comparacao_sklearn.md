# Comparação: Cluster Fácil vs. scikit-learn

Este documento compara a implementação da clusterização de textos usando a biblioteca Cluster Fácil versus uma implementação equivalente usando diretamente as funcionalidades do scikit-learn.

## Objetivo

Demonstrar como o Cluster Fácil simplifica o processo de clusterização de textos, reduzindo significativamente a quantidade de código necessário e abstraindo a complexidade técnica, enquanto mantém toda a funcionalidade e poder do scikit-learn.

## Cenário Comparativo

Consideramos o mesmo cenário para ambas as implementações:
- Carregar dados de uma planilha Excel
- Processar textos de uma coluna específica
- Aplicar TF-IDF para vetorização
- Determinar o número ideal de clusters usando o método do cotovelo
- Executar a clusterização com K-Means
- Salvar os resultados (DataFrame completo e amostras de cada cluster)

## Usando Cluster Fácil (4 linhas de código)

```python
from cluster_facil import ClusterFacil

cf = ClusterFacil('suaPlanilha.xlsx')
cf.preparar(coluna_textos='nome_da_coluna_com_textos')
cf.clusterizar(num_clusters=5)  # Após analisar o gráfico do cotovelo gerado por preparar()
cf.salvar()
```

## Equivalente usando scikit-learn (cerca de 120 linhas de código)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
from nltk.corpus import stopwords

# Lista de stopwords em português (normalmente obtida via NLTK)
STOPWORDS_PT = ['a', 'à', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as',
    'às', 'até', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles',
    'depois', 'do', 'dos', 'e', 'é', 'ela', 'elas', 'ele', 'eles', 'em', 'entre',
    'era', 'eram', 'éramos', 'essa', 'essas', 'esse', 'esses', 'esta', 'está',
    'estamos', 'estão', 'estar', 'estas', 'estava', 'estavam', 'estávamos', 'este',
    'esteja', 'estejam', 'estejamos', 'estes', 'esteve', 'estive', 'estivemos',
    'estiver', 'estivera', 'estiveram', 'estivéramos', 'estiverem', 'estivermos',
    'estivesse', 'estivessem', 'estivéssemos', 'estou', 'eu', 'foi', 'fomos']  # Lista truncada para exemplo

# 1. Carregamento de dados
def carregar_dados(caminho_arquivo, aba=None):
    if caminho_arquivo.endswith('.xlsx'):
        try:
            import openpyxl  # Verificar se a biblioteca está instalada
            df = pd.read_excel(caminho_arquivo, sheet_name=aba)
        except ImportError:
            raise ImportError("A biblioteca 'openpyxl' é necessária para ler arquivos .xlsx")
    elif caminho_arquivo.endswith('.csv'):
        df = pd.read_csv(caminho_arquivo, encoding='utf-8-sig')
    else:
        raise ValueError(f"Formato de arquivo não suportado: {caminho_arquivo}")
    
    return df

# Carregar os dados
caminho_arquivo = 'suaPlanilha.xlsx'
aba = None  # Usar primeira aba
df = carregar_dados(caminho_arquivo, aba)

# 2. Pré-processamento de textos
coluna_textos = 'nome_da_coluna_com_textos'
if coluna_textos not in df.columns:
    raise KeyError(f"A coluna '{coluna_textos}' não foi encontrada no DataFrame")

# Converter para minúsculas e tratar valores nulos
textos_processados = df[coluna_textos].fillna('').astype(str).str.lower()

# 3. Vetorização TF-IDF
vectorizer = TfidfVectorizer(stop_words=STOPWORDS_PT)
X = vectorizer.fit_transform(textos_processados)

# 4. Análise do número ideal de clusters (Método do Cotovelo)
def calcular_inercias_kmeans(X, limite_k, random_state=42):
    # K não pode ser maior que o número de amostras
    k_max = min(limite_k, X.shape[0])
    inercias = []
    
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        kmeans.fit(X)
        inercias.append(kmeans.inertia_)
    
    return range(1, k_max + 1), inercias

# Definir parâmetros para análise do cotovelo
limite_k = 10
random_state = 42

# Calcular inércias para o método do cotovelo
k_range, inercias = calcular_inercias_kmeans(X, limite_k, random_state)

# Plotar o gráfico do cotovelo
plt.figure(figsize=(10, 6))
plt.plot(k_range, inercias, marker='o')
plt.title('Método do Cotovelo para determinação do número de clusters')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia')
plt.grid(True)
plt.show()

# 5. Clusterização K-Means com o K escolhido
# (O usuário deve inspecionar o gráfico e escolher o valor de K)
num_clusters = 5  # Substitua pelo K escolhido após analisar o gráfico
kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init='auto')
cluster_labels = kmeans.fit_predict(X)

# Adicionar os resultados ao DataFrame original
df['cluster'] = cluster_labels

# 6. Salvar resultados em CSV (DataFrame completo)
caminho_saida_completo = 'resultados_completos.csv'
df.to_csv(caminho_saida_completo, index=False, encoding='utf-8-sig')
print(f"DataFrame completo salvo em: {caminho_saida_completo}")

# 7. Gerar e salvar amostras de cada cluster
def gerar_amostras_por_cluster(df, coluna_cluster, num_clusters):
    df_amostras = pd.DataFrame()
    
    for cluster_id in range(num_clusters):
        df_cluster = df[df[coluna_cluster] == cluster_id]
        # Pegar no máximo 10 amostras por cluster
        tamanho_amostra = min(10, len(df_cluster))
        
        if tamanho_amostra > 0:
            # Usar random_state para reprodutibilidade
            amostra_cluster = df_cluster.sample(tamanho_amostra, random_state=42)
            df_amostras = pd.concat([df_amostras, amostra_cluster], ignore_index=True)
    
    return df_amostras

# Gerar amostras
df_amostras = gerar_amostras_por_cluster(df, 'cluster', num_clusters)

# Salvar amostras em Excel
caminho_amostras = 'amostras_clusters.xlsx'
try:
    df_amostras.to_excel(caminho_amostras, index=False)
    print(f"Amostras dos clusters salvas em: {caminho_amostras}")
except ImportError:
    print("A biblioteca 'openpyxl' é necessária para salvar em Excel. Salvando em CSV...")
    df_amostras.to_csv('amostras_clusters.csv', index=False, encoding='utf-8-sig')
```

## Principais Vantagens do Cluster Fácil

1. **Significativa Redução de Código**: 4 linhas de código versus cerca de 120 linhas para a mesma funcionalidade.

2. **Tratamento Automático de Erros**: O Cluster Fácil implementa dezenas de validações que evitam erros comuns:
   - Verificação da existência de colunas
   - Validação de tipos de dados
   - Tratamento de valores nulos
   - Verificação de dependências
   - Validação de parâmetros

3. **Funcionalidades Adicionais**:
   - Múltiplas rodadas de clusterização
   - Classificação manual e subclusters (agrupamento hierárquico)
   - Opções flexíveis de salvamento em diversos formatos (CSV, Excel, Parquet, JSON)
   - Logging detalhado das operações
   - Integração com a estrutura de pastas do projeto

4. **Configurações Avançadas Mantidas**: Apesar da simplificação, o Cluster Fácil permite acessar todos os parâmetros avançados da API scikit-learn:
   ```python
   # Parâmetros avançados do TF-IDF
   cf.preparar(coluna_textos='textos', min_df=5, max_df=0.85, ngram_range=(1, 2))
   
   # Parâmetros avançados do K-Means
   cf.clusterizar(num_clusters=5, max_iter=500, tol=1e-5, algorithm='full')
   ```

5. **Visualização Automática**: O gráfico do cotovelo é gerado automaticamente para facilitar a escolha de K.

6. **Reprodutibilidade Garantida**: O uso consistente de `random_state` garante resultados reproduzíveis.

7. **Menor Dependência de Conhecimento Especializado**: Não é necessário conhecer detalhes da API do scikit-learn ou do pandas para executar operações de clusterização eficientes.

## Suporte para Formatos Variados

Enquanto a implementação manual com scikit-learn exigiria mais código para cada formato suportado, o Cluster Fácil automaticamente gerencia:

- Arquivos CSV (com tratamento de encoding)
- Planilhas Excel (.xlsx)
- Arquivos Parquet (para datasets grandes)
- Arquivos JSON

## Casos de Uso Avançados

O Cluster Fácil também simplifica cenários mais complexos que normalmente exigiriam dezenas de linhas de código adicionais:

### Clusterização Hierárquica

```python
# Com Cluster Fácil
cf = ClusterFacil('dados.xlsx')
cf.preparar(coluna_textos='texto')
cf.clusterizar(num_clusters=5)
cf.classificar(cluster_ids=[0, 2], classificacao='Categoria A')
cf_sub = cf.subcluster('Categoria A')
cf_sub.preparar(coluna_textos='texto')
cf_sub.clusterizar(num_clusters=3)
cf_sub.salvar()

# Com scikit-learn puro: exigiria >100 linhas adicionais
```

### Múltiplas Rodadas de Clusterização

```python
# Com Cluster Fácil
cf = ClusterFacil('dados.xlsx')
cf.preparar(coluna_textos='texto')
cf.clusterizar(num_clusters=5)
cf.classificar(cluster_ids=2, classificacao='Interesse')
# Segunda rodada (processa apenas textos não classificados)
cf.preparar(coluna_textos='texto')
cf.clusterizar(num_clusters=3)
cf.salvar()

# Com scikit-learn puro: exigiria código manual para rastreamento
```

## Conclusão

A biblioteca Cluster Fácil transforma um processo que normalmente requer conhecimento especializado de várias APIs do scikit-learn em uma operação simples e direta, acessível mesmo para quem não tem experiência avançada em processamento de linguagem natural ou machine learning.

Ao abstrair a complexidade técnica sem sacrificar a flexibilidade ou o poder do scikit-learn, o Cluster Fácil permite que pesquisadores, analistas e desenvolvedores foquem no que realmente importa: a interpretação e aplicação dos resultados da clusterização.