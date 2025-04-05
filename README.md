# Cluster Fácil 🚀

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)

Uma biblioteca Python intuitiva para realizar clusterização de documentos textuais. Simplifica o processo desde a preparação dos dados e análise do número ideal de clusters até a aplicação do algoritmo e exportação dos resultados. Ideal para agrupar grandes volumes de texto, como decisões judiciais, artigos ou comentários, de forma eficiente e com poucas linhas de código.

## Por que Cluster Fácil?

Cluster Fácil automatiza as etapas mais comuns de pré-processamento e clusterização, permitindo que você foque na análise dos resultados e na interpretação dos grupos formados. Com uma interface simples, mesmo quem está começando pode realizar análises complexas rapidamente.

## Instalação

Você pode instalar a biblioteca diretamente do GitHub usando pip:

```bash
pip install git+https://github.com/bdcdo/cluster-facil.git
```

**Dependências:**

*   **Principais:** `pandas`, `scikit-learn`, `nltk`, `scipy` (instaladas automaticamente).
*   **Plotagem:** `matplotlib` (necessária se `plotar_cotovelo=True` em `preparar`). Instale com `pip install matplotlib`.
*   **Leitura/Escrita Excel:** `openpyxl` (necessária para ler/salvar `.xlsx`). Instale com `pip install openpyxl`.
*   **Leitura/Escrita Parquet:** `pyarrow` (necessária para ler/salvar `.parquet`). Instale com `pip install pyarrow`.
*   **Documentação (Opcional, para construir localmente):** `sphinx`, `sphinx_rtd_theme`, `myst-parser`. Instale com `pip install sphinx sphinx_rtd_theme myst-parser`.

## Uso Rápido (Quick Start)

Clusterizar textos de uma planilha Excel (`.xlsx`) com poucas linhas de código:

*(Nota: Certifique-se de ter `openpyxl` instalado: `pip install openpyxl`)*

```python
from cluster_facil import ClusterFacil

cf = ClusterFacil('suaPlanilha.xlsx')
cf.preparar(coluna_textos='nome_da_coluna_com_textos')
# (Analise o gráfico do cotovelo que será exibido para escolher o K)
cf.finalizar(num_clusters=3) # Substitua 3 pelo K escolhido
```

Isso realizará todo o processo: carregamento, pré-processamento, análise do cotovelo, clusterização e salvamento dos resultados (DataFrame completo e amostras).

Para mais detalhes e opções, veja o exemplo completo em [`examples/uso_basico.ipynb`](examples/uso_basico.ipynb) e a documentação da API abaixo.

## Como Funciona (Resumo Técnico)

Clusterização é uma técnica de aprendizado não supervisionado que visa agrupar itens semelhantes. No contexto de textos, isso significa encontrar documentos que tratam de assuntos parecidos, sem saber previamente quais são esses assuntos.

O Cluster Fácil automatiza um fluxo comum para clusterização de textos usando as seguintes etapas e algoritmos:

1.  **Carregamento e Pré-processamento:**
    *   Os dados são carregados de um arquivo (Excel, CSV, etc.) ou DataFrame.
    *   Os textos da coluna especificada são convertidos para minúsculas e valores nulos são tratados para evitar erros.

2.  **Vetorização TF-IDF:**
    *   Textos não podem ser processados diretamente por algoritmos de clusterização. Eles precisam ser convertidos em vetores numéricos.
    *   A biblioteca utiliza a técnica **TF-IDF (Term Frequency-Inverse Document Frequency)**. Ela calcula um peso para cada palavra em cada documento, dando mais importância a palavras que são frequentes em um documento específico, mas raras no conjunto total de documentos.
    *   Stopwords em português (palavras comuns como "o", "a", "de", que não carregam muito significado distintivo) são removidas durante este processo (usando a lista padrão do NLTK).

3.  **Análise do Número Ideal de Clusters (Método do Cotovelo):**
    *   Um desafio comum na clusterização é definir quantos grupos (K) devem ser formados.
    *   O método `preparar` implementa o **Método do Cotovelo (Elbow Method)**. Ele executa o algoritmo **K-Means** (explicado a seguir) várias vezes, com diferentes valores de K (de 1 até `limite_k`).
    *   Para cada K, calcula-se a **inércia** (soma das distâncias quadráticas dos pontos de dados ao centro do cluster mais próximo).
    *   Um gráfico da Inércia vs. K é plotado. Geralmente, a curva "dobra" (forma um cotovelo) em um ponto que representa um bom equilíbrio entre ter poucos clusters (alta inércia) e ter muitos clusters (diminuição marginal da inércia). A análise visual desse gráfico ajuda a escolher um valor adequado para K.

4.  **Clusterização K-Means:**
    *   Após a escolha de K, o algoritmo **K-Means** é aplicado aos vetores TF-IDF.
    *   O K-Means tenta particionar os dados em K clusters, onde cada ponto de dado pertence ao cluster cujo centro (centróide) está mais próximo. Ele faz isso iterativamente:
        *   Inicializa K centróides aleatoriamente (ou de forma mais inteligente).
        *   Atribui cada ponto de dado ao centróide mais próximo.
        *   Recalcula a posição de cada centróide como a média dos pontos atribuídos a ele.
        *   Repete os dois últimos passos até que os centróides não mudem significativamente ou um número máximo de iterações seja atingido.

5.  **Resultados e Pós-processamento:**
    *   Uma nova coluna é adicionada ao DataFrame original, indicando a qual cluster (de 0 a K-1) cada documento foi atribuído.
    *   A biblioteca permite salvar o DataFrame completo com a nova coluna e também amostras de cada cluster para facilitar a análise e interpretação dos grupos formados.
    *   Funcionalidades como `classificar` e `subcluster` permitem refinar a análise e explorar grupos específicos em mais detalhes.

## Funcionalidades Principais (API)

A classe `ClusterFacil` oferece os seguintes métodos principais:

*   `__init__(entrada, aba=None, prefixo_cluster="cluster_", nome_coluna_classificacao="classificacao", random_state=42)`: Inicializa a classe com um DataFrame ou caminho de arquivo (`.csv`, `.xlsx`, `.parquet`, `.json`). Permite definir prefixos para colunas de cluster, nome da coluna de classificação manual e a semente para reprodutibilidade.
*   `preparar(coluna_textos, limite_k=10, n_init=1, plotar_cotovelo=True, **tfidf_kwargs)`: Realiza o pré-processamento (TF-IDF) e calcula/plota o gráfico do método do cotovelo para ajudar a escolher K. Aceita argumentos do `TfidfVectorizer`.
*   `clusterizar(num_clusters, **kmeans_kwargs)`: Executa o K-Means com o K escolhido. Em rodadas subsequentes (>1), clusteriza apenas linhas não classificadas (se a coluna de classificação existir). Aceita argumentos do `KMeans`. Retorna o nome da coluna de cluster criada (ex: `'cluster_1'`).
*   `classificar(cluster_ids, classificacao, rodada=None)`: Atribui um rótulo (string) a um ou mais clusters de uma rodada específica na coluna de classificação.
*   `subcluster(classificacao_desejada)`: Cria e retorna uma **nova instância** de `ClusterFacil` contendo apenas os dados de uma classificação específica, pronta para uma nova clusterização (com prefixo `'subcluster_'`).
*   `salvar(o_que_salvar='ambos', formato_tudo='csv', formato_amostras='xlsx', caminho_tudo=None, caminho_amostras=None, diretorio_saida=None)`: Salva o DataFrame completo e/ou amostras por cluster em diversos formatos.
*   `finalizar(num_clusters, **kwargs_salvar)`: Método de conveniência que chama `clusterizar()` seguido por `salvar()`.
*   `resetar()`: Remove colunas de cluster/classificação e reseta o estado da instância, permitindo recomeçar o processo no mesmo DataFrame.
*   `listar_classificacoes()`: Retorna uma lista das classificações únicas presentes.
*   `contar_classificacoes()`: Retorna uma Series Pandas com a contagem de cada classificação.
*   `obter_subcluster_df(classificacao_desejada)`: Retorna um DataFrame filtrado por uma classificação, sem iniciar uma nova instância `ClusterFacil`.

## Roadmap Futuro 🗺️

Temos planos para continuar melhorando o Cluster Fácil! Aqui estão algumas ideias:

*   **Configuração do Método do Cotovelo:** Avaliar o uso de `n_init` maior que 1 no cálculo do método do cotovelo para maior estabilidade do gráfico (atualmente `n_init=1` por padrão em `preparar` para agilidade).
*   **Sugestão de K:** Integrar uma ferramenta (como `kneed`) para analisar o gráfico do cotovelo e *sugerir* um número de clusters (K) ideal.
*   **Interpretação dos Clusters:** Adicionar uma funcionalidade para mostrar as palavras/termos mais importantes de cada cluster, ajudando a entender o "tema" de cada grupo.
*   **Conjunto de Dados para Testes:** Criar um conjunto de dados de exemplo (ex: decisões judiciais anonimizadas) para facilitar testes e demonstrações.

Se você tem outras ideias ou gostaria de ajudar, veja a seção de Contribuição!

## Documentação

A documentação completa da API, gerada automaticamente a partir das docstrings do código usando Sphinx, está disponível na pasta `docs/`.

Para construir a documentação HTML localmente:

1.  Instale as dependências: `pip install -r docs/requirements.txt`
2.  Navegue até a pasta `docs/`: `cd docs`
3.  Execute o comando de build: `make html` (Linux/macOS) ou `.\make.bat html` (Windows)
4.  Abra o arquivo `docs/_build/html/index.html` no seu navegador.

Você também pode encontrar um exemplo de uso mais detalhado no notebook [`examples/uso_basico.ipynb`](examples/uso_basico.ipynb).

## Contribuição

Encontrou um bug? Tem sugestões para melhorar a biblioteca? Abra uma [Issue](https://github.com/bdcdo/cluster-facil/issues) no GitHub!

Pull Requests com melhorias ou correções também são muito bem-vindos.

## Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
