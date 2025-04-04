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

A instalação cuidará automaticamente das dependências necessárias: `pandas`, `scikit-learn`, `nltk`, `matplotlib` e `scipy`.

## Uso Rápido (Quick Start)

Veja como é simples usar o Cluster Fácil:

```python
import pandas as pd
from cluster_facil import ClusterFacil # Importa a classe principal

# 1. Seus dados (exemplo simples com textos de decisões)
data = {'id': [1, 2, 3, 4, 5, 6],
        'texto_decisao': ["Juiz decidiu a favor do autor na sentença.",
                          "Réu apresentou recurso de apelação contra a decisão.",
                          "Sentença foi favorável ao requerente em primeira instância.",
                          "Apelação interposta pelo réu foi negada pelo tribunal.",
                          "O processo foi arquivado por falta de provas.",
                          "Caso encerrado sem resolução de mérito."]}
df = pd.DataFrame(data)

# 2. Inicialize o ClusterFacil com seu DataFrame
cf = ClusterFacil(df)

# 3. Prepare os dados e veja o gráfico do cotovelo
#    O gráfico será exibido automaticamente para ajudar a escolher o número de clusters (K)
print("Analisando o número ideal de clusters (gráfico do cotovelo será exibido)...")
# Use a coluna que contém os textos e defina até quantos clusters testar (limite_k)
cf.preparar(coluna_textos='texto_decisao', limite_k=5)

# --- PAUSA PARA ANÁLISE ---
# Olhe o gráfico gerado. Onde a linha "dobra" (o cotovelo)?
# Esse ponto sugere um bom número de clusters (K).
# Vamos supor que você escolheu K=2 para este exemplo.

# 4. Escolha o K e finalize a clusterização
print("\nFinalizando com K=2...")
# Informe o número de clusters escolhido (num_clusters)
# Opcional: defina um prefixo para os arquivos de saída (prefixo_saida)
cf.finaliza(num_clusters=2, prefixo_saida='meu_projeto')
# Isso salvará:
# - 'meu_projeto_clusters_1.csv': Seu DataFrame original com uma nova coluna 'cluster_1'
# - 'meu_projeto_amostras_por_cluster_1.xlsx': Um arquivo Excel com até 10 amostras de cada cluster

# 5. Veja o resultado diretamente no DataFrame
print("\nDataFrame com a coluna de clusters adicionada:")
print(cf.df)

# Próxima rodada? Se quiser rodar de novo com outro K ou outros dados no mesmo objeto:
# cf.preparar(...)
# cf.finaliza(num_clusters=3, prefixo_saida='tentativa_k3') # Gerará arquivos com _2 no final
```

Para um exemplo mais detalhado e interativo, veja o Jupyter Notebook na pasta [`examples/uso_basico.ipynb`](examples/uso_basico.ipynb).

## Como Funciona (Resumo Técnico)

Para os curiosos, o Cluster Fácil segue estes passos principais:

1.  **Pré-processamento:** Os textos da coluna especificada são convertidos para minúsculas e valores nulos são tratados.
2.  **Vetorização TF-IDF:** Os textos são transformados em vetores numéricos usando a técnica TF-IDF (Term Frequency-Inverse Document Frequency), que pondera a importância das palavras nos documentos. Stopwords em português (palavras comuns como "o", "a", "de") são removidas (usando NLTK).
3.  **Método do Cotovelo:** Para ajudar na escolha do número ideal de clusters (K), o algoritmo K-Means é executado para diferentes valores de K (de 1 até `limite_k`). A "inércia" (soma das distâncias quadráticas dentro de cada cluster) é calculada para cada K. O gráfico da inércia vs. K geralmente mostra um "cotovelo", indicando um ponto onde adicionar mais clusters não traz uma melhoria significativa na separação.
4.  **Clusterização K-Means:** Após você escolher o número de clusters (K) com base no gráfico do cotovelo, o algoritmo K-Means é aplicado final para agrupar os documentos nos K clusters definidos.
5.  **Resultados:** Uma nova coluna indicando o cluster de cada documento é adicionada ao seu DataFrame original. Opcionalmente, arquivos CSV e Excel com os resultados e amostras são salvos.

## Roadmap Futuro 🗺️

Temos planos para continuar melhorando o Cluster Fácil! Aqui estão algumas ideias e funcionalidades que gostaríamos de adicionar no futuro:

*   **Entrada de Dados Aprimorada:** Permitir carregar dados diretamente de uma planilha específica dentro de um arquivo Excel (`.xlsx`).
*   **Gerenciamento de Rodadas:** Identificar rodadas de clusterização anteriores no DataFrame e permitir ao usuário sobreescrevê-las se desejar.
*   **Feedback de Erros:** Melhorar o feedback ao usuário caso ocorra erro apenas no salvamento dos arquivos de resultado (CSV/Excel).
*   **Sugestão de K:** Integrar uma ferramenta (como `kneed`) para analisar o gráfico do cotovelo e *sugerir* um número de clusters (K) ideal, auxiliando usuários iniciantes.
*   **Diretório de Saída:** Permitir configurar uma pasta específica para salvar todos os arquivos gerados pela biblioteca.
*   **Interpretação dos Clusters:** Adicionar uma funcionalidade para mostrar as palavras/termos mais importantes de cada cluster, ajudando a entender o "tema" de cada grupo.
*   **Biblioteca para testes:** Criar uma biblioteca opcional com um conjunto de decisões judiciais para a realização de testes.

Se você tem outras ideias ou gostaria de ajudar com alguma dessas, veja a seção de Contribuição!

## Contribuição

Encontrou um bug? Tem sugestões para melhorar a biblioteca? Abra uma [Issue](https://github.com/bdcdo/cluster-facil/issues) no GitHub!

Pull Requests com melhorias ou correções também são muito bem-vindos.

## Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
