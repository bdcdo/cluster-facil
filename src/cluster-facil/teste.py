import pandas as pd
from cluster import ClusterFacil
df = pd.read_csv('doençasRaras_semDuplicatas.csv')
df_clusters = ClusterFacil(df)
df_clusters.preparar('texto')
df_clusters.finaliza(4)