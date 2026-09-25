# Segmentação de Clientes com Aprendizado Não Supervisionado

**UNISINOS · Aprendizado Não Supervisionado · Atividade Prática**

Segmentação de clientes do [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) com RFM (Recency, Frequency, Monetary) e três algoritmos de clustering (K-Means, Hierárquico e DBSCAN), com tradução dos segmentos em recomendações de negócio.

- **Dashboard interativo:** https://unisinos-segmentacao-clientes-rfm.streamlit.app
- **Notebook completo:** [`segmentacao_clientes.ipynb`](segmentacao_clientes.ipynb)

## Resultados

4.317 clientes segmentados em 4 grupos via K-Means (k=4). Valores em libras esterlinas (£), com receita líquida de devoluções.

| Segmento | Clientes | Recência (dias) | Pedidos | Valor (£) | % Receita |
|---|---:|---:|---:|---:|---:|
| Champions | 694 | 8 | 10 | 3.718 | 64,4% |
| Loyal Customers | 1.181 | 53 | 4 | 1.325 | 23,5% |
| Lost | 1.609 | 177 | 1 | 292 | 6,7% |
| New Customers | 833 | 17 | 2 | 460 | 5,4% |

*Recência, pedidos e valor são medianas por cliente.*

Principais achados:

- **Concentração:** 16% dos clientes (Champions) geram 64,4% da receita.
- **Risco individual, não por segmento:** 457 clientes recorrentes já passaram do dobro do próprio intervalo entre compras (£ 631 mil em receita histórica). A maioria está em segmentos saudáveis, como Loyal Customers.
- **Atrito operacional:** Champions e Loyal Customers emitem o dobro de notas de cancelamento/devolução dos demais, com valor devolvido baixo (~2%), o que indica muitos ajustes pequenos típicos de compradores de volume.

Métricas do modelo final: silhueta 0,339 · estabilidade sob reamostragem (ARI) 0,918 · Hopkins 0,944.

## Metodologia

1. **Limpeza:** remoção de transações sem `CustomerID` e de lançamentos de ajuste. Cancelamentos e devoluções (notas `C...`) são separados e descontados do valor de cada cliente.
2. **RFM:** data de referência fixa em 10/12/2011. O Monetary é líquido (vendas menos devoluções), e clientes com saldo ≤ 0 ficam de fora.
3. **Preparação:** `log1p` seguido de `StandardScaler`.
4. **Clustering:**
   - K-Means com escolha de k por inércia, silhueta e estabilidade;
   - Hierárquico com linkages ward, complete e average, e dendrograma;
   - DBSCAN com `eps` escolhido pelo k-distance graph.
5. **Validação:** silhueta, Davies-Bouldin, Calinski-Harabasz, estabilidade por reamostragem (ARI), convergência sob múltiplas inicializações e estatística de Hopkins.
6. **Nomeação dos segmentos:** regra baseada em R, F e M frente à mediana global, validada com duas variáveis auxiliares que não entram no clustering: *Tenure* (tempo desde a primeira compra) e *Atraso* (recência dividida pelo intervalo médio entre compras do próprio cliente).

**Limitação principal:** a estrutura de agrupamento é fraca (silhueta moderada; o DBSCAN não separa Champions de Loyal por densidade). Os segmentos são uma partição útil de um contínuo, não grupos naturalmente separados. As limitações estão discutidas em detalhe no notebook e no relatório.

## Estrutura do repositório

```
.
├── segmentacao_clientes.ipynb   # análise completa: EDA, RFM, clustering, validação, visualizações
├── streamlitapp.py              # dashboard interativo (Streamlit + Plotly)
├── requirements.txt             # dependências do dashboard
└── data/
    ├── online_retail.xlsx       # dataset original (UCI Online Retail)
    ├── online+retail.zip        # dataset original compactado
    └── rfm_clusters.csv         # saída do notebook, consumida pelo dashboard
```

## Como executar

### Dashboard

```bash
pip install -r requirements.txt
streamlit run streamlitapp.py
```

Abre em `http://localhost:8501`. O app lê `data/rfm_clusters.csv`.

Requer Python 3.10 ou superior.

### Notebook

O notebook usa bibliotecas adicionais às do dashboard:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn plotly openpyxl jupyter
jupyter notebook segmentacao_clientes.ipynb
```

Execute com **Restart Kernel and Run All Cells**. Todas as etapas aleatórias usam semente fixa (`RS = 42`), então os resultados são reproduzíveis. A última célula regrava `data/rfm_clusters.csv`.

## Dataset

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

Transações de um varejista online do Reino Unido entre dez/2010 e dez/2011 (541.909 linhas).
