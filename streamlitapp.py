import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Segmentação de Clientes — Dashboard", layout="wide")

PALETA = ["#0B7285", "#D9480F", "#5F3DC4", "#2B8A3E", "#B08A2E", "#9C36B5"]
FEATURES = ["Recency", "Frequency", "Monetary"]


@st.cache_data
def carregar_dados(caminho="data/rfm_clusters.csv"):
    return pd.read_csv(caminho)


df = carregar_dados()

st.title("Segmentação de Clientes — RFM")
st.caption("UNISINOS · Aprendizado Não Supervisionado · UCI Online Retail Dataset · valores em libras (£), receita líquida de devoluções")

segmentos_disponiveis = sorted(df["segmento"].unique())
cor_por_segmento = {seg: PALETA[i % len(PALETA)] for i, seg in enumerate(segmentos_disponiveis)}
df_filtrado = df

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clientes", f"{len(df_filtrado):,}")
col2.metric("Receita líquida total", f"£ {df_filtrado['Monetary'].sum():,.0f}")
col3.metric("Segmentos", len(segmentos_disponiveis))
col4.metric("Segmento de maior receita", df.groupby("segmento")["Monetary"].sum().idxmax())

st.divider()

# --- linha 1: dispersão PCA + pizza de receita ---
c1, c2 = st.columns([2, 1])

with c1:
    fig_scatter = px.scatter(
        df_filtrado, x="PC1", y="PC2", color="segmento",
        color_discrete_map=cor_por_segmento, opacity=0.6,
        title="Clientes por segmento — projeção PCA 2D",
        labels={"PC1": "PC1", "PC2": "PC2"},
    )
    fig_scatter.update_layout(legend_title_text="Segmento")
    st.plotly_chart(fig_scatter, width="stretch")

with c2:
    receita_segmento = df_filtrado.groupby("segmento")["Monetary"].sum().reset_index()
    fig_pizza = px.pie(
        receita_segmento, values="Monetary", names="segmento",
        color="segmento", color_discrete_map=cor_por_segmento,
        title="Contribuição de receita",
    )
    st.plotly_chart(fig_pizza, width="stretch")

# --- linha 2: radar + barras ---
c3, c4 = st.columns(2)

with c3:
    # mesma normalização do notebook: min-max em espaço log1p, com Recency invertida
    # para que "mais para fora" signifique "melhor" nos três eixos
    perfil_log = np.log1p(df_filtrado.groupby("segmento")[FEATURES].median())
    perfil_norm = (perfil_log - perfil_log.min()) / (perfil_log.max() - perfil_log.min() + 1e-9)
    perfil_norm["Recency"] = 1 - perfil_norm["Recency"]
    eixos = ["Recency (invertida)", "Frequency", "Monetary"]
    fig_radar = go.Figure()
    for seg in perfil_norm.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=perfil_norm.loc[seg].values, theta=eixos, fill="toself",
            name=seg, line_color=cor_por_segmento.get(seg),
        ))
    fig_radar.update_layout(
        title="Perfil normalizado por segmento (mais para fora = melhor)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    )
    st.plotly_chart(fig_radar, width="stretch")

with c4:
    contagem = df_filtrado["segmento"].value_counts().reset_index()
    contagem.columns = ["segmento", "clientes"]
    fig_bar = px.bar(
        contagem, x="segmento", y="clientes", color="segmento",
        color_discrete_map=cor_por_segmento, title="Clientes por segmento",
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, width="stretch")

# --- linha 3: boxplot + tabela ---
c5, c6 = st.columns([1, 1])

with c5:
    fig_box = px.box(
        df_filtrado, x="segmento", y="Monetary", color="segmento",
        color_discrete_map=cor_por_segmento, log_y=True,
        title="Distribuição de Monetary líquido por segmento (£, log)",
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, width="stretch")

with c6:
    st.subheader("Perfil por segmento")
    tabela = df_filtrado.groupby("segmento").agg(
        clientes=("CustomerID", "count"),
        recency_mediana=("Recency", "median"),
        frequency_mediana=("Frequency", "median"),
        monetary_mediana_gbp=("Monetary", "median"),
        receita_total_gbp=("Monetary", "sum"),
    ).round(1)
    tabela["receita_%"] = (tabela["receita_total_gbp"] / tabela["receita_total_gbp"].sum() * 100).round(1)
    st.dataframe(tabela, width="stretch")

st.divider()
st.caption("Dashboard gerado a partir da segmentação K-Means (RFM). Gráficos são interativos — use zoom, hover e clique na legenda para isolar segmentos.")
