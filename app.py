import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import json
import urllib.request

# ======================
# CONFIGURAÇÃO INICIAL
# ======================
st.set_page_config(page_title="Dashboard RTVs", layout="wide")

URL_SHEET = "https://docs.google.com/spreadsheets/d/1OcX3jxFqnEEKe7wqJVm_pl4QYz3foCNLM1Dnf9aYie4/export?format=xlsx"

# ======================
# CARREGAR DADOS
# ======================
@st.cache_data(ttl=600)  # Atualiza a cada 10 minutos
def load_data():
    xls = pd.ExcelFile(URL_SHEET)
    df = pd.read_excel(xls, sheet_name="RTVs")
    municipios = pd.read_excel(xls, sheet_name="Municípios")
    return df, municipios

df, municipios = load_data()

# Carregar GeoJSON (arquivo local no repositório)
with open("mun_PR.json", "r", encoding="utf-8") as f:
    geojson = json.load(f)

# ======================
# LIMPEZA DE DADOS
# ======================
df = df.dropna(how="all")  # Remove linhas 100% vazias
df = df.fillna(0)          # Substitui N/A numéricos por 0

# Normalizar strings
df["Município"] = df["Município"].astype(str)
df["Região"] = df["Região"].astype(str)
df["STATUS"] = df["STATUS"].astype(str)

# Converter datas
if "Previsão Entrega" in df.columns:
    df["Previsão Entrega"] = pd.to_datetime(df["Previsão Entrega"], errors="coerce")

# ======================
# SIDEBAR - FILTROS
# ======================
st.sidebar.header("🔎 Filtros")

regioes = st.sidebar.multiselect("Região", options=sorted(df["Região"].unique()))
municipios_filtro = st.sidebar.multiselect("Município", options=sorted(df["Município"].unique()))
status_filtro = st.sidebar.multiselect("Status", options=sorted(df["STATUS"].unique()))

criterio_mapa = st.sidebar.radio(
    "Colorir mapa por:",
    ["Extensão Total (km)", "Quantidade de RTVs"]
)

# Aplicar filtros
df_filtered = df.copy()
if regioes:
    df_filtered = df_filtered[df_filtered["Região"].isin(regioes)]
if municipios_filtro:
    df_filtered = df_filtered[df_filtered["Município"].isin(municipios_filtro)]
if status_filtro:
    df_filtered = df_filtered[df_filtered["STATUS"].isin(status_filtro)]

# ======================
# KPIs
# ======================
st.title("📊 Dashboard de Monitoramento de RTVs")
st.markdown("### Visão geral dos Relatórios Técnicos de Vistoria (RTVs)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📄 Total RTVs", len(df_filtered))
col2.metric("📏 Extensão Total (km)", round(df_filtered["Extensão (km)"].sum(), 2))
col3.metric("🏘 Municípios", df_filtered["Município"].nunique())
col4.metric("✅ Percentual Entregue", f"{100 * len(df[df['STATUS']=='Entregue'])/len(df):.1f} %")

st.divider()

# ======================
# GRÁFICO INCREMENTAL
# ======================
st.subheader("📈 Entregas Acumuladas (Curva S)")

df_entregues = df_filtered[(df_filtered["STATUS"] == "Entregue") & (df_filtered["Entregue"] != 0)]

if not df_entregues.empty and not df_entregues["Previsão Entrega"].isna().all():
    entregas_por_data = (
        df_entregues.groupby("Previsão Entrega")
        .size()
        .reset_index(name="Qtd")
        .sort_values("Previsão Entrega")
    )
    entregas_por_data["Acumulado"] = entregas_por_data["Qtd"].cumsum()

    fig = px.line(
        entregas_por_data,
        x="Previsão Entrega",
        y="Acumulado",
        markers=True,
        line_shape="linear",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Nenhuma entrega registrada para os filtros aplicados.")

st.divider()

# ======================
# GRÁFICOS AUXILIARES
# ======================
colA, colB = st.columns(2)

with colA:
    st.subheader("📌 Distribuição por Status")
    fig_status = px.histogram(df_filtered, x="STATUS", color="STATUS")
    st.plotly_chart(fig_status, use_container_width=True)

with colB:
    st.subheader("🛣 Distribuição por Pavimento")
    fig_pav = px.histogram(df_filtered, x="Pavimento", color="Pavimento")
    st.plotly_chart(fig_pav, use_container_width=True)

st.subheader("🌍 RTVs por Região")
fig_reg = px.bar(
    df_filtered.groupby("Região").size().reset_index(name="RTVs"),
    x="Região", y="RTVs", color="Região"
)
st.plotly_chart(fig_reg, use_container_width=True)

st.divider()

# ======================
# MAPA INTERATIVO (sem geopandas)
# ======================
st.subheader("🗺 Mapa de Municípios com RTVs")

map_data = (
    df_filtered.groupby(["CodIBGE", "Município"])
    .agg({"Extensão (km)": "sum", "STATUS": "count"})
    .reset_index()
    .rename(columns={"Extensão (km)": "Extensao_km", "STATUS": "Qtd_RTVs"})
)

# Criar dicionário para merge manual
map_dict = map_data.set_index("CodIBGE")[["Extensao_km", "Qtd_RTVs"]].to_dict("index")

# Injetar atributos no GeoJSON
for feature in geojson["features"]:
    cod = int(feature["properties"]["CD_MUN"])
    if cod in map_dict:
        feature["properties"]["Extensao_km"] = float(map_dict[cod]["Extensao_km"])
        feature["properties"]["Qtd_RTVs"] = int(map_dict[cod]["Qtd_RTVs"])
    else:
        feature["properties"]["Extensao_km"] = 0
        feature["properties"]["Qtd_RTVs"] = 0

# Definir coluna de coloração
coluna_color = "Extensao_km" if criterio_mapa == "Extensão Total (km)" else "Qtd_RTVs"

# Calcular max para normalização
max_val = max([f["properties"][coluna_color] for f in geojson["features"]]) or 1

# Atribuir escala de cor normalizada
for feature in geojson["features"]:
    val = feature["properties"][coluna_color]
    intensity = int((val / max_val) * 255)
    feature["properties"]["color_value"] = intensity

# Camada Pydeck
layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="[properties.color_value, 70, 160, 200]",
    get_line_color=[50, 50, 50],
)

view_state = pdk.ViewState(latitude=-24.5, longitude=-51.5, zoom=6)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "{Município}\nExtensão: {Extensao_km} km\nRTVs: {Qtd_RTVs}"}
))

# ======================
# LEGENDA FLUTUANTE
# ======================
st.markdown(f"""
<style>
.legend {{
  position: relative;
  width: 250px;
  height: 40px;
  margin-top: -40px;
}}
.bar {{
  height: 15px;
  background: linear-gradient(to right, rgba(0,70,160,0.2), rgba(255,70,160,0.9));
}}
.labels {{
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  margin-top: 3px;
}}
</style>

<div class="legend">
  <div><b>Legenda – {criterio_mapa}</b></div>
  <div class="bar"></div>
  <div class="labels"><span>0</span><span>{int(max_val)}</span></div>
</div>
""", unsafe_allow_html=True)
