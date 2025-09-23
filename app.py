import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import json
from PIL import Image

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
# Remover linhas 100% vazias
df = df.dropna(how="all")

# Corrigir coluna de extensão (vírgula/ponto → float)
if "Extensão (km)" in df.columns:
    df["Extensão (km)"] = (
        df["Extensão (km)"]
        .astype(str)
        .str.replace(".", ",")   # uniformiza
        .str.replace(",", ".")   # Python-friendly
    )
    df["Extensão (km)"] = pd.to_numeric(df["Extensão (km)"], errors="coerce").fillna(0)

# Normalizar strings nas colunas principais
for col in ["Município", "Região", "STATUS"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

# Corrigir CodIBGE para int
if "CodIBGE" in df.columns:
    df["CodIBGE"] = pd.to_numeric(df["CodIBGE"], errors="coerce").fillna(0).astype(int)

# Converter datas
if "Previsão Entrega" in df.columns:
    df["Previsão Entrega"] = pd.to_datetime(df["Previsão Entrega"], errors="coerce")

# Remover registros sem município ou região
df = df[df["Município"].notna() & (df["Município"] != "0") & (df["Município"] != "nan")]
df = df[df["Região"].notna() & (df["Região"] != "0") & (df["Região"] != "nan")]

# ======================
# SIDEBAR - LOGO + FILTROS
# ======================
st.sidebar.image("logo_idr.png", use_column_width=True)  # Logo no topo da sidebar
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
st.subheader("📈 Entregas Acumuladas em 2025 (Curva S)")

df_entregues = df_filtered[(df_filtered["STATUS"] == "Entregue") & (df_filtered["Entregue"] != 0)]
df_entregues = df_entregues[df_entregues["Previsão Entrega"].dt.year == 2025]

if not df_entregues.empty and not df_entregues["Previsão Entrega"].isna().all():
    entregas_por_data = (
        df_entregues.groupby("Previsão Entrega")
        .size()
        .reset_index(name="Qtd")
        .sort_values("Previsão Entrega")
    )

    # Criar calendário completo de 2025
    calendario_2025 = pd.DataFrame({
        "Previsão Entrega": pd.date_range("2025-01-01", "2025-12-31", freq="D")
    })

    # Merge para preencher dias sem entregas
    entregas_por_data = calendario_2025.merge(
        entregas_por_data, on="Previsão Entrega", how="left"
    ).fillna(0)

    entregas_por_data["Qtd"] = entregas_por_data["Qtd"].astype(int)
    entregas_por_data["Acumulado"] = entregas_por_data["Qtd"].cumsum()

    # Gráfico
    fig = px.line(
        entregas_por_data,
        x="Previsão Entrega",
        y="Acumulado",
        markers=True,
        line_shape="linear",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Nenhuma entrega registrada para 2025 nos filtros aplicados.")

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
df_regioes = df_filtered[df_filtered["Região"].notna() & (df_filtered["Região"] != "nan")]
fig_reg = px.bar(
    df_regioes.groupby("Região").size().reset_index(name="RTVs"),
    x="Região", y="RTVs", color="Região"
)
st.plotly_chart(fig_reg, use_container_width=True)

st.divider()

# ======================
# MAPA INTERATIVO
# ======================
st.subheader("🗺 Mapa de Municípios com RTVs")

map_data = (
    df_filtered.groupby(["CodIBGE", "Município"], as_index=False)
    .agg({"Extensão (km)": "sum", "STATUS": "count"})
    .rename(columns={"Extensão (km)": "Extensao_km", "STATUS": "Qtd_RTVs"})
)

map_dict = map_data.drop_duplicates("CodIBGE").set_index("CodIBGE")[["Extensao_km", "Qtd_RTVs"]].to_dict("index")

# Detectar chave correta no GeoJSON
geojson_keys = list(geojson["features"][0]["properties"].keys())
geojson_key = None
for k in geojson_keys:
    if "CD" in k.upper() or "IBGE" in k.upper():
        geojson_key = k
        break
if geojson_key is None:
    geojson_key = geojson_keys[0]

# Injetar atributos
for feature in geojson["features"]:
    try:
        cod = int(feature["properties"][geojson_key])
    except:
        cod = None
    if cod in map_dict:
        feature["properties"]["Extensao_km"] = float(map_dict[cod]["Extensao_km"])
        feature["properties"]["Qtd_RTVs"] = int(map_dict[cod]["Qtd_RTVs"])
    else:
        feature["properties"]["Extensao_km"] = 0
        feature["properties"]["Qtd_RTVs"] = 0

coluna_color = "Extensao_km" if criterio_mapa == "Extensão Total (km)" else "Qtd_RTVs"
max_val = max([f["properties"][coluna_color] for f in geojson["features"]]) or 1

for feature in geojson["features"]:
    val = feature["properties"][coluna_color]
    intensity = int((val / max_val) * 255)
    feature["properties"]["color_value"] = intensity

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
