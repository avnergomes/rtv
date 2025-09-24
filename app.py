import copy
from datetime import datetime
import json

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from pandas.api.types import is_categorical_dtype
from plotly.colors import sample_colorscale, sequential

st.set_page_config(page_title="Dashboard RTVs", layout="wide")

URL_SHEET = "https://docs.google.com/spreadsheets/d/1OcX3jxFqnEEKe7wqJVm_pl4QYz3foCNLM1Dnf9aYie4/export?format=xlsx"
STATUS_ORDER = ["Planejado", "Em Andamento", "Entregue"]


@st.cache_data(ttl=600)  # Atualiza a cada 10 minutos
def load_data():
    xls = pd.ExcelFile(URL_SHEET)
    df = pd.read_excel(xls, sheet_name="RTVs")
    municipios = pd.read_excel(xls, sheet_name="Municípios")
    return df, municipios


@st.cache_data
def load_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_number(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "-"
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def convert_df_to_csv(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def color_to_rgba(color: str, alpha: int = 200) -> list[int]:
    if color.startswith("rgb"):
        values = color[color.find("(") + 1 : color.find(")")].split(",")
        rgb = [int(float(v)) for v in values[:3]]
        return rgb + [alpha]
    hex_color = color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


def detect_geojson_key(geojson: dict) -> str:
    sample_feature = geojson.get("features", [{}])[0]
    properties = sample_feature.get("properties", {})
    for key in properties.keys():
        upper_key = key.upper()
        if "CD" in upper_key or "IBGE" in upper_key:
            return key
    return next(iter(properties.keys()), "")


df, municipios = load_data()
geojson_base = load_geojson("mun_PR.json")
GEOJSON_KEY = detect_geojson_key(geojson_base)

# ======================
# LIMPEZA DE DADOS
# ======================
df = df.dropna(how="all").copy()
df.columns = df.columns.str.strip()

if "Extensão (km)" in df.columns:
    df["Extensão (km)"] = (
        df["Extensão (km)"]
        .astype("string")
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["Extensão (km)"] = pd.to_numeric(df["Extensão (km)"], errors="coerce").fillna(0.0)

for col in ["Município", "Região", "STATUS", "Pavimento"]:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

if "STATUS" in df.columns:
    df["STATUS"] = df["STATUS"].str.title()
    extra_status = sorted(set(df["STATUS"].dropna()) - set(STATUS_ORDER))
    status_categories = STATUS_ORDER + [status for status in extra_status if status not in STATUS_ORDER]
    df["STATUS"] = pd.Categorical(df["STATUS"], categories=status_categories, ordered=True)

if "Região" in df.columns:
    df["Região"] = df["Região"].str.title()

if "Município" in df.columns:
    df["Município"] = df["Município"].str.title()

if "Pavimento" in df.columns:
    df["Pavimento"] = (
        df["Pavimento"]
        .fillna("Não informado")
        .replace("", "Não informado")
        .str.title()
    )

if "CodIBGE" in df.columns:
    df["CodIBGE"] = pd.to_numeric(df["CodIBGE"], errors="coerce").astype("Int64")

if "Previsão Entrega" in df.columns:
    df["Previsão Entrega"] = pd.to_datetime(
        df["Previsão Entrega"], errors="coerce", dayfirst=True
    )

if "Município" in df.columns:
    df = df[df["Município"].notna() & (df["Município"].str.len() > 0)]

if "Região" in df.columns:
    df = df[df["Região"].notna() & (df["Região"].str.len() > 0)]

# ======================
# SIDEBAR - LOGO + FILTROS
# ======================
st.sidebar.image("IDR_GOV_Seab_V_1.webp", use_container_width=True)
st.sidebar.markdown("## 🔍 Filtros")

if "STATUS" in df.columns and is_categorical_dtype(df["STATUS"]):
    status_options = [str(status) for status in df["STATUS"].cat.categories if status is not None]
else:
    status_options = sorted(df["STATUS"].dropna().unique()) if "STATUS" in df.columns else []

with st.sidebar.expander("Configurar filtros", expanded=True):
    regioes = st.multiselect(
        "Região",
        options=sorted(df["Região"].dropna().unique()) if "Região" in df.columns else [],
    )

    municipios_base = df[df["Região"].isin(regioes)] if regioes else df
    municipios_filtro = st.multiselect(
        "Município",
        options=sorted(municipios_base["Município"].dropna().unique()) if "Município" in municipios_base.columns else [],
    )

    status_filtro = st.multiselect(
        "Status",
        options=status_options,
        default=status_options,
    )

criterio_mapa = st.sidebar.radio(
    "Colorir mapa por:",
    ["Extensão Total (km)", "Quantidade de RTVs"],
    index=0,
)

# Aplicar filtros
df_filtered = df.copy()
if regioes:
    df_filtered = df_filtered[df_filtered["Região"].isin(regioes)]
if municipios_filtro:
    df_filtered = df_filtered[df_filtered["Município"].isin(municipios_filtro)]
if status_filtro:
    df_filtered = df_filtered[df_filtered["STATUS"].isin(status_filtro)]

st.sidebar.markdown("---")
st.sidebar.metric("RTVs selecionadas", len(df_filtered))
if not df_filtered.empty:
    st.sidebar.download_button(
        "⬇️ Baixar dados filtrados",
        data=convert_df_to_csv(df_filtered),
        file_name="rtvs_filtradas.csv",
        mime="text/csv",
    )
else:
    st.sidebar.caption("Ajuste os filtros para visualizar os dados.")

# ======================
# LAYOUT PRINCIPAL
# ======================
st.title("📊 Dashboard de Monitoramento de RTVs")
st.markdown("### Visão geral dos Relatórios Técnicos de Vistoria (RTVs)")
st.caption("Atualizado automaticamente a cada 10 minutos a partir da planilha oficial do IDR-PR.")

if df_filtered.empty:
    st.warning("Não há registros que atendam aos filtros selecionados.")
else:
    municipios_base_total = int(municipios.shape[0]) if isinstance(municipios, pd.DataFrame) else None
    overview_tab, distribuicao_tab, mapa_tab = st.tabs(
        ["Visão geral", "Distribuições", "Mapa e tabela"]
    )

    with overview_tab:
        st.subheader("Indicadores gerais")
        total_rtv = len(df_filtered)
        extensao_total = df_filtered["Extensão (km)"].sum()
        municipios_total = df_filtered["Município"].nunique()
        entregues_total = df_filtered[df_filtered["STATUS"] == "Entregue"].shape[0]
        percent_entregue = (entregues_total / total_rtv * 100) if total_rtv else 0
        extensao_media = df_filtered["Extensão (km)"].mean() if total_rtv else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 Total de RTVs", format_number(total_rtv))
        col2.metric("📏 Extensão total (km)", format_number(extensao_total, 2))
        if municipios_base_total:
            col3.metric(
                "🏘 Municípios atendidos",
                f"{format_number(municipios_total)} de {format_number(municipios_base_total)}",
            )
        else:
            col3.metric("🏘 Municípios atendidos", format_number(municipios_total))
        col4.metric("✅ % Entregue", f"{percent_entregue:.1f}%")

        col5, col6 = st.columns(2)
        col5.metric("📐 Extensão média por RTV (km)", format_number(extensao_media, 2))
        col6.metric("📦 RTVs entregues", format_number(entregues_total))

        st.markdown(
            f"**Avanço das entregas:** {format_number(entregues_total)} de {format_number(total_rtv)} RTVs concluídas."
        )
        st.progress(percent_entregue / 100 if total_rtv else 0.0)

        st.markdown("---")
        st.subheader("📈 Evolução das entregas")
        anos_disponiveis = (
            sorted(df_filtered["Previsão Entrega"].dropna().dt.year.unique())
            if "Previsão Entrega" in df_filtered.columns
            else []
        )

        if anos_disponiveis:
            default_year = anos_disponiveis[-1]
            col_config_ano, col_config_metric = st.columns([2, 2])
            with col_config_ano:
                ano_selecionado = st.selectbox(
                    "Ano da previsão de entrega",
                    options=anos_disponiveis,
                    index=anos_disponiveis.index(default_year),
                    key="ano_curva",
                )
            with col_config_metric:
                metrica_curva = st.selectbox(
                    "Métrica da curva S",
                    options=["Quantidade de RTVs", "Extensão (km)"],
                    index=0,
                    key="metrica_curva",
                )

            df_entregues = df_filtered[df_filtered["STATUS"] == "Entregue"].copy()
            if "Previsão Entrega" in df_entregues.columns:
                df_entregues = df_entregues[df_entregues["Previsão Entrega"].notna()]
                df_entregues = df_entregues[
                    df_entregues["Previsão Entrega"].dt.year == ano_selecionado
                ]
            else:
                df_entregues = pd.DataFrame()

            if not df_entregues.empty:
                calendario = pd.date_range(
                    start=datetime(ano_selecionado, 1, 1),
                    end=datetime(ano_selecionado, 12, 31),
                    freq="D",
                )
                if metrica_curva == "Quantidade de RTVs":
                    serie = df_entregues.groupby("Previsão Entrega").size()
                    hover_format = ":.0f"
                else:
                    serie = df_entregues.groupby("Previsão Entrega")["Extensão (km)"].sum()
                    hover_format = ":.2f"

                serie = serie.reindex(calendario, fill_value=0).reset_index()
                serie.columns = ["Previsão Entrega", "Valor diário"]
                serie["Acumulado"] = serie["Valor diário"].cumsum()

                hover_template = (
                    "<b>%{x|%d/%m/%Y}</b><br>"
                    f"Acumulado: %{{y{hover_format}}}<br>"
                    f"Valor diário: %{{customdata[0]{hover_format}}}<extra></extra>"
                )

                fig_curva = px.line(
                    serie,
                    x="Previsão Entrega",
                    y="Acumulado",
                    markers=True,
                    color_discrete_sequence=["#0F70B7"],
                    custom_data=["Valor diário"],
                )
                fig_curva.update_traces(line_shape="linear", fill="tozeroy", hovertemplate=hover_template)
                fig_curva.update_layout(
                    xaxis_title="Data",
                    yaxis_title=f"{metrica_curva} acumulada",
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_curva, use_container_width=True)
                st.caption("A curva considera apenas RTVs com status 'Entregue'.")
            else:
                st.info(
                    "Nenhuma entrega registrada para o ano selecionado nos filtros aplicados."
                )
        else:
            st.info("Os filtros atuais não possuem previsão de entrega associada.")

    with distribuicao_tab:
        st.subheader("📊 Distribuições")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### Situação das RTVs")
            if "STATUS" in df_filtered.columns:
                df_status = (
                    df_filtered[df_filtered["STATUS"].notna()]
                    .groupby("STATUS")
                    .size()
                    .reset_index(name="RTVs")
                )
                if not df_status.empty:
                    fig_status = px.bar(
                        df_status,
                        x="STATUS",
                        y="RTVs",
                        text_auto=True,
                        color="STATUS",
                        color_discrete_sequence=sequential.Blues,
                    )
                    fig_status.update_layout(
                        showlegend=False,
                        xaxis_title="",
                        yaxis_title="Quantidade de RTVs",
                        margin=dict(l=10, r=10, t=30, b=40),
                    )
                    fig_status.update_traces(
                        hovertemplate="<b>%{x}</b><br>RTVs: %{y}<extra></extra>"
                    )
                    st.plotly_chart(fig_status, use_container_width=True)
                else:
                    st.info("Sem dados de status para os filtros aplicados.")
            else:
                st.info("Base atual não contém informação de status.")

        with colB:
            st.markdown("#### Tipo de pavimento")
            if "Pavimento" in df_filtered.columns:
                df_pav = (
                    df_filtered["Pavimento"]
                    .fillna("Não informado")
                    .replace("", "Não informado")
                    .value_counts()
                    .reset_index()
                )
                df_pav.columns = ["Pavimento", "RTVs"]
                if not df_pav.empty:
                    fig_pav = px.pie(
                        df_pav,
                        names="Pavimento",
                        values="RTVs",
                        hole=0.45,
                        color="Pavimento",
                        color_discrete_sequence=sequential.Sunset,
                    )
                    fig_pav.update_traces(textposition="inside", textinfo="percent+label")
                    fig_pav.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_pav, use_container_width=True)
                else:
                    st.info("Sem dados de pavimento para os filtros aplicados.")
            else:
                st.info("Base atual não contém informação de pavimento.")

        st.markdown("#### RTVs por região")
        if "Região" in df_filtered.columns:
            df_regioes = (
                df_filtered[df_filtered["Região"].notna()]
                .groupby("Região")
                .agg(
                    RTVs=("STATUS", "count"),
                    Extensao=("Extensão (km)", "sum"),
                )
                .reset_index()
                .sort_values("RTVs", ascending=False)
            )
            if not df_regioes.empty:
                fig_reg = px.bar(
                    df_regioes,
                    x="RTVs",
                    y="Região",
                    orientation="h",
                    text_auto=True,
                    color="Região",
                    color_discrete_sequence=sequential.Tealgrn,
                    custom_data=["Extensao"],
                )
                fig_reg.update_traces(
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>RTVs: %{x}<br>Extensão: %{customdata[0]:.2f} km<extra></extra>",
                )
                fig_reg.update_layout(
                    showlegend=False,
                    xaxis_title="Quantidade de RTVs",
                    yaxis_title="",
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_reg, use_container_width=True)
            else:
                st.info("Sem registros de região nos filtros aplicados.")
        else:
            st.info("A base filtrada não contém a coluna de região.")

    with mapa_tab:
        st.subheader("🗺 Mapa de municípios com RTVs")
        map_data = (
            df_filtered.groupby(["CodIBGE", "Município"], dropna=False)
            .agg({"Extensão (km)": "sum", "STATUS": "count"})
            .reset_index()
            .rename(
                columns={"Extensão (km)": "Extensao_km", "STATUS": "Qtd_RTVs"}
            )
        )

        map_data = map_data.dropna(subset=["CodIBGE"])
        if not map_data.empty:
            map_data["CodIBGE"] = map_data["CodIBGE"].astype(int)
            map_data["Extensao_km"] = map_data["Extensao_km"].round(2)
            map_data["Qtd_RTVs"] = map_data["Qtd_RTVs"].astype(int)

        col_mapa, col_tabela = st.columns((3, 2), gap="large")

        with col_mapa:
            if map_data.empty:
                st.info("Ajuste os filtros para visualizar o mapa.")
            else:
                geojson = copy.deepcopy(geojson_base)
                map_dict = map_data.set_index("CodIBGE")[
                    ["Município", "Extensao_km", "Qtd_RTVs"]
                ].to_dict("index")

                coluna_color = (
                    "Extensao_km"
                    if criterio_mapa == "Extensão Total (km)"
                    else "Qtd_RTVs"
                )
                max_val = map_data[coluna_color].max()
                min_val = map_data[coluna_color].min()
                map_colorscale_name = (
                    "Blues" if coluna_color == "Extensao_km" else "YlOrRd"
                )

                for feature in geojson.get("features", []):
                    props = feature.setdefault("properties", {})
                    try:
                        raw_cod = props.get(GEOJSON_KEY)
                        cod = int(float(raw_cod))
                    except (TypeError, ValueError):
                        cod = None

                    info = map_dict.get(cod)
                    if info:
                        valor = info[coluna_color]
                        fraction = valor / max_val if max_val else 0
                        color_hex = sample_colorscale(map_colorscale_name, [fraction])[0]
                        props["fillColor"] = color_to_rgba(color_hex, alpha=200)
                        props["Município"] = info["Município"]
                        props["Extensao_km"] = info["Extensao_km"]
                        props["Qtd_RTVs"] = info["Qtd_RTVs"]
                        props["tooltip_text"] = (
                            f"{info['Município']}\nExtensão: {info['Extensao_km']:.2f} km\nRTVs: {info['Qtd_RTVs']}"
                        )
                    else:
                        nome_padrao = props.get("Município") or props.get("NM_MUN") or "Sem registro"
                        props["Extensao_km"] = 0.0
                        props["Qtd_RTVs"] = 0
                        props["fillColor"] = [220, 220, 220, 80]
                        props["tooltip_text"] = f"{nome_padrao}\nSem dados registrados"

                layer = pdk.Layer(
                    "GeoJsonLayer",
                    geojson,
                    pickable=True,
                    stroked=True,
                    filled=True,
                    get_fill_color="properties.fillColor",
                    get_line_color=[80, 80, 80],
                    line_width_min_pixels=1,
                )

                view_state = pdk.ViewState(latitude=-24.5, longitude=-51.5, zoom=6)

                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        tooltip={"text": "{tooltip_text}"},
                    )
                )

                start_color = sample_colorscale(map_colorscale_name, [0])[0]
                end_color = sample_colorscale(map_colorscale_name, [1])[0]
                legend_decimals = 0 if coluna_color == "Qtd_RTVs" else 2
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:12px;margin-top:10px;">
                        <span style=\"font-weight:600;\">Legenda ({criterio_mapa})</span>
                        <div style=\"flex-grow:1;height:12px;border-radius:4px;background:linear-gradient(90deg,{start_color},{end_color});\"></div>
                        <span style=\"font-size:0.8rem;\">{format_number(min_val, legend_decimals)}</span>
                        <span style=\"font-size:0.8rem;\">{format_number(max_val, legend_decimals)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_tabela:
            if map_data.empty:
                st.empty()
            else:
                st.markdown("#### Destaques por município")
                ranking = map_data.sort_values(
                    by=["Qtd_RTVs", "Extensao_km"], ascending=[False, False]
                )[["Município", "Qtd_RTVs", "Extensao_km"]]
                ranking = ranking.rename(
                    columns={"Qtd_RTVs": "RTVs", "Extensao_km": "Extensão (km)"}
                )
                st.dataframe(
                    ranking.head(20),
                    use_container_width=True,
                )
                st.download_button(
                    "⬇️ Exportar ranking por município",
                    data=convert_df_to_csv(ranking),
                    file_name="rtvs_por_municipio.csv",
                    mime="text/csv",
                )

                colunas_detalhe = [
                    col
                    for col in [
                        "Município",
                        "Região",
                        "STATUS",
                        "Extensão (km)",
                        "Previsão Entrega",
                        "Pavimento",
                    ]
                    if col in df_filtered.columns
                ]
                with st.expander("Ver dados detalhados das RTVs filtradas"):
                    st.dataframe(
                        df_filtered[colunas_detalhe].sort_values(by=colunas_detalhe[:2])
                        if colunas_detalhe[:2]
                        else df_filtered,
                        use_container_width=True,
                    )
