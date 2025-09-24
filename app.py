import copy
from datetime import datetime
import json
import re

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from plotly.colors import sample_colorscale, sequential
from pydeck.data_utils import viewport_helpers

st.set_page_config(page_title="Dashboard RTVs", layout="wide")

URL_SHEET = "https://docs.google.com/spreadsheets/d/1OcX3jxFqnEEKe7wqJVm_pl4QYz3foCNLM1Dnf9aYie4/export?format=xlsx"
STATUS_ORDER = ["Planejado", "Em Andamento", "Entregue"]

DECIMAL_CANDIDATE_PATTERN = re.compile(
    r"""
    -?\s*(?:R\$)?\s*\d{1,3}(?:\.\d{3})*(?:,\d+)?\s*(?:km|m2|m²|m|ha|%)?\s*
    |
    -?\s*(?:R\$)?\s*\d+(?:,\d+)?\s*(?:km|m2|m²|m|ha|%)?\s*
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
TRAILING_UNIT_PATTERN = re.compile(r"(?i)(km|m2|m²|m|ha|%)$")
CURRENCY_PATTERN = re.compile(r"(?i)r\$")
THOUSANDS_SEPARATOR_PATTERN = re.compile(r"(?<=\d)\.(?=\d{3}(?:\D|$))")
NUMERIC_HINT_KEYWORDS = (
    "extens",
    "km",
    "ha",
    "valor",
    "custo",
    "invest",
    "qtd",
    "quant",
    "area",
    "metros",
    "percent",
    "popul",
    "indice",
)


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


def standardize_decimal_columns(
    dataframe: pd.DataFrame, decimal_sep: str = ","
) -> pd.DataFrame:
    """Convert object columns that contain decimal strings with comma separators."""
    if dataframe is None:
        return dataframe

    df_local = dataframe.copy()
    candidate_columns = df_local.select_dtypes(include=["object", "string"]).columns

    if not len(candidate_columns):
        return df_local

    for column in candidate_columns:
        series = df_local[column].astype("string").str.strip()
        series = series.replace("", pd.NA)
        sample = series.dropna()
        if sample.empty:
            continue

        matches = sample.str.fullmatch(DECIMAL_CANDIDATE_PATTERN)
        match_ratio = matches.mean() if len(matches) else 0.0
        column_lower = column.lower()
        threshold = 0.6
        if any(keyword in column_lower for keyword in NUMERIC_HINT_KEYWORDS):
            threshold = 0.3
        if pd.isna(match_ratio) or match_ratio < threshold:
            continue

        normalized = series.str.replace("\u00a0", "", regex=False)
        normalized = normalized.str.replace(CURRENCY_PATTERN, "", regex=True)
        normalized = normalized.str.replace(TRAILING_UNIT_PATTERN, "", regex=True)
        normalized = normalized.str.replace(" ", "", regex=False)
        normalized = normalized.str.replace(THOUSANDS_SEPARATOR_PATTERN, "", regex=True)
        normalized = normalized.str.replace(decimal_sep, ".", regex=False)

        converted = pd.to_numeric(normalized, errors="coerce")
        if converted.notna().any():
            df_local[column] = converted

    return df_local


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


def extract_geojson_code(feature: dict, key: str):
    properties = feature.get("properties", {})
    raw_code = properties.get(key) if key else None
    if raw_code is None:
        for candidate_value in properties.values():
            try:
                return int(float(candidate_value)), candidate_value
            except (TypeError, ValueError):
                continue
        return None, None
    try:
        return int(float(raw_code)), raw_code
    except (TypeError, ValueError):
        return None, raw_code


def build_geojson_lookup(geojson: dict, key: str) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        code, raw_code = extract_geojson_code(feature, key)
        if code is None:
            continue

        name = (
            properties.get("Município")
            or properties.get("NM_MUN")
            or properties.get("NM_MUNICIP")
            or properties.get("NOME_MUNI")
            or properties.get("NM_NN")
            or str(code)
        )
        lookup[code] = str(name).strip()
    return lookup


def iter_lon_lat_pairs(values):
    if isinstance(values, (list, tuple)):
        if len(values) >= 2 and all(isinstance(v, (int, float)) for v in values[:2]):
            yield float(values[0]), float(values[1])
        else:
            for item in values:
                yield from iter_lon_lat_pairs(item)


def compute_bounds_from_coordinates(coordinates):
    min_lon = min_lat = max_lon = max_lat = None
    for lon, lat in iter_lon_lat_pairs(coordinates):
        if min_lon is None:
            min_lon = max_lon = lon
            min_lat = max_lat = lat
            continue
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
    if min_lon is None:
        return None
    return (min_lon, min_lat, max_lon, max_lat)


def merge_bounds(bounds_a, bounds_b):
    if bounds_a is None:
        return bounds_b
    if bounds_b is None:
        return bounds_a
    return (
        min(bounds_a[0], bounds_b[0]),
        min(bounds_a[1], bounds_b[1]),
        max(bounds_a[2], bounds_b[2]),
        max(bounds_a[3], bounds_b[3]),
    )


def build_geojson_bounds_lookup(geojson: dict, key: str):
    bounds_lookup: dict[int, tuple[float, float, float, float]] = {}
    overall_bounds = None
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        bounds = compute_bounds_from_coordinates(geometry.get("coordinates"))
        overall_bounds = merge_bounds(overall_bounds, bounds)
        if bounds is None:
            continue
        code, _ = extract_geojson_code(feature, key)
        if code is None or code in bounds_lookup:
            continue
        bounds_lookup[code] = bounds
    return bounds_lookup, overall_bounds


def bounds_to_points(bounds):
    if not bounds:
        return []
    min_lon, min_lat, max_lon, max_lat = bounds
    if min_lon == max_lon and min_lat == max_lat:
        return [[min_lon, min_lat]]
    return [
        [min_lon, min_lat],
        [min_lon, max_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
    ]


df, municipios = load_data()
geojson_base = load_geojson("mun_PR.json")
GEOJSON_KEY = detect_geojson_key(geojson_base)
MUNICIPIOS_LOOKUP = build_geojson_lookup(geojson_base, GEOJSON_KEY)
GEOJSON_BOUNDS_LOOKUP, GEOJSON_TOTAL_BOUNDS = build_geojson_bounds_lookup(
    geojson_base, GEOJSON_KEY
)
TOTAL_MUNICIPIOS_PR = len(MUNICIPIOS_LOOKUP)

# ======================
# LIMPEZA DE DADOS
# ======================
df = df.dropna(how="all").copy()
df.columns = df.columns.str.strip()
df = standardize_decimal_columns(df)

if isinstance(municipios, pd.DataFrame):
    municipios = municipios.dropna(how="all").copy()
    municipios.columns = municipios.columns.str.strip()
    municipios = standardize_decimal_columns(municipios)

if "Extensão (km)" in df.columns:
    if not is_numeric_dtype(df["Extensão (km)"]):
        df["Extensão (km)"] = pd.to_numeric(
            df["Extensão (km)"]
            .astype("string")
            .str.replace("\u00a0", "", regex=False)
            .str.replace(TRAILING_UNIT_PATTERN, "", regex=True)
            .str.replace(" ", "", regex=False)
            .str.replace(THOUSANDS_SEPARATOR_PATTERN, "", regex=True)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )
    df["Extensão (km)"] = df["Extensão (km)"].fillna(0.0)

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
rtvs_selecionadas = len(df_filtered)
municipios_filtrados = (
    df_filtered["Município"].nunique()
    if "Município" in df_filtered.columns
    else 0
)

st.sidebar.metric("RTVs selecionadas", format_number(rtvs_selecionadas))
if TOTAL_MUNICIPIOS_PR:
    cobertura_atual = (
        municipios_filtrados / TOTAL_MUNICIPIOS_PR * 100
        if TOTAL_MUNICIPIOS_PR
        else 0
    )
    st.sidebar.metric(
        "Municípios com RTVs (filtro)",
        f"{format_number(municipios_filtrados)} de {format_number(TOTAL_MUNICIPIOS_PR)}",
        delta=f"{cobertura_atual:.1f}% do estado",
    )
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
st.caption("Atualizado automaticamente a cada 10 minutos a partir da planilha oficial do IDR-Paraná.")

if df_filtered.empty:
    st.warning("Não há registros que atendam aos filtros selecionados.")
else:
    municipios_base_total = (
        TOTAL_MUNICIPIOS_PR
        or (
            int(municipios.shape[0])
            if isinstance(municipios, pd.DataFrame)
            else None
        )
    )
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
            cobertura_pct = (
                municipios_total / municipios_base_total * 100
                if municipios_base_total
                else 0
            )
            col3.metric(
                "🏘 Municípios atendidos",
                f"{format_number(municipios_total)} de {format_number(municipios_base_total)}",
                delta=f"Cobertura: {cobertura_pct:.1f}%",
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

        municipios_sem_registro: list[str] = []
        if TOTAL_MUNICIPIOS_PR:
            registrados = set(map_data["CodIBGE"].tolist()) if not map_data.empty else set()
            municipios_sem_registro = sorted(
                {
                    MUNICIPIOS_LOOKUP[codigo]
                    for codigo in MUNICIPIOS_LOOKUP.keys()
                    if codigo not in registrados and MUNICIPIOS_LOOKUP.get(codigo)
                }
            )

        col_mapa, col_tabela = st.columns((3, 2), gap="large")

        with col_mapa:
            if map_data.empty:
                st.info("Ajuste os filtros para visualizar o mapa.")
            else:
                geojson = copy.deepcopy(geojson_base)
                map_dict = map_data.set_index("CodIBGE")[
                    ["Município", "Extensao_km", "Qtd_RTVs"]
                ].to_dict("index")
                selected_codes = set(map_data["CodIBGE"].tolist())
                combined_bounds = None
                for code in selected_codes:
                    code_bounds = GEOJSON_BOUNDS_LOOKUP.get(code)
                    if code_bounds is None:
                        continue
                    combined_bounds = merge_bounds(combined_bounds, code_bounds)
                if combined_bounds is None:
                    combined_bounds = GEOJSON_TOTAL_BOUNDS

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
                        nome_padrao = (
                            MUNICIPIOS_LOOKUP.get(cod)
                            or props.get("Município")
                            or props.get("NM_MUN")
                            or "Sem registro"
                        )
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
                view_points = bounds_to_points(combined_bounds)
                if view_points:
                    computed_view = viewport_helpers.compute_view(view_points)
                    zoom_value = computed_view.zoom
                    if zoom_value is not None:
                        max_zoom = 11.0
                        min_zoom = 5.5 if len(selected_codes) > 3 else 6.0
                        computed_view.zoom = max(
                            min(zoom_value, max_zoom),
                            min_zoom,
                        )
                    computed_view.pitch = 0
                    computed_view.bearing = 0
                    view_state = computed_view

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

                if municipios_sem_registro:
                    st.caption(
                        f"{len(municipios_sem_registro)} municípios do Paraná não possuem RTV registrada nos filtros atuais."
                    )
                    with st.expander("Municípios sem RTV registrada"):
                        st.dataframe(
                            pd.DataFrame(
                                {"Município": municipios_sem_registro}
                            ),
                            use_container_width=True,
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
