"""Funciones de extracción, limpieza y combinación para Ray of Hope.

El archivo original de IHME se conserva sin modificaciones. Este módulo
estandariza sus columnas, prepara los indicadores del Banco Mundial y combina
ambas fuentes por código ISO3 y año.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuración general del proyecto
# ---------------------------------------------------------------------------
START_YEAR = 2014
END_YEAR = 2023
MIN_YEARS_FOR_RANKING = 8

WB_BASE_URL = "https://api.worldbank.org/v2"
WB_INDICATORS = {
    "population": "SP.POP.TOTL",
    "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
}

# El GBD contiene categorías en portugués. Se utilizan variantes
# del portugués de Brasil y Portugal para ampliar las equivalencias.
try:
    import pycountry
    from babel import Locale

    _PT_TERRITORIES = [
        Locale.parse("pt_BR").territories,
        Locale.parse("pt").territories,
    ]

except ImportError:
    pycountry = None
    _PT_TERRITORIES = []


# Equivalencias puntuales para nombres del IHME que pueden diferir de Babel,
# del Banco Mundial o de los nombres oficiales.
MANUAL_ISO3: Dict[str, str] = {
    "Holanda": "NLD",
    "Países Bajos": "NLD",
    "Barém": "BHR",
    "Bielorrússia": "BLR",
    "Bolívia": "BOL",
    "Brunei": "BRN",
    "Cabo Verde": "CPV",
    "Cingapura": "SGP",
    "Congo": "COG",
    "Coreia do Norte": "PRK",
    "Coreia do Sul": "KOR",
    "Costa do Marfim": "CIV",
    "Estados Federados da Micronésia": "FSM",
    "Estados Unidos": "USA",
    "Fíji": "FJI",
    "Irã": "IRN",
    "Laos": "LAO",
    "Macedônia": "MKD",
    "Moldávia": "MDA",
    "Mianmar": "MMR",
    "Palestina": "PSE",
    "Quiribati": "KIR",
    "Quirguistão": "KGZ",
    "Rússia": "RUS",
    "Síria": "SYR",
    "Suazilândia": "SWZ",
    "Taiwan": "TWN",
    "Tanzânia": "TZA",
    "Tajiquistão": "TJK",
    "Tchéquia": "CZE",
    "Vietnã": "VNM",
    "Venezuela": "VEN",
    "Ilhas Virgens, EUA": "VIR",
    "República Democrática do Congo": "COD",
    "República Centro-Africana": "CAF",
    "República Tcheca": "CZE",
    "São Cristóvão e Névis": "KNA",
    "São Tomé e Príncipe": "STP",
    "São Vicente e Granadinas": "VCT",
}


def _normalizar_texto(valor: object) -> str:
    """Normaliza texto para comparaciones sin acentos ni diferencias de caja."""
    texto = str(valor).strip().casefold()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _crear_mapa_iso3() -> Dict[str, str]:
    """Crea un mapa de nombres de países/localidades a códigos ISO3."""
    mapa: Dict[str, str] = {}

    if pycountry is not None:

        # Nombres en portugués de Brasil y Portugal.
        for territorios in _PT_TERRITORIES:
            for iso2, nombre_pt in territorios.items():
                if len(iso2) == 2:
                    pais = pycountry.countries.get(alpha_2=iso2)

                    if pais:
                        mapa[_normalizar_texto(nombre_pt)] = pais.alpha_3

        # Nombres oficiales, comunes y códigos ISO.
        for pais in pycountry.countries:
            mapa[_normalizar_texto(pais.name)] = pais.alpha_3
            mapa[_normalizar_texto(pais.alpha_2)] = pais.alpha_3
            mapa[_normalizar_texto(pais.alpha_3)] = pais.alpha_3

            if hasattr(pais, "common_name"):
                mapa[_normalizar_texto(pais.common_name)] = pais.alpha_3

    # Equivalencias específicas del IHME.
    for nombre, iso3 in MANUAL_ISO3.items():
        mapa[_normalizar_texto(nombre)] = iso3

    return mapa

ISO3_BY_NAME = _crear_mapa_iso3()


def mapear_iso3(nombre: object) -> str | None:
    """Devuelve el código ISO3 para un nombre de localidad del IHME."""
    return ISO3_BY_NAME.get(_normalizar_texto(nombre))


def limpiar_ihme(ruta_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lee y limpia el CSV IHME; devuelve datos anchos y nombres sin ISO3.

    El archivo recibido utiliza nombres de columnas y categorías en portugués.
    Se conservan únicamente prevalencia, ambos sexos, todas las edades,
    ansiedad y trastornos depresivos. Se excluye trastorno depresivo mayor
    para no mezclar una subcategoría con la categoría amplia de depresión.

    Las tasas se mantienen como tasas por 100.000 habitantes; no se convierten
    a porcentajes ni se describen como ajustadas por edad.
    """
    df = pd.read_csv(ruta_csv)

    columnas_requeridas = {
        "location_name", "year", "cause_name", "measure_name",
        "metric_name", "age_name", "sex_name", "val", "lower", "upper",
    }
    faltantes = columnas_requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(
            "El CSV de IHME no contiene las columnas requeridas: "
            + ", ".join(sorted(faltantes))
        )

    # Estandarizar etiquetas originales del archivo descargado.
    df["measure_name"] = df["measure_name"].astype(str).str.strip().str.casefold()
    df["cause_name"] = df["cause_name"].astype(str).str.strip().str.casefold()
    df["metric_name"] = df["metric_name"].astype(str).str.strip().str.casefold()
    df["age_name"] = df["age_name"].astype(str).str.strip().str.casefold()
    df["sex_name"] = df["sex_name"].astype(str).str.strip().str.casefold()

    causas = {
        "transtornos de ansiedade": "anxiety",
        "transtornos depressivos": "depression",
    }
    metricas = {"número": "cases", "numero": "cases", "taxa": "rate"}

    df = df.loc[
        df["year"].between(START_YEAR, END_YEAR)
        & df["measure_name"].isin(["prevalência", "prevalencia"])
        & df["sex_name"].isin(["ambos", "both"])
        & df["age_name"].isin(["todas as idades", "all ages"])
        & df["cause_name"].isin(causas)
        & df["metric_name"].isin(metricas)
    ].copy()

    if df.empty:
        raise ValueError(
            "No hay registros después de aplicar los filtros. "
            "Comprueba las etiquetas de medida, causa, sexo y edad del CSV."
        )

    df["cause"] = df["cause_name"].map(causas)
    df["metric"] = df["metric_name"].map(metricas)
    df["iso3"] = df["location_name"].map(mapear_iso3)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["lower"] = pd.to_numeric(df["lower"], errors="coerce")
    df["upper"] = pd.to_numeric(df["upper"], errors="coerce")

    # Localidades sin equivalencia: se devuelven para revisión manual.
    sin_mapa = (
        df.loc[df["iso3"].isna(), ["location_name"]]
        .drop_duplicates()
        .sort_values("location_name")
        .reset_index(drop=True)
    )

    df = df.dropna(subset=["iso3", "year", "val"])
    df["year"] = df["year"].astype(int)

    # Una fila por localidad-país-año-causa-métrica antes del pivot.
    claves = ["iso3", "year", "cause", "metric"]
    duplicados = df.duplicated(claves, keep=False)
    if duplicados.any():
        ejemplos = df.loc[duplicados, claves + ["location_name"]].head(10)
        raise ValueError(
            "Se encontraron registros duplicados para país-año-causa-métrica.\n"
            + ejemplos.to_string(index=False)
        )

    # El nombre original se conserva como etiqueta legible del país.
    nombres = (
        df.groupby("iso3", as_index=False)["location_name"]
        .agg(lambda valores: valores.mode().iloc[0])
        .rename(columns={"location_name": "country"})
    )

    ancho = df.pivot(
        index=["iso3", "year"],
        columns=["cause", "metric"],
        values="val",
    )
    ancho.columns = [f"{causa}_{metrica}" for causa, metrica in ancho.columns]
    ancho = ancho.reset_index().merge(nombres, on="iso3", how="left")

    # Garantizar las cuatro columnas aunque una categoría falte en una descarga.
    for columna in [
        "anxiety_rate", "depression_rate", "anxiety_cases", "depression_cases"
    ]:
        if columna not in ancho.columns:
            ancho[columna] = pd.NA

    ancho = ancho.rename(
        columns={
            "anxiety_rate": "anxiety_rate_per_100k",
            "depression_rate": "depression_rate_per_100k",
        }
    )
    ancho = ancho[
        [
            "country", "iso3", "year", "anxiety_rate_per_100k",
            "depression_rate_per_100k", "anxiety_cases", "depression_cases",
        ]
    ].sort_values(["country", "year"]).reset_index(drop=True)

    return ancho, sin_mapa


def _solicitar_json(url: str, params: dict | None = None) -> list:
    """Realiza una petición GET a la API del Banco Mundial."""
    respuesta = requests.get(url, params=params, timeout=60)
    respuesta.raise_for_status()
    datos = respuesta.json()
    if not isinstance(datos, list) or len(datos) < 2 or datos[1] is None:
        return []
    return datos[1]


def obtener_paises_banco_mundial() -> pd.DataFrame:
    """Obtiene ISO3 y grupo de ingresos actual de la API del Banco Mundial."""
    registros = _solicitar_json(
        f"{WB_BASE_URL}/country",
        params={"format": "json", "per_page": 400},
    )
    paises = []
    for item in registros:
        iso3 = item.get("id")
        region = (item.get("region") or {}).get("value")
        income = (item.get("incomeLevel") or {}).get("value")
        # Excluir agregados regionales y económicos del Banco Mundial.
        if iso3 and len(iso3) == 3 and region and region != "Aggregates":
            paises.append(
                {
                    "iso3": iso3,
                    "income_group": income,
                }
            )
    return pd.DataFrame(paises).drop_duplicates("iso3")


def obtener_indicador_banco_mundial(
    codigo_indicador: str,
    nombre_columna: str,
    paises_iso3: Iterable[str],
) -> pd.DataFrame:
    """Descarga un indicador anual para 2014–2023 y filtra países válidos."""
    registros = _solicitar_json(
        f"{WB_BASE_URL}/country/all/indicator/{codigo_indicador}",
        params={
            "date": f"{START_YEAR}:{END_YEAR}",
            "format": "json",
            "per_page": 20000,
        },
    )
    validos = set(paises_iso3)
    filas = []
    for item in registros:
        iso3 = item.get("countryiso3code")
        if iso3 in validos:
            filas.append(
                {
                    "iso3": iso3,
                    "year": int(item["date"]),
                    nombre_columna: item.get("value"),
                }
            )
    return pd.DataFrame(filas, columns=["iso3", "year", nombre_columna])


def obtener_datos_banco_mundial() -> pd.DataFrame:
    """Combina población, PIB per cápita y grupo de ingresos actual."""
    paises = obtener_paises_banco_mundial()
    codigos = paises["iso3"].dropna().unique().tolist()

    poblacion = obtener_indicador_banco_mundial(
        WB_INDICATORS["population"], "population", codigos
    )
    pib = obtener_indicador_banco_mundial(
        WB_INDICATORS["gdp_per_capita_usd"], "gdp_per_capita_usd", codigos
    )

    datos = poblacion.merge(pib, on=["iso3", "year"], how="outer")
    datos = datos.merge(paises, on="iso3", how="left")
    return datos.sort_values(["iso3", "year"]).reset_index(drop=True)


def build_dataframes(ruta_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye los dataframes IHME, Banco Mundial y combinado país-año."""
    df_mental, localidades_sin_iso3 = limpiar_ihme(ruta_csv)
    df_world_bank = obtener_datos_banco_mundial()

    df_decada = df_mental.merge(
        df_world_bank,
        on=["iso3", "year"],
        how="left",
        validate="one_to_one",
    )

    # La columna se conserva como atributo para facilitar la auditoría.
    df_decada.attrs["localidades_sin_iso3"] = localidades_sin_iso3
    return df_mental, df_world_bank, df_decada


def rank_top10(
    df_decada: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calcula rankings por tasa media anual (por 100.000 habitantes).

    El ranking exige al menos MIN_YEARS_FOR_RANKING años disponibles. Se
    excluye la categoría de trastorno depresivo mayor en la limpieza inicial.
    """
    resumen = df_decada.groupby(["country", "iso3"], as_index=False).agg(
        anxiety_years_available=("anxiety_rate_per_100k", "count"),
        anxiety_mean_rate_per_100k=("anxiety_rate_per_100k", "mean"),
        anxiety_cases_mean=("anxiety_cases", "mean"),
        depression_years_available=("depression_rate_per_100k", "count"),
        depression_mean_rate_per_100k=("depression_rate_per_100k", "mean"),
        depression_cases_mean=("depression_cases", "mean"),
        gdp_per_capita_latest_year_usd=("gdp_per_capita_usd", "last"),
        current_world_bank_income_group=("income_group", "last"),
    )

    top10_ansiedad = (
        resumen.loc[resumen["anxiety_years_available"] >= MIN_YEARS_FOR_RANKING]
        .sort_values("anxiety_mean_rate_per_100k", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10_depresion = (
        resumen.loc[resumen["depression_years_available"] >= MIN_YEARS_FOR_RANKING]
        .sort_values("depression_mean_rate_per_100k", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return resumen, top10_ansiedad, top10_depresion
