"""
Taxa livre de risco (Selic) para o Markowitz e o Sharpe.

Lê dados/selic_diaria.csv (gerado pelo baixar_selic.py) e calcula
o retorno acumulado e a taxa anualizada da Selic em um período.

Uso dentro de outro script (ex.: markowitz_calculo_liquido.py):
    from taxa_livre_risco import taxa_livre_risco_anual
    rf = taxa_livre_risco_anual("2020-01-01", "2025-12-31")
"""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ARQUIVO_SELIC = RAIZ / "dados" / "selic_diaria.csv"

DIAS_UTEIS_ANO = 252  # convenção do mercado brasileiro


def carregar_selic(caminho: Path = ARQUIVO_SELIC) -> pd.DataFrame:
    """Lê o CSV da Selic diária, com a coluna 'data' já como data."""
    return pd.read_csv(caminho, parse_dates=["data"])


def filtrar_periodo(selic: pd.DataFrame, inicio: str, fim: str) -> pd.DataFrame:
    """Mantém só as linhas entre inicio e fim (inclusive). Formato 'AAAA-MM-DD'."""
    filtro = (selic["data"] >= pd.Timestamp(inicio)) & (selic["data"] <= pd.Timestamp(fim))
    return selic.loc[filtro]


def retorno_acumulado(selic: pd.DataFrame) -> float:
    """Juros compostos: (1+r1) * (1+r2) * ... - 1."""
    return (1 + selic["selic_dia"]).prod() - 1


def taxa_anualizada(selic: pd.DataFrame) -> float:
    """Converte o acumulado de n dias úteis em taxa anual equivalente."""
    n = len(selic)
    if n == 0:
        raise ValueError("Nenhum dia de Selic no período informado.")
    return (1 + retorno_acumulado(selic)) ** (DIAS_UTEIS_ANO / n) - 1


def taxa_livre_risco_anual(inicio: str, fim: str) -> float:
    """Atalho: carrega, filtra o período e devolve a Selic anualizada (decimal)."""
    selic = filtrar_periodo(carregar_selic(), inicio, fim)
    return taxa_anualizada(selic)


def selic_anual_por_ano(selic: pd.DataFrame) -> pd.Series:
    """Selic acumulada em cada ano-calendário (útil para conferir com a história)."""
    por_ano = selic.groupby(selic["data"].dt.year)["selic_dia"]
    return por_ano.apply(lambda s: (1 + s).prod() - 1)


if __name__ == "__main__":
    selic = filtrar_periodo(carregar_selic(), "2020-01-01", "2025-12-31")

    print(f"Dias úteis no período : {len(selic)}")
    print(f"Retorno acumulado     : {retorno_acumulado(selic):.2%}")
    print(f"Taxa anualizada (rf)  : {taxa_anualizada(selic):.2%}")
    print("\nSelic acumulada por ano:")
    print(selic_anual_por_ano(selic).map("{:.2%}".format))