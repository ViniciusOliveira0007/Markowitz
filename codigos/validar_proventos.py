from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CSV = RAIZ / "dados" / "proventos_b3_manual.csv"

DATA_INICIO = "2020-01-01"
DATA_FIM = "2025-12-31"
TICKERS = ["HGLG11", "KNRI11", "MXRF11", "TGAR11", "XPML11"]

df = pd.read_csv(CAMINHO_CSV, parse_dates=["data_com", "data_pagamento"], encoding="utf-8-sig")

faltando = sorted(set(TICKERS) - set(df["ticker"]))
print("tickers faltando:", faltando or "nenhum")

for ticker, g in df.groupby("ticker"):
    g = g.sort_values("data_com")
    janela = g[(g["data_com"] >= DATA_INICIO) & (g["data_com"] <= DATA_FIM)]
    repetidos = g["data_com"].dt.to_period("M").duplicated().sum()
    print(f"{ticker}: {len(g)} linhas | 1a data com {g['data_com'].min().date()} | "
          f"janela: {len(janela)} linhas, soma {janela['valor_por_cota'].sum():.4f} | meses repetidos: {repetidos}")