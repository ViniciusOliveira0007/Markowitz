"""Conserta o dados/proventos_b3_manual.csv: arruma a ordem das colunas das ações e deixa o\nB3SA3 anterior a mai/2021 com o valor NOMINAL (corrige se estiver x1/3 ou x9). Faz backup antes.\nPode rodar de novo sem estragar."""
import csv
import shutil
from pathlib import Path

import pandas as pd

ARQ = Path(__file__).resolve().parent.parent / "dados" / "proventos_b3_manual.csv"
FIIS = {"HGLG11", "KNRI11", "MXRF11", "TGAR11", "XPML11"}
COLUNAS = ["ticker", "data_com", "data_pagamento", "valor_por_cota", "competencia", "fonte", "tipo"]
RENOMEAR = {"codneg": "ticker", "valor_bruto": "valor_por_cota", "valor": "valor_por_cota"}

shutil.copy(ARQ, ARQ.with_name("proventos_b3_manual_backup.csv"))
with open(ARQ, newline="", encoding="utf-8-sig") as f:
    linhas = [l for l in csv.reader(f) if l]
cabecalho = [RENOMEAR.get(c.strip().lower(), c.strip().lower()) for c in linhas[0]]
print("cabeçalho que estava no arquivo:", cabecalho)

fiis, acoes = [], []
for r in linhas[1:]:
    if r[0].strip().lower() in ("ticker", "codneg"):
        continue                                   # cabeçalho repetido no meio do arquivo
    if r[0].strip().upper() in FIIS:
        fiis.append(dict(zip(cabecalho, r)))       # FIIs: seguem o cabeçalho do arquivo
    else:
        acoes.append(dict(zip(COLUNAS, r)))        # ações: ordem do CSV que eu montei

df = pd.concat([pd.DataFrame(fiis), pd.DataFrame(acoes)], ignore_index=True)
df.loc[df["ticker"].str.upper().isin(FIIS), "tipo"] = "Rendimento"

# B3SA3 antes do desdobramento: o montar espera o valor NOMINAL (o Status Invest vem dividido por 3).
# Dividendo de 2020-09-24 como referencia: ajustado = 0.2162475 | nominal = 0.6487425 | x3 duas vezes = 1.9462275
pre = (df["ticker"] == "B3SA3") & (df["data_com"] < "2021-05-14")
ref = df[pre & (df["data_com"] == "2020-09-24") & (df["tipo"] == "Dividendo")]["valor_por_cota"]
atual = float(ref.iloc[0]) if len(ref) == 1 else None
fator = None
if atual is not None and abs(atual - 0.2162475) < 1e-6:
    fator, msg = 3, "estava ajustado: multiplicado por 3 (agora nominal)."
elif atual is not None and abs(atual - 1.9462275) < 1e-6:
    fator, msg = 1 / 3, "estava multiplicado por 9: dividido por 3 (agora nominal)."
elif atual is not None and abs(atual - 0.6487425) < 1e-6:
    print("B3SA3 anterior a mai/2021: já está nominal, nada a fazer.")
else:
    print(f"ATENÇÃO: B3SA3 de 2020-09-24 (dividendo) = {atual}. Esperado 0.6487425. Veja à mão.")
if fator:
    df.loc[pre, "valor_por_cota"] = [f"{float(v) * fator:.8f}" for v in df.loc[pre, "valor_por_cota"]]
    print("B3SA3 anterior a mai/2021:", msg)

df[COLUNAS].to_csv(ARQ, index=False)
print("\nLinhas por ticker:")
print(df.groupby("ticker").size().to_string())