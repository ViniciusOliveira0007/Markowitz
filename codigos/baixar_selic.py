from pathlib import Path
import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "dados" / "selic_diaria.csv"

DATA_INICIO = "01/01/2020"   # a API do BCB usa dia/mês/ano
DATA_FIM = "31/12/2025"


def baixar_selic(data_ini: str, data_fim: str) -> pd.DataFrame:
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
    params = {"formato": "json", "dataInicial": data_ini, "dataFinal": data_fim}
    resposta = requests.get(url, params=params, timeout=30)
    resposta.raise_for_status()                      # para aqui se o BCB devolver erro
    df = pd.DataFrame(resposta.json())               # lista de dicionários -> tabela
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["selic_dia"] = df["valor"].astype(float) / 100   # % ao dia -> decimal
    return df[["data", "selic_dia"]]


if __name__ == "__main__":
    selic = baixar_selic(DATA_INICIO, DATA_FIM)
    selic.to_csv(SAIDA, index=False)
    print(len(selic), selic["data"].min().date(), selic["data"].max().date())
    print(selic["selic_dia"].min(), selic["selic_dia"].max())