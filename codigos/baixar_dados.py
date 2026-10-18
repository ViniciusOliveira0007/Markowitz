"""
Baixa os dois arquivos opcionais que o markowitz_liquido.py usa:

    dados/dados_limpos/selic_diaria.csv   colunas: data, valor   (% ao dia, série 11 do Banco Central)
    dados/dados_limpos/proventos.csv      colunas: data_ex, codneg, tipo, valor

Requisitos:  pip install yfinance        (precisa de internet)
Rode UMA vez e confira o resumo impresso no final (veja o que checar nos comentários).

LIMITAÇÕES DO YAHOO FINANCE (confira antes de confiar nos números):
  1. Não separa DIVIDENDO de JCP. Por isso cada pagamento de AÇÃO é dividido em duas linhas
     (DIVIDENDO e JCP) segundo FRACAO_JCP_*, que é uma SUPOSIÇÃO sua, não um dado.
  2. Não sei se o valor do JCP vem bruto ou líquido de IR. Confira um caso que você conheça.
  3. Os valores costumam vir ajustados por desdobramento. A tabela de rendimento no final serve
     para conferir: o rendimento de um ativo não deve dar um salto de 3x ou 10x entre anos.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PASTA_DADOS = BASE_DIR / "dados" / "dados_limpos"
ARQUIVO_COTAHIST = PASTA_DADOS / "cotahist_limpo.csv"
ARQUIVO_SELIC = PASTA_DADOS / "selic_diaria.csv"
ARQUIVO_PROVENTOS = PASTA_DADOS / "proventos.csv"
# Plano B: se a API do Banco Central falhar, exporte a série 11 em https://www3.bcb.gov.br/sgspub
# (CSV com separador ";" e decimal ",") e salve neste caminho. O script converte sozinho.
ARQUIVO_SELIC_MANUAL = BASE_DIR / "dados" / "selic_baixada_manualmente.csv"

DATA_INICIO = pd.Timestamp("2020-01-01")
DATA_FIM = pd.Timestamp("2025-12-31")

RENOMEAR_TICKERS = {"ELET3": "AXIA3", "EMBR3": "EMBJ3"}

# Cópia do que está no markowitz_liquido.py (mantenha os dois iguais)
CLASSE_ATIVO = {
    "AXIA3": "acao", "B3SA3": "acao", "EMBJ3": "acao", "ITUB4": "acao", "PETR4": "acao",
    "BOVA11": "etf", "DIVO11": "etf", "GOLD11": "etf", "IVVB11": "etf", "SMAL11": "etf",
    "HGLG11": "fii", "KNRI11": "fii", "MXRF11": "fii", "TGAR11": "fii", "XPML11": "fii",
}
EVENTOS_CORPORATIVOS = {
    ("B3SA3", "2021-05-17"): 3,
}

# Tickers do Yahoo (sufixo .SA). Ativos que mudaram de código: busca o novo e o antigo e junta.
TICKERS_YAHOO = {
    "AXIA3": ["AXIA3.SA", "ELET3.SA"],
    "EMBJ3": ["EMBJ3.SA", "EMBR3.SA"],
}

# Fração de cada pagamento de AÇÃO tratada como JCP (o resto vira dividendo isento).
# 0.5 é só um ponto de partida: ajuste por ticker se souber a proporção real.
FRACAO_JCP_PADRAO = 0.5
FRACAO_JCP_ACOES = {
    # "ITUB4": 0.8,
}


# ----------------------------------------------------------------------
# 1. Selic diária (Banco Central, série 11)
# ----------------------------------------------------------------------

def _pedir_json(url: str, tentativas: int = 4):
    """GET com tentativas e mensagens claras. A API do BC é instável (timeouts, corpo vazio)."""
    espera, ultimo_erro = 2, ""
    for n in range(1, tentativas + 1):
        status, corpo = None, ""
        try:
            pedido = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(pedido, timeout=60) as resposta:
                status = resposta.status
                corpo = resposta.read().decode("utf-8", errors="replace")
            if not corpo.strip():
                ultimo_erro = f"HTTP {status} com corpo VAZIO"
            else:
                return json.loads(corpo)
        except json.JSONDecodeError:
            ultimo_erro = f"HTTP {status}, mas a resposta não é JSON. Começa com: {corpo[:200]!r}"
        except urllib.error.HTTPError as erro:
            ultimo_erro = f"HTTP {erro.code}: {erro.read()[:200]!r}"
        except urllib.error.URLError as erro:
            ultimo_erro = f"falha de conexão: {erro.reason}"
        print(f"    tentativa {n}/{tentativas} falhou: {ultimo_erro}")
        if n < tentativas:
            time.sleep(espera)
            espera *= 2
    raise RuntimeError(ultimo_erro)


def baixar_serie_bcb(codigo: int, inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.DataFrame:
    """Baixa uma série do SGS ano a ano (janelas pequenas, abaixo do limite de 10 anos)."""
    pedacos = []
    for ano in range(inicio.year, fim.year + 1):
        ini_ano = max(inicio, pd.Timestamp(ano, 1, 1))
        fim_ano = min(fim, pd.Timestamp(ano, 12, 31))
        url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=json"
               f"&dataInicial={ini_ano:%d/%m/%Y}&dataFinal={fim_ano:%d/%m/%Y}")
        print(f"  {ano}...")
        dados = _pedir_json(url)
        if dados:
            pedacos.append(pd.DataFrame(dados))
    if not pedacos:
        raise RuntimeError("a API respondeu, mas sem nenhum dado")

    df = pd.concat(pedacos, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float)
    return df.drop_duplicates("data").sort_values("data").reset_index(drop=True)


def baixar_selic(inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.DataFrame:
    """Selic diária em % ao dia (série 11 do SGS)."""
    return baixar_serie_bcb(11, inicio, fim)


def converter_selic_manual(caminho: Path) -> pd.DataFrame:
    """Lê o CSV exportado do site do BC (sep ';', decimal ',') e devolve data/valor."""
    df = pd.read_csv(caminho, sep=";", decimal=",", encoding="latin-1")
    df = df.iloc[:, :2].copy()
    df.columns = ["data", "valor"]
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = df["valor"].astype(float)
    return df[(df["data"] >= DATA_INICIO) & (df["data"] <= DATA_FIM)].sort_values("data").reset_index(drop=True)


def obter_selic() -> pd.DataFrame | None:
    """Tenta a API; se falhar, usa o arquivo manual; se nada der certo, devolve None."""
    try:
        return baixar_selic(DATA_INICIO, DATA_FIM)
    except Exception as erro:
        print(f"\nNão consegui a Selic pela API: {erro}")
    if ARQUIVO_SELIC_MANUAL.exists():
        print(f"Usando o arquivo manual: {ARQUIVO_SELIC_MANUAL}")
        return converter_selic_manual(ARQUIVO_SELIC_MANUAL)
    print("\nPlano B: abra https://www3.bcb.gov.br/sgspub, exporte a série 11 (Selic diária) em CSV\n"
          f"e salve em {ARQUIVO_SELIC_MANUAL}. Depois rode este script de novo.\n"
          "Enquanto isso, o markowitz_liquido.py roda sem o Tesouro Selic.\n")
    return None


def resumir_selic(selic: pd.DataFrame) -> None:
    acumulada = (1 + selic["valor"] / 100).prod() - 1
    anos = (selic["data"].iloc[-1] - selic["data"].iloc[0]).days / 365
    print(f"Selic: {len(selic)} dias, de {selic['data'].iloc[0].date()} a {selic['data'].iloc[-1].date()}")
    print(f"  acumulada no período: {acumulada:.2%} (equivale a {(1 + acumulada) ** (1 / anos) - 1:.2%} ao ano)")
    print("  Confira: deve ficar entre ~2% (2020/21) e ~15% (2025) ao ano; "
          "um dia com valor 0 ou fora disso indica erro.\n")


# ----------------------------------------------------------------------
# 2. Proventos (Yahoo Finance via yfinance)
# ----------------------------------------------------------------------

def classificar(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Define o 'tipo' de cada provento segundo a classe do ativo."""
    classe = CLASSE_ATIVO.get(ticker, "acao")

    if classe == "fii":
        return df.assign(tipo="RENDIMENTO")

    if classe == "acao":
        fracao = FRACAO_JCP_ACOES.get(ticker, FRACAO_JCP_PADRAO)
        dividendo = df.assign(tipo="DIVIDENDO", valor=df["valor"] * (1 - fracao))
        jcp = df.assign(tipo="JCP", valor=df["valor"] * fracao)
        return pd.concat([dividendo[dividendo["valor"] > 0], jcp[jcp["valor"] > 0]])

    print(f"  AVISO: {ticker} é ETF e distribuiu proventos (incomum). Tratado como DIVIDENDO: revise.")
    return df.assign(tipo="DIVIDENDO")


def ler_tickers() -> list[str]:
    tickers = pd.read_csv(ARQUIVO_COTAHIST, usecols=["codneg"])["codneg"]
    return sorted(tickers.replace(RENOMEAR_TICKERS).unique())


def baixar_proventos(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf   # importado aqui para só exigir a instalação se chegar neste passo

    partes, falhas = [], []
    for ticker in tickers:
        brutos = []
        for codigo in TICKERS_YAHOO.get(ticker, [ticker + ".SA"]):
            try:
                serie = yf.Ticker(codigo).dividends
            except Exception as erro:               # rede, limite de acessos etc.
                falhas.append(f"{codigo}: {erro}")
                continue
            if serie is None or len(serie) == 0:    # ticker inexistente no Yahoo (ex.: código antigo)
                print(f"  ({codigo}: sem dados no Yahoo)")
                continue
            brutos.append(serie)
            time.sleep(1)                           # evita bloqueio por excesso de pedidos

        if not brutos:
            print(f"  {ticker}: nenhum provento encontrado")
            continue

        serie = pd.concat(brutos)
        datas = pd.DatetimeIndex(serie.index)
        if datas.tz is not None:
            datas = datas.tz_localize(None)

        df = pd.DataFrame({"data_ex": datas.normalize(), "codneg": ticker, "valor": serie.to_numpy()})
        df = df.drop_duplicates(subset=["data_ex", "valor"])
        df = df[(df["data_ex"] >= DATA_INICIO) & (df["data_ex"] <= DATA_FIM)]
        print(f"  {ticker}: {len(df)} pagamentos, total R$ {df['valor'].sum():.2f} por papel")
        partes.append(classificar(ticker, df))

    if falhas:
        print("\nFalhas de download:\n  " + "\n  ".join(falhas))
    if not partes:
        raise SystemExit("Nenhum provento foi baixado. Verifique a internet e a instalação do yfinance.")

    colunas = ["data_ex", "codneg", "tipo", "valor"]
    return (pd.concat(partes, ignore_index=True)[colunas]
            .sort_values(["codneg", "data_ex", "tipo"]).reset_index(drop=True))


def resumir_rendimento(prov: pd.DataFrame) -> None:
    """Rendimento anual (proventos do ano / preço médio do ano), em %, por ticker.

    Como conferir: FIIs costumam ficar entre 6% e 12% ao ano; ações entre 0% e 10%.
    Um ativo com rendimento 3x ou 10x maior/menor em alguns anos sugere provento sem ajuste
    de desdobramento (veja a B3SA3 antes e depois de 2021).
    """
    df = pd.read_csv(ARQUIVO_COTAHIST, parse_dates=["data_pregao"],
                     usecols=["data_pregao", "codneg", "preult"])
    df["codneg"] = df["codneg"].replace(RENOMEAR_TICKERS)
    precos = df.pivot(index="data_pregao", columns="codneg", values="preult").sort_index()

    for (ticker, data), fator in EVENTOS_CORPORATIVOS.items():
        if ticker in precos.columns:
            antes = precos.index < pd.Timestamp(data)
            precos.loc[antes, ticker] = precos.loc[antes, ticker] / fator

    preco_medio = precos.groupby(precos.index.year).mean()
    soma = (prov.assign(ano=prov["data_ex"].dt.year)
            .pivot_table(index="ano", columns="codneg", values="valor", aggfunc="sum"))
    rendimento = (100 * soma.div(preco_medio)).round(1)

    print("\nRendimento anual por ticker (%), proventos do ano / preço médio do ano:")
    print(rendimento.to_string())


def resumir_cobertura(prov: pd.DataFrame) -> None:
    """Primeiro e último provento de cada ticker. Se um FII/ação só começa depois do início da
    sua amostra, o retorno dele fica subestimado nos anos anteriores (falta renda, não é queda)."""
    cobertura = prov.groupby("codneg")["data_ex"].agg(primeiro="min", ultimo="max", pagamentos="size")
    print("\nCobertura dos proventos (primeiro e último pagamento por ticker):")
    print(cobertura.to_string())


# ----------------------------------------------------------------------
# 3. Execução
# ----------------------------------------------------------------------

def main() -> None:
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)

    print("Baixando Selic (Banco Central)...")
    selic = obter_selic()
    if selic is not None:
        selic.to_csv(ARQUIVO_SELIC, index=False)
        resumir_selic(selic)

    print("Baixando proventos (Yahoo Finance)...")
    prov = baixar_proventos(ler_tickers())
    prov.to_csv(ARQUIVO_PROVENTOS, index=False)
    prov.to_csv(PASTA_DADOS / "proventos_yahoo.csv", index=False)   # cópia só-Yahoo (usada pelo completar_fiis.py)
    resumir_rendimento(prov)
    resumir_cobertura(prov)

    print(f"\nArquivos salvos em: {PASTA_DADOS}")
    print("  selic_diaria.csv  e  proventos.csv")


if __name__ == "__main__":
    main()