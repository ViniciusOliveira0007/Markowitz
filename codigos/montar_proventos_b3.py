"""
Monta o proventos.csv a partir dos proventos que VOCÊ copiou da B3 / Status Invest (valores BRUTOS).

Entrada:  dados/proventos_b3_manual.csv   colunas: codneg, tipo, data_com, valor_bruto
            codneg      ticker (ex.: PETR4). Pode usar o código antigo (ELET3): ele é renomeado.
                        (o cabeçalho "ticker" também é aceito)
            tipo        o que a B3 mostrar: Dividendo, JCP (ou "Jrs Cap Proprio"), Rendimento
            data_com    último dia COM direito, em 31/03/2022 ou 2022-03-31
            valor_bruto R$ por ação/cota, BRUTO (JCP antes do IR). Ponto ou vírgula decimal.
                        (o cabeçalho "valor_por_cota" também é aceito)
            data_pagamento (opcional) vai junto para o proventos.csv, porque o IR do JCP
                        depende da data de PAGAMENTO (15% até 2025, 17,5% a partir de 01/01/2026)
Saída:    dados/dados_limpos/proventos.csv  colunas: data_ex, codneg, tipo, valor, data_pagamento
          (o que o markowitz_liquido.py lê)

O que o script faz por você:
  * data ex = primeiro pregão DEPOIS da data com (regra da B3)
  * ajusta pelos desdobramentos/grupamentos/bonificações de EVENTOS_CORPORATIVOS (a B3 mostra o
    valor da época; o proventos.csv final fica na mesma escala dos preços ajustados, então
    PROVENTOS_JA_AJUSTADOS = True no markowitz_liquido.py continua certo).
    ATENÇÃO: os valores do CSV precisam estar NOMINAIS (da época). O Status Invest já devolve
    ajustado quando mostra o ícone vermelho de exclamação (ex.: B3SA3 antes de mai/2021).
  * confere tipo x classe do ativo, repetições, e rendimento fora do normal (erro de digitação)
  * "Rend. Tributado" de AÇÃO é ignorado com aviso (não é dividendo nem JCP)
  * compara os totais por ticker e ano com o Yahoo (proventos_yahoo.csv), se existir

Primeira execução: cria o arquivo de entrada só com o cabeçalho. Preencha e rode de novo.
É seguro repetir: ele sempre lê o arquivo manual do zero e sobrescreve o proventos.csv.
Não rode o baixar_dados.py depois, ou ele devolve o proventos.csv do Yahoo (rode este de novo).

Precisa estar na mesma pasta do baixar_dados.py e do completar_fiis.py (reaproveita funções deles).
"""

import shutil
import unicodedata

import pandas as pd

from baixar_dados import (ARQUIVO_COTAHIST, ARQUIVO_PROVENTOS, BASE_DIR, CLASSE_ATIVO,
                          EVENTOS_CORPORATIVOS, PASTA_DADOS, RENOMEAR_TICKERS,
                          resumir_cobertura, resumir_rendimento)
from completar_fiis import datas_de_pregao, proximo_pregao

ARQUIVO_B3 = BASE_DIR / "dados" / "proventos_b3_manual.csv"
ARQUIVO_YAHOO = PASTA_DADOS / "proventos_yahoo.csv"
ARQUIVO_ANTES_B3 = PASTA_DADOS / "proventos_antes_b3.csv"

COLUNAS = ["codneg", "tipo", "data_com", "valor_bruto"]
COLUNAS_BASE = ["data_ex", "codneg", "tipo", "valor"]          # o que baixar_dados.py espera
COLUNAS_SAIDA = COLUNAS_BASE + ["data_pagamento"]

# nomes de tipo aceitos (sem acento, maiúsculo, "_" vira espaço) -> tipo usado pelo modelo
TIPOS = {
    "DIVIDENDO": "DIVIDENDO", "DIVIDENDOS": "DIVIDENDO",
    "JCP": "JCP", "JRS CAP PROPRIO": "JCP",
    "JUROS SOBRE CAPITAL PROPRIO": "JCP", "JUROS S/ CAPITAL PROPRIO": "JCP",
    "RENDIMENTO": "RENDIMENTO", "REND TRIBUTADO": "RENDIMENTO", "RENDIMENTO TRIBUTADO": "RENDIMENTO",
    "AMORTIZACAO": "RENDIMENTO",
}
TIPOS_POR_CLASSE = {"acao": {"DIVIDENDO", "JCP"}, "fii": {"RENDIMENTO"}}   # ETF: aceita qualquer, com aviso

RENDIMENTO_MAX_ACAO = 0.15      # um único pagamento de ação acima de 15% do preço: confira
RENDIMENTO_MAX_FII = 0.03       # um pagamento de FII acima de 3% do preço: confira
RENDIMENTO_MIN_FII = 0.002      # e abaixo de 0,2%: confira
DIFERENCA_MAX_YAHOO = 0.05      # só mostra ticker/ano que diverge mais de 5% do Yahoo


def normalizar_tipo(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return sem_acento.upper().replace("_", " ").strip()


def ler_datas(serie: pd.Series) -> pd.Series:
    """Aceita 2022-03-31 e 31/03/2022. NÃO usa dayfirst: ele inverte mês e dia nas datas ISO."""
    texto = serie.astype("string").str.strip()
    iso = pd.to_datetime(texto, format="%Y-%m-%d", errors="coerce")
    br = pd.to_datetime(texto, format="%d/%m/%Y", errors="coerce")
    return iso.fillna(br)


# ----------------------------------------------------------------------
# Leitura e validação
# ----------------------------------------------------------------------

def criar_modelo_vazio() -> None:
    ARQUIVO_B3.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=COLUNAS).to_csv(ARQUIVO_B3, index=False)
    print(f"Arquivo criado (só o cabeçalho): {ARQUIVO_B3}\n")
    print("Preencha uma linha por provento (dividendo e JCP da mesma data são duas linhas) e rode de novo.")
    print("Período: a partir de dez/2020 (a sua amostra começa em 15/12/2020).")


def ler_b3() -> pd.DataFrame:
    df = pd.read_csv(ARQUIVO_B3, sep=None, engine="python", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"ticker": "codneg",
                            "valor_por_cota": "valor_bruto",
                            "valor": "valor_bruto"})
    if "tipo" not in df.columns:
        df["tipo"] = pd.NA
    df["tipo"] = df["tipo"].fillna("Rendimento")   # sem tipo = FII


    faltam = set(COLUNAS) - set(df.columns)
    if faltam:
        raise SystemExit(f"Faltam colunas no arquivo da B3: {sorted(faltam)}")
    df = df.dropna(how="all").copy()

    df["codneg"] = df["codneg"].str.strip().str.upper().replace(RENOMEAR_TICKERS)
    texto_tipo = df["tipo"].fillna("").map(normalizar_tipo)
    df["tipo"] = texto_tipo.map(TIPOS)
    desconhecidos = sorted(set(texto_tipo[df["tipo"].isna()]))
    if desconhecidos:
        raise SystemExit(f"Tipo(s) de provento desconhecido(s): {desconhecidos}. "
                         "Aceitos: Dividendo, JCP, Rendimento (me avise se a B3 usar outro nome).")
    if (texto_tipo == "AMORTIZACAO").any():
        print("AVISO: há 'Amortização'. Tratada como RENDIMENTO (devolução de capital, isenta p/ PF).\n")

    valor = (df["valor_bruto"].fillna("").str.replace("R$", "", regex=False)
             .str.replace(" ", "", regex=False).str.replace(",", ".", regex=False))
    df["valor_bruto"] = pd.to_numeric(valor, errors="coerce")
    df["data_com"] = ler_datas(df["data_com"])
    if "data_pagamento" in df.columns:
        df["data_pagamento"] = ler_datas(df["data_pagamento"])
    else:
        df["data_pagamento"] = pd.NaT

    invalidas = df[df["valor_bruto"].isna() | df["data_com"].isna() | (df["valor_bruto"] <= 0)]
    if len(invalidas):
        print(f"AVISO: {len(invalidas)} linha(s) com data ou valor inválido foram ignoradas "
              f"(linhas do arquivo: {', '.join(str(i + 2) for i in invalidas.index[:15])}"
              f"{'...' if len(invalidas) > 15 else ''}).\n")
        df = df.drop(invalidas.index)
    return df


def validar_classes(df: pd.DataFrame) -> pd.DataFrame:
    desconhecidos = sorted(set(df["codneg"]) - set(CLASSE_ATIVO))
    if desconhecidos:
        raise SystemExit(f"Ticker(s) fora do CLASSE_ATIVO do baixar_dados.py: {desconhecidos}")

    classe = df["codneg"].map(CLASSE_ATIVO)
    esperado = classe.map(lambda c: TIPOS_POR_CLASSE.get(c))
    incompativel = [t not in e if e is not None else False for t, e in zip(df["tipo"], esperado)]
    erradas = df[pd.Series(incompativel, index=df.index)]
    if len(erradas):
        print(f"AVISO: {len(erradas)} linha(s) com tipo incompatível com a classe do ativo foram ignoradas "
              "(ex.: dividendo em FII, ou 'Rend. Tributado' em ação). Primeiras:")
        print(erradas.head(8).to_string(), "\n")
        df = df.drop(erradas.index)
    if (df["codneg"].map(CLASSE_ATIVO) == "etf").any():
        print("AVISO: há proventos de ETF (incomum, ETFs costumam acumular). Revise.\n")
    return df


def remover_repetidas(df: pd.DataFrame) -> pd.DataFrame:
    """Descarta só linhas IDÊNTICAS. Parcelas do mesmo provento (mesma data com, datas de
    pagamento diferentes) são pagamentos reais e ficam."""
    chave = [c for c in ["codneg", "tipo", "data_com", "data_pagamento", "valor_bruto"] if c in df.columns]
    repetidas = df.duplicated(chave, keep="first")
    if repetidas.any():
        print(f"AVISO: {int(repetidas.sum())} linha(s) idêntica(s) (mesmo ticker, tipo, data com, "
              "pagamento e valor) descartada(s).\n")
    return df[~repetidas]


# ----------------------------------------------------------------------
# Datas, conferências e ajustes
# ----------------------------------------------------------------------

def preencher_data_ex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(data_ex=proximo_pregao(df["data_com"], datas_de_pregao()))
    fora = df[df["data_ex"].isna()]
    if len(fora):
        print(f"AVISO: {len(fora)} linha(s) com data com fora do período dos preços foram ignoradas.\n")
    return df.dropna(subset=["data_ex"])


def conferir_rendimento_por_evento(df: pd.DataFrame) -> None:
    """Valor nominal do provento / preço nominal na data ex. Pega erro de digitação."""
    cotacoes = pd.read_csv(ARQUIVO_COTAHIST, parse_dates=["data_pregao"],
                           usecols=["data_pregao", "codneg", "preult"])
    cotacoes["codneg"] = cotacoes["codneg"].replace(RENOMEAR_TICKERS)
    precos = cotacoes.pivot(index="data_pregao", columns="codneg", values="preult")

    suspeitos = []
    for linha in df.itertuples():
        if linha.codneg not in precos.columns or linha.data_ex not in precos.index:
            continue
        rendimento = linha.valor_bruto / precos.loc[linha.data_ex, linha.codneg]
        classe = CLASSE_ATIVO[linha.codneg]
        fora = ((classe == "acao" and rendimento > RENDIMENTO_MAX_ACAO)
                or (classe == "fii" and not RENDIMENTO_MIN_FII <= rendimento <= RENDIMENTO_MAX_FII))
        if fora:
            suspeitos.append(f"  {linha.codneg} {linha.tipo} data com {linha.data_com.date()}: "
                             f"R$ {linha.valor_bruto:.4f} = {rendimento:.2%} do preço")
    if suspeitos:
        print("ATENÇÃO: valores fora do normal (confira a digitação; pode ser pagamento extraordinário real):")
        print("\n".join(suspeitos), "\n")


def ajustar_por_eventos(df: pd.DataFrame) -> pd.DataFrame:
    """Divide os proventos ANTERIORES a cada desdobramento/grupamento/bonificação pelo fator
    (mesma escala dos preços ajustados). Os valores do CSV têm que estar NOMINAIS."""
    df = df.copy()
    for (ticker, data), fator in EVENTOS_CORPORATIVOS.items():
        antes = (df["codneg"] == ticker) & (df["data_ex"] < pd.Timestamp(data))
        df.loc[antes, "valor"] = df.loc[antes, "valor"] / fator
    return df


def comparar_com_yahoo(final: pd.DataFrame) -> None:
    if not ARQUIVO_YAHOO.exists():
        return
    yahoo = pd.read_csv(ARQUIVO_YAHOO, parse_dates=["data_ex"])

    def total_por_ano(df):
        return df.assign(ano=df["data_ex"].dt.year).groupby(["codneg", "ano"])["valor"].sum()

    tabela = pd.DataFrame({"b3": total_por_ano(final), "yahoo": total_por_ano(yahoo)})
    tabela = tabela[tabela.index.get_level_values("ano") >= 2021]      # 2020 só entra parcialmente na amostra
    tabela["diferenca"] = tabela["b3"] / tabela["yahoo"] - 1
    divergentes = tabela[(tabela["diferenca"].abs() > DIFERENCA_MAX_YAHOO) | tabela["yahoo"].isna() | tabela["b3"].isna()]

    print("Comparação B3 x Yahoo (total do ano por ticker; só aparece quem diverge mais de "
          f"{DIFERENCA_MAX_YAHOO:.0%} ou existe só em uma fonte):")
    print("  Obs.: o Yahoo não separa dividendo de JCP, então a comparação é do TOTAL bruto.")
    print(divergentes.round(4).to_string() if len(divergentes) else "  nenhuma divergência relevante.", "\n")


# ----------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------

def main() -> None:
    if not ARQUIVO_B3.exists():
        criar_modelo_vazio()
        return

    df = remover_repetidas(validar_classes(ler_b3()))
    if df.empty:
        print("Nenhuma linha válida em proventos_b3_manual.csv. Nada a fazer.")
        return

    df = preencher_data_ex(df)
    conferir_rendimento_por_evento(df)

    final = df.rename(columns={"valor_bruto": "valor"})[COLUNAS_SAIDA]
    final = ajustar_por_eventos(final)
    final = final.sort_values(["codneg", "data_ex", "tipo"]).reset_index(drop=True)

    if ARQUIVO_PROVENTOS.exists() and not ARQUIVO_ANTES_B3.exists():
        shutil.copy(ARQUIVO_PROVENTOS, ARQUIVO_ANTES_B3)       # guarda a versão anterior uma vez
    final.to_csv(ARQUIVO_PROVENTOS, index=False)
    print(f"proventos.csv gerado com {len(final)} linhas (fonte única: B3, valores brutos).\n")

    # as funções do baixar_dados.py recebem só as 4 colunas de sempre
    base = final[COLUNAS_BASE]
    comparar_com_yahoo(base)
    resumir_rendimento(base)
    resumir_cobertura(base)


if __name__ == "__main__":
    main()