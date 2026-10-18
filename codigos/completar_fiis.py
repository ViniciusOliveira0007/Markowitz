"""
Completa os rendimentos de FIIs que o Yahoo não traz (meses do começo da amostra).

Fluxo (rode este mesmo script duas vezes):

  1ª vez  -> cria  dados/proventos_fiis_manual.csv  com UMA LINHA por FII e mês faltante
  você    -> preenche  data_com  e  valor  (R$ por cota) em cada linha; veja as dicas abaixo
  2ª vez  -> confere o que você preencheu e junta tudo em  dados/dados_limpos/proventos.csv

Dicas de preenchimento:
  * Fontes: página de "Dividendos" do fundo no Status Invest ou no Funds Explorer, ou os
    relatórios mensais do gestor. Use o valor por cota e a DATA COM (não a de pagamento).
  * O script calcula a data ex sozinho (primeiro pregão depois da data com).
  * Se um mês teve dois pagamentos, duplique a linha. Se não teve, deixe valor vazio.
  * Pode salvar pelo Excel (separador ';' e vírgula decimal) ou como CSV simples: os dois funcionam.
  * Datas aceitas: 31/03/2022 ou 2022-03-31.

Seguro de repetir: o script sempre parte do arquivo proventos_yahoo.csv (a versão só-Yahoo),
então rodar de novo não duplica nada. Se você rodar o baixar_dados.py outra vez, ele recria
o proventos.csv só com o Yahoo; basta rodar este script de novo para juntar o manual.

Precisa estar na mesma pasta do baixar_dados.py (reaproveita configurações e conferências dele).
"""

import shutil

import pandas as pd

from baixar_dados import (ARQUIVO_COTAHIST, ARQUIVO_PROVENTOS, BASE_DIR, CLASSE_ATIVO,
                          PASTA_DADOS, RENOMEAR_TICKERS, resumir_cobertura, resumir_rendimento)

ARQUIVO_YAHOO = PASTA_DADOS / "proventos_yahoo.csv"
ARQUIVO_MANUAL = BASE_DIR / "dados" / "proventos_fiis_manual.csv"

INICIO_AMOSTRA = pd.Timestamp("2020-12-01")
FIM_PADRAO = pd.Period("2025-12", freq="M")     # usado se o FII não tiver nada no Yahoo
RENDIMENTO_MENSAL_MAX = 0.03                    # acima de 3% do preço num mês: provável erro de digitação
RENDIMENTO_MENSAL_MIN = 0.002                   # abaixo de 0,2%: idem


# ----------------------------------------------------------------------
# Passo 1: gerar o modelo
# ----------------------------------------------------------------------

def gerar_modelo(prov_yahoo: pd.DataFrame) -> None:
    fiis = sorted(t for t, classe in CLASSE_ATIVO.items() if classe == "fii")
    linhas = []
    for ticker in fiis:
        datas = prov_yahoo.loc[prov_yahoo["codneg"] == ticker, "data_ex"]
        # meses sem dado = do início da amostra até o mês anterior ao 1º pagamento do Yahoo
        fim = datas.min().to_period("M") - 1 if len(datas) else FIM_PADRAO
        for mes in pd.period_range(INICIO_AMOSTRA.to_period("M"), fim, freq="M"):
            linhas.append({"codneg": ticker, "mes": str(mes), "data_com": "", "valor": ""})

    modelo = pd.DataFrame(linhas, columns=["codneg", "mes", "data_com", "valor"])
    ARQUIVO_MANUAL.parent.mkdir(parents=True, exist_ok=True)
    modelo.to_csv(ARQUIVO_MANUAL, index=False)

    print(f"Modelo criado com {len(modelo)} linhas em:\n  {ARQUIVO_MANUAL}\n")
    print(modelo.groupby("codneg")["mes"].agg(primeiro="min", ultimo="max", linhas="size").to_string())
    print("\nPreencha data_com e valor (R$ por cota) e rode este script de novo.")


# ----------------------------------------------------------------------
# Passo 2: ler o que foi preenchido, conferir e mesclar
# ----------------------------------------------------------------------

def ler_manual() -> pd.DataFrame:
    df = pd.read_csv(ARQUIVO_MANUAL, sep=None, engine="python", dtype=str)   # detecta ',' ou ';'
    df.columns = [c.strip().lower() for c in df.columns]
    df["codneg"] = df["codneg"].str.strip().str.upper().replace(RENOMEAR_TICKERS)

    valor = (df["valor"].fillna("").str.replace("R$", "", regex=False)
             .str.replace(" ", "", regex=False).str.replace(",", ".", regex=False))
    df["valor"] = pd.to_numeric(valor, errors="coerce")
    df["data_com"] = pd.to_datetime(df["data_com"], dayfirst=True, format="mixed", errors="coerce")
    return df


def datas_de_pregao() -> pd.DatetimeIndex:
    datas = pd.read_csv(ARQUIVO_COTAHIST, usecols=["data_pregao"], parse_dates=["data_pregao"])
    return pd.DatetimeIndex(sorted(datas["data_pregao"].unique()))


def proximo_pregao(data_com: pd.Series, pregoes: pd.DatetimeIndex) -> pd.Series:
    """Data ex = primeiro pregão DEPOIS da data com."""
    posicao = pregoes.searchsorted(data_com, side="right")
    validas = (posicao < len(pregoes)) & (data_com >= pregoes[0])
    datas_ex = pd.Series(pd.NaT, index=data_com.index, dtype="datetime64[ns]")
    datas_ex[validas] = pregoes[posicao[validas]]
    return datas_ex


def conferir_rendimento_mensal(novos: pd.DataFrame) -> None:
    """Avisa quando o provento é grande/pequeno demais perto do preço da cota (erro de digitação)."""
    df = pd.read_csv(ARQUIVO_COTAHIST, parse_dates=["data_pregao"], usecols=["data_pregao", "codneg", "preult"])
    df["codneg"] = df["codneg"].replace(RENOMEAR_TICKERS)
    precos = df.pivot(index="data_pregao", columns="codneg", values="preult")

    suspeitos = []
    for linha in novos.itertuples():
        if linha.codneg in precos.columns and linha.data_ex in precos.index:
            rendimento = linha.valor / precos.loc[linha.data_ex, linha.codneg]
            if not RENDIMENTO_MENSAL_MIN <= rendimento <= RENDIMENTO_MENSAL_MAX:
                suspeitos.append(f"  {linha.codneg} {linha.data_ex.date()}: R$ {linha.valor:.4f} "
                                 f"= {rendimento:.2%} do preço")
    if suspeitos:
        print("ATENÇÃO: valores fora do normal para um rendimento mensal (confira a digitação):")
        print("\n".join(suspeitos), "\n")


def mesclar(prov_yahoo: pd.DataFrame) -> None:
    manual = ler_manual()

    erradas = manual[manual["codneg"].map(CLASSE_ATIVO) != "fii"]
    if len(erradas):
        print(f"AVISO: {len(erradas)} linha(s) com ticker que não é FII no seu CLASSE_ATIVO foram ignoradas.")
        manual = manual.drop(erradas.index)

    preenchidas = manual.dropna(subset=["valor"])
    sem_data = preenchidas[preenchidas["data_com"].isna()]
    if len(sem_data):
        print(f"AVISO: {len(sem_data)} linha(s) com valor mas SEM data_com válida foram ignoradas.")
    usaveis = preenchidas.dropna(subset=["data_com"])
    usaveis = usaveis[usaveis["valor"] > 0]

    if usaveis.empty:
        print("Nenhuma linha preenchida (valor e data_com) em proventos_fiis_manual.csv. Nada a fazer.")
        return

    novos = usaveis.assign(data_ex=proximo_pregao(usaveis["data_com"], datas_de_pregao()), tipo="RENDIMENTO")
    fora = novos[novos["data_ex"].isna()]
    if len(fora):
        print(f"AVISO: {len(fora)} linha(s) com data_com fora do período dos preços foram ignoradas.")
    novos = novos.dropna(subset=["data_ex"])[["data_ex", "codneg", "tipo", "valor"]]

    conferir_rendimento_mensal(novos)

    junto = pd.concat([prov_yahoo, novos], ignore_index=True)
    antes = len(junto)
    junto = junto.drop_duplicates(subset=["codneg", "data_ex", "tipo"], keep="first")   # Yahoo vence
    junto = junto.sort_values(["codneg", "data_ex", "tipo"]).reset_index(drop=True)
    junto.to_csv(ARQUIVO_PROVENTOS, index=False)

    vazias = len(manual) - len(preenchidas)
    print(f"Mesclado: {len(novos)} pagamentos manuais ({antes - len(junto)} repetidos descartados). "
          f"{vazias} linha(s) do modelo continuam sem valor.")
    print(f"Salvo em: {ARQUIVO_PROVENTOS}")
    resumir_rendimento(junto)
    resumir_cobertura(junto)


# ----------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------

def main() -> None:
    if not ARQUIVO_YAHOO.exists():                  # 1ª execução: guarda a versão só-Yahoo
        shutil.copy(ARQUIVO_PROVENTOS, ARQUIVO_YAHOO)
    prov_yahoo = pd.read_csv(ARQUIVO_YAHOO, parse_dates=["data_ex"])

    if not ARQUIVO_MANUAL.exists():
        gerar_modelo(prov_yahoo)
    else:
        mesclar(prov_yahoo)


if __name__ == "__main__":
    main()