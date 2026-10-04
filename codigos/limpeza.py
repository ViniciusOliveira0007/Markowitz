"""
Leitura, filtragem e tratamento dos arquivos COTAHIST da B3 para o projeto
PI-MARKOWITZ, restrito aos 15 ativos escolhidos nas 3 classes negociadas em
bolsa (5 ações, 5 FIIs, 5 ETFs). LCI/LCA fica fora deste pipeline por não
ser negociada na B3.

Lê todos os COTAHIST_A*.TXT de dados/dados_brutos/, filtra conforme o
dicionário de dados oficial e salva um único CSV limpo em dados/dados_limpos/.

Layout oficial:
https://www.b3.com.br/data/files/C8/F3/08/B4/297BE410F816C9E492D828A8/SeriesHistoricas_Layout.pdf
"""

from pathlib import Path

import polars as pl

# ----------------------------------------------------------------------
# Caminhos (ajustados à estrutura do projeto: codigos/limpeza.py)
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PASTA_DADOS_BRUTOS = BASE_DIR / "dados" / "dados_brutos"
PASTA_DADOS_LIMPOS = BASE_DIR / "dados" / "dados_limpos"
ARQUIVO_SAIDA = PASTA_DADOS_LIMPOS / "cotahist_limpo.csv"

# ----------------------------------------------------------------------
# Dicionário de dados: registro "01" (cotações), posições 1-indexadas
# conforme o PDF da B3
# ----------------------------------------------------------------------

CAMPOS_REGISTRO_01 = {
    "tipreg":      (1, 2),
    "data_pregao": (3, 10),
    "codbdi":      (11, 12),
    "codneg":      (13, 24),
    "tpmerc":      (25, 27),
    "nomres":      (28, 39),
    "especi":      (40, 49),
    "prazot":      (50, 52),
    "modref":      (53, 56),
    "preabe":      (57, 69),
    "premax":      (70, 82),
    "premin":      (83, 95),
    "premed":      (96, 108),
    "preult":      (109, 121),
    "preofc":      (122, 134),
    "preofv":      (135, 147),
    "totneg":      (148, 152),
    "quatot":      (153, 170),
    "voltot":      (171, 188),
    "codisi":      (231, 242),
}

CAMPOS_PRECO_2_DECIMAIS = [
    "preabe", "premax", "premin", "premed", "preult",
    "preofc", "preofv", "voltot",
]

# As 5 classes do projeto foram reduzidas a ativos específicos e populares
# (LCI/LCA fica fora: não é negociada na B3, não existe no COTAHIST).
TICKERS_ACOES = ["PETR4", "ITUB4", "B3SA3", "AXIA3", "EMBJ3"]
TICKERS_FII = ["MXRF11", "XPML11", "HGLG11", "KNRI11", "TGAR11"]
TICKERS_ETF = ["BOVA11", "SMAL11", "IVVB11", "GOLD11", "DIVO11"]

TICKER_PARA_CLASSE = {
    **{t: "acao" for t in TICKERS_ACOES},
    **{t: "fii" for t in TICKERS_FII},
    **{t: "etf" for t in TICKERS_ETF},
}

# Mercado à vista -- evita duplicar o mesmo ticker no mesmo dia por causa
# do mercado fracionário (TPMERC 020), que usa o mesmo CODNEG
TPMERC_VISTA = "010"


def _slice_expr(nome: str, pos_inicial: int, pos_final: int) -> pl.Expr:
    """Converte posição 1-indexada (início, fim) do PDF em str.slice(offset, tamanho)."""
    offset = pos_inicial - 1
    tamanho = pos_final - pos_inicial + 1
    return pl.col("linha").str.slice(offset, tamanho).alias(nome)


def carregar_e_extrair(caminho_arquivo: Path) -> pl.LazyFrame:
    """Lê um arquivo COTAHIST, mantém só o registro '01' e extrai os campos."""
    lf = pl.scan_csv(
        caminho_arquivo,
        has_header=False,
        separator="\x01",
        new_columns=["linha"],
        encoding="utf8-lossy",
    )
    lf = lf.filter(pl.col("linha").str.slice(0, 2) == "01")
    expressoes = [_slice_expr(nome, ini, fim) for nome, (ini, fim) in CAMPOS_REGISTRO_01.items()]
    return lf.with_columns(expressoes).drop("linha")


def filtrar_classes_de_interesse(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Mantém só os 15 tickers escolhidos (ações, FIIs e ETFs), só mercado à vista,
    e marca a classe de cada ativo numa coluna nova."""
    todos_tickers = list(TICKER_PARA_CLASSE.keys())
    codneg_limpo = pl.col("codneg").str.strip_chars()

    classe_expr = (
        pl.when(codneg_limpo.is_in(TICKERS_ACOES)).then(pl.lit("acao"))
        .when(codneg_limpo.is_in(TICKERS_FII)).then(pl.lit("fii"))
        .when(codneg_limpo.is_in(TICKERS_ETF)).then(pl.lit("etf"))
        .otherwise(None)
        .alias("classe_ativo")
    )

    lf = lf.filter(codneg_limpo.is_in(todos_tickers) & (pl.col("tpmerc") == TPMERC_VISTA))
    return lf.with_columns(classe_expr)


def padronizar_tipos(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Converte texto posicional em tipos utilizáveis (datas, floats, strings limpas)."""
    conversoes = [pl.col(col).str.strip_chars() for col in ["codneg", "nomres", "especi", "modref", "codisi"]]
    conversoes.append(pl.col("data_pregao").str.strptime(pl.Date, "%Y%m%d"))
    conversoes += [pl.col(col).cast(pl.Float64) / 100 for col in CAMPOS_PRECO_2_DECIMAIS]
    conversoes += [pl.col(col).cast(pl.Int64) for col in ["totneg", "quatot"]]
    return lf.with_columns(conversoes)


def processar_arquivo(caminho_arquivo: Path) -> pl.DataFrame:
    """Pipeline completo para um único arquivo COTAHIST."""
    lf = carregar_e_extrair(caminho_arquivo)
    lf = filtrar_classes_de_interesse(lf)
    lf = padronizar_tipos(lf)
    return lf.collect()


def main() -> None:
    PASTA_DADOS_LIMPOS.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(PASTA_DADOS_BRUTOS.glob("COTAHIST_A*.TXT"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum COTAHIST_A*.TXT encontrado em {PASTA_DADOS_BRUTOS}")

    dataframes = []
    for caminho in arquivos:
        print(f"Processando {caminho.name}...")
        df = processar_arquivo(caminho)
        print(f"  {df.height} linhas mantidas após o filtro")
        dataframes.append(df)

    df_final_polars = pl.concat(dataframes)

    # Conversão final para pandas, conforme solicitado (polars para a parte
    # pesada de leitura/filtragem, pandas para a exportação final)
    df_final_pandas = df_final_polars.to_pandas()
    df_final_pandas.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")

    print(f"\nArquivo final salvo em: {ARQUIVO_SAIDA}")
    print(f"Total de linhas: {len(df_final_pandas)}")


if __name__ == "__main__":
    main()