"""
Fronteira Eficiente de Markowitz em formato LÍQUIDO (após tributação) - PI-MARKOWITZ.
Baseado no documento "PI - Ativos" (Tesouro Selic, LCI/LCA, Ações, ETFs, FIIs).

Fluxo:
    preços + proventos (+ Selic)
      -> retorno total diário BRUTO   R_t = (P_t + D_t) / P_(t-1) - 1
      -> retorno total diário LÍQUIDO de proventos (dividendo, JCP, rendimento de FII)
      -> retorno esperado líquido (média anual - "peso" do imposto sobre o ganho de capital)
      -> covariância (calculada sobre os retornos BRUTOS, como no PDF)
      -> carteiras aleatórias -> fronteira eficiente

Saídas: a MESMA carteira é avaliada em duas bases, só muda o imposto:
    retorno BRUTO   (retorno total: preço + proventos, antes do imposto)
    retorno LÍQUIDO (depois do imposto)
  -> fronteira_eficiente_bruto.png, fronteira_eficiente_liquido.png e
     comparacao_bruto_liquido.png (as duas nuvens sobrepostas)

Arquivos de entrada:
    dados/dados_limpos/cotahist_limpo.csv  (obrigatório)  colunas: data_pregao, codneg, preult, tpmerc, codbdi
    dados/dados_limpos/proventos.csv       (opcional)     colunas: data_ex, codneg, tipo, valor [, data_pagamento]
                                           tipo = DIVIDENDO | JCP | RENDIMENTO (FII)
                                           valor = R$ por ação/cota BRUTO (data_ex = dia em que o preço sente o provento)
                                           data_pagamento (opcional) = define a alíquota do JCP: 15% se pago até 2025, 17,5% depois
    dados/selic_diaria.csv                 (opcional)     colunas: data, selic_dia
                                           (gerado pelo baixar_selic.py; selic_dia já em DECIMAL ao dia)

A taxa livre de risco (rf) do Sharpe vem de taxa_livre_risco.py, calculada sobre o MESMO
período da amostra de retornos.

ATENÇÃO: as alíquotas abaixo refletem a legislação em vigor em out/2026 para pessoa física.
Confirme antes de usar em trabalho oficial.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from taxa_livre_risco import taxa_livre_risco_anual

# ----------------------------------------------------------------------
# Configurações gerais
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PASTA_DADOS = BASE_DIR / "dados" / "dados_limpos"
ARQUIVO_ENTRADA = PASTA_DADOS / "cotahist_limpo.csv"
ARQUIVO_PROVENTOS = PASTA_DADOS / "proventos.csv"
ARQUIVO_SELIC = BASE_DIR / "dados" / "selic_diaria.csv"   # onde o baixar_selic.py salva
PASTA_RESULTADOS = BASE_DIR / "resultados" / "liquido"   # pasta própria: não sobrescreve o Markowitz bruto

DIAS_UTEIS_ANO = 252
N_CARTEIRAS = 2_000_000
SEMENTE = 42

# Taxa livre de risco do Sharpe:
#   False -> Selic BRUTA anualizada no período da amostra (convenção comum)
#   True  -> retorno esperado LÍQUIDO do Tesouro Selic (mesma base das carteiras, que são líquidas)
#            Só funciona se o Tesouro Selic estiver na carteira; senão cai para a Selic bruta.
RF_USA_SELIC_LIQUIDA = False

# Proventos: o proventos.csv já está ajustado por desdobramento/grupamento?
#   False -> o código divide os proventos anteriores a cada evento pelo fator
#            (use se os valores são os nominais da época, como a B3 divulga)
#   True  -> use depois de gerar o proventos.csv com o montar_proventos_b3.py (ele já ajusta)
PROVENTOS_JA_AJUSTADOS = True     # proventos.csv vem do montar_proventos_b3.py, que já ajusta

# Quantos pontos desenhar em cada gráfico. O cálculo (máximo Sharpe etc.) usa TODAS as
# carteiras; o gráfico desenha só uma amostra, para ficar rápido e legível.
PONTOS_NO_GRAFICO = 400_000

RENOMEAR_TICKERS = {
    "ELET3": "AXIA3",   # código antigo -> código atual
    "EMBR3": "EMBJ3",
}

# ----------------------------------------------------------------------
# Classe de cada ativo (VOCÊ preenche). Quem não estiver aqui vira "acao".
# Valores aceitos: "acao", "etf", "fii"
# ----------------------------------------------------------------------

CLASSE_ATIVO = {
    # ações
    "AXIA3": "acao", "B3SA3": "acao", "EMBJ3": "acao", "ITUB4": "acao", "PETR4": "acao",
    # ETFs
    "BOVA11": "etf", "DIVO11": "etf", "GOLD11": "etf", "IVVB11": "etf", "SMAL11": "etf",
    # FIIs
    "HGLG11": "fii", "KNRI11": "fii", "MXRF11": "fii", "TGAR11": "fii", "XPML11": "fii",
}
CLASSE_PADRAO = "acao"

# ----------------------------------------------------------------------
# Desdobramentos / grupamentos (o COTAHIST NÃO ajusta os preços antigos)
# Formato: ("TICKER", "AAAA-MM-DD do primeiro pregão com o preço novo"): fator
#   desdobramento 1 para 10  -> fator 10     (preço cai 10x; preços ANTERIORES são divididos por 10)
#   grupamento 10 para 1     -> fator 0.1    (preço sobe 10x; preços anteriores multiplicados por 10)
# Rode uma vez: o código lista os saltos suspeitos e a razão de preço sugerida.
# ----------------------------------------------------------------------
EVENTOS_CORPORATIVOS = {
    ("B3SA3", "2021-05-17"): 3,    # desdobramento 1 para 3 (data COM 14/05/2021)
    # outros eventos entram aqui, ex.: ("TICKER", "AAAA-MM-DD"): 10,
    ("ITUB4", "2025-03-18"): 1.10,
    ("ITUB4", "2025-12-26"): 1.03,
    ("AXIA3", "2025-12-22"): 1.2628,   # aproximação: o bônus veio em AXIA7
}
LIMITE_SALTO = 0.25              # variação diária acima disso é tratada como suspeita
PARAR_SE_HOUVER_SALTOS = True    # False = só avisa e segue (resultado pode ficar distorcido!)

# ----------------------------------------------------------------------
# Regras tributárias (pessoa física)
# ----------------------------------------------------------------------

# Imposto sobre o GANHO DE CAPITAL na venda (swing trade), assumindo venda no fim do horizonte
ALIQUOTA_GANHO = {"acao": 0.15, "etf": 0.15, "fii": 0.20}

# Ações: vendas até R$ 20 mil/mês são isentas ("tudo ou nada"). Depende do SEU volume de vendas;
# ETF e FII não têm essa isenção. Deixe False para o cenário conservador.
APLICAR_ISENCAO_20K = False

# Proventos
ALIQUOTA_DIVIDENDO = 0.0           # isento p/ PF (10% só se > R$ 50 mil/mês pagos por UMA empresa)
ALIQUOTA_RENDIMENTO_FII = 0.0      # isento p/ PF (fundo com 100+ cotistas, PF com menos de 10% das cotas)
DATA_MUDANCA_JCP = pd.Timestamp("2026-01-01")
ALIQUOTA_JCP_ANTES = 0.15          # até 31/12/2025
ALIQUOTA_JCP_DEPOIS = 0.175        # a partir de 01/01/2026 (LC 224/2025)

# Tesouro Selic
SELIC_EM_PERCENTUAL = False        # False: o selic_diaria.csv do baixar_selic.py já vem em decimal (ex.: 0.000551)
VALOR_APLICADO_SELIC = 50_000.0    # R$ assumidos no título (a custódia depende do valor!)
TAXA_CUSTODIA_B3 = 0.0020          # 0,20% a.a.
LIMITE_ISENCAO_CUSTODIA = 10_000.0 # Selic: não paga custódia nos primeiros R$ 10 mil
DIAS_ANO_CUSTODIA = 365

# LCI/LCA: isentas de IR para pessoa física, sem IOF e sem custódia na B3 (o banco emissor guarda).
# O retorno é um percentual do CDI; a Selic efetiva diária (selic_diaria.csv) é a proxy do CDI,
# pois o CDI fica a centésimos dela. A taxa fixa é uma PREMISSA: usa o valor de hoje em toda a amostra.
INCLUIR_LCI_LCA = True
PERCENTUAL_CDI_LCI_LCA = 0.88      # taxa média de 2026: LCI 88,00% do CDI, LCA 88,24% (documento, seção 2.3)


# ----------------------------------------------------------------------
# 1. Carregar dados
# ----------------------------------------------------------------------

def carregar_precos(caminho: Path) -> pd.DataFrame:
    """Preços de fechamento: linhas = datas, colunas = tickers."""
    df = pd.read_csv(
        caminho,
        parse_dates=["data_pregao"],
        dtype={"tpmerc": str, "codbdi": str},
    )
    df["codneg"] = df["codneg"].replace(RENOMEAR_TICKERS)
    precos = df.pivot(index="data_pregao", columns="codneg", values="preult").sort_index()
    return ajustar_eventos(precos, EVENTOS_CORPORATIVOS)


def diagnosticar_historico(precos: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "primeira_data": precos.apply(pd.Series.first_valid_index),
        "ultima_data": precos.apply(pd.Series.last_valid_index),
        "dias_com_preco": precos.notna().sum(),
    }).sort_values("primeira_data")


def ajustar_eventos(precos: pd.DataFrame, eventos: dict) -> pd.DataFrame:
    """Ajusta preços ANTERIORES à data de cada desdobramento/grupamento."""
    precos = precos.copy()
    for (ticker, data), fator in eventos.items():
        if ticker not in precos.columns:
            continue
        data = pd.Timestamp(data)
        precos.loc[precos.index < data, ticker] = precos.loc[precos.index < data, ticker] / fator
    return precos


def ajustar_proventos_eventos(prov: pd.DataFrame, eventos: dict) -> pd.DataFrame:
    """Proventos pagos ANTES do evento estão em R$ por ação da época: divide pelo mesmo fator
    dos preços. Use isto se proventos.csv traz valores nominais (como a B3 divulga). Se a sua
    fonte já entrega proventos ajustados, deixe EVENTOS_CORPORATIVOS vazio para os proventos."""
    prov = prov.copy()
    for (ticker, data), fator in eventos.items():
        antes = (prov["codneg"] == ticker) & (prov["data_ex"] < pd.Timestamp(data))
        prov.loc[antes, "valor"] = prov.loc[antes, "valor"] / fator
    return prov


def detectar_saltos(precos: pd.DataFrame, limite: float = LIMITE_SALTO) -> pd.DataFrame:
    """Lista variações diárias acima do limite (candidatas a desdobramento/grupamento)."""
    variacao = precos / precos.shift(1) - 1
    tabela = (variacao.rename_axis("data").reset_index()
              .melt(id_vars="data", var_name="codneg", value_name="retorno"))
    tabela = tabela[tabela["retorno"].abs() > limite].copy()
    tabela["razao_preco_antes_depois"] = 1 / (1 + tabela["retorno"])
    return tabela.sort_values(["codneg", "data"])


def carregar_proventos(caminho: Path) -> pd.DataFrame:
    """Lê os proventos. Se o arquivo não existir, devolve tabela vazia (e avisa)."""
    if not caminho.exists():
        print("AVISO: proventos.csv não encontrado. Ações e FIIs terão só o retorno de PREÇO "
              "(sem dividendos/JCP/rendimentos). Isso subestima muito o retorno de FIIs.\n")
        return pd.DataFrame(columns=["data_ex", "codneg", "tipo", "valor"])

    colunas_data = ["data_ex"]
    if "data_pagamento" in pd.read_csv(caminho, nrows=0).columns:
        colunas_data.append("data_pagamento")
    prov = pd.read_csv(caminho, parse_dates=colunas_data)
    prov["codneg"] = prov["codneg"].replace(RENOMEAR_TICKERS)
    prov["tipo"] = prov["tipo"].str.upper().str.strip()

    invalidos = set(prov["tipo"]) - {"DIVIDENDO", "JCP", "RENDIMENTO"}
    if invalidos:
        raise ValueError(f"Tipos de provento desconhecidos: {invalidos}")
    return prov


def carregar_selic(caminho: Path, datas: pd.DatetimeIndex) -> pd.Series | None:
    """Taxa Selic diária (fração ao dia) alinhada às datas dos preços. None se não houver arquivo."""
    if not caminho.exists():
        print("AVISO: selic_diaria.csv não encontrado. Tesouro Selic NÃO entra na carteira.\n")
        return None

    df = pd.read_csv(caminho, parse_dates=["data"])
    taxa = df.set_index("data")["selic_dia"].sort_index()
    if SELIC_EM_PERCENTUAL:
        taxa = taxa / 100
    return taxa.reindex(datas).ffill().rename("SELIC")


# ----------------------------------------------------------------------
# 2. Classes e alíquotas
# ----------------------------------------------------------------------

ATIVOS_SINTETICOS = {"SELIC": "selic", "LCI_LCA": "lci_lca"}   # não vêm do COTAHIST


def classe_do_ativo(ticker: str) -> str:
    if ticker in ATIVOS_SINTETICOS:
        return ATIVOS_SINTETICOS[ticker]
    return CLASSE_ATIVO.get(ticker, CLASSE_PADRAO)


def avisar_tickers_sem_classe(tickers) -> None:
    sem_classe = [t for t in tickers if t not in ATIVOS_SINTETICOS and t not in CLASSE_ATIVO]
    if sem_classe:
        print(f"AVISO: {len(sem_classe)} ticker(s) fora de CLASSE_ATIVO, tratados como "
              f"'{CLASSE_PADRAO}': {', '.join(sem_classe)}\n")


def aliquota_provento(prov: pd.DataFrame) -> pd.Series:
    """Alíquota de cada provento. O JCP muda de 15% para 17,5% em 01/01/2026, e vale a data de
    PAGAMENTO (JCP aprovado em 2025 e pago em 2026 paga 17,5%). Sem data_pagamento, usa a data ex."""
    tipo = prov["tipo"]
    data = prov["data_ex"]
    if "data_pagamento" in prov.columns:
        data = prov["data_pagamento"].fillna(prov["data_ex"])
    condicoes = [
        tipo == "DIVIDENDO",
        tipo == "RENDIMENTO",
        (tipo == "JCP") & (data >= DATA_MUDANCA_JCP),
        tipo == "JCP",
    ]
    valores = [ALIQUOTA_DIVIDENDO, ALIQUOTA_RENDIMENTO_FII,
               ALIQUOTA_JCP_DEPOIS, ALIQUOTA_JCP_ANTES]
    return pd.Series(np.select(condicoes, valores, default=0.0), index=prov.index)


def aliquota_ir_renda_fixa(dias_corridos: int) -> float:
    """Tabela regressiva do IR (Tesouro Direto)."""
    if dias_corridos <= 180:
        return 0.225
    if dias_corridos <= 360:
        return 0.20
    if dias_corridos <= 720:
        return 0.175
    return 0.15


def aliquota_iof(dias_corridos: int) -> float:
    """IOF regressivo: 96% no dia 1, cai até 0% a partir do dia 30."""
    if dias_corridos >= 30:
        return 0.0
    return float(np.floor(100 * (30 - dias_corridos) / 30) / 100)


# ----------------------------------------------------------------------
# 3. Proventos por dia (bruto e líquido)
# ----------------------------------------------------------------------

def proventos_por_dia(prov: pd.DataFrame, datas: pd.DatetimeIndex,
                      colunas: pd.Index, coluna_valor: str) -> pd.DataFrame:
    """Tabela datas x tickers com o provento (R$/ação) pago em cada data EX."""
    if prov.empty:
        return pd.DataFrame(0.0, index=datas, columns=colunas)
    tabela = prov.pivot_table(index="data_ex", columns="codneg",
                              values=coluna_valor, aggfunc="sum")
    tabela = tabela.reindex(columns=colunas)
    perdido = tabela.sum().sum() - tabela.reindex(datas).sum().sum()
    if perdido > 1e-9:
        print(f"AVISO: R$ {perdido:.2f} de proventos caíram em datas sem pregão e foram ignorados.\n")
    return tabela.reindex(datas).fillna(0.0)


# ----------------------------------------------------------------------
# 4. Retornos totais
# ----------------------------------------------------------------------

def calcular_retornos_totais(precos, prov_bruto, prov_liquido):
    """Retorno total diário: R_t = (P_t + D_t) / P_(t-1) - 1.

    CORREÇÃO em relação ao PDF: o PDF soma o provento ao preço só no dia t
    (P_ajustado,t = P_t + D_t) e divide pelo P_ajustado,t-1. No dia seguinte o provento
    some do numerador e o retorno "devolve" o que ganhou, anulando o efeito. O certo
    é somar o provento no numerador e dividir pelo PREÇO de ontem.
    """
    precos_comuns = precos.dropna()
    anterior = precos_comuns.shift(1)
    d_bruto = prov_bruto.reindex(precos_comuns.index)
    d_liquido = prov_liquido.reindex(precos_comuns.index)

    ret_bruto = ((precos_comuns + d_bruto) / anterior - 1).dropna()
    ret_liquido = ((precos_comuns + d_liquido) / anterior - 1).dropna()
    return ret_bruto, ret_liquido, precos_comuns


# ----------------------------------------------------------------------
# 5. Imposto sobre o ganho de capital (vira um "desconto" anual no retorno esperado)
# ----------------------------------------------------------------------

def reducao_anual_imposto_ganho(precos_comuns: pd.DataFrame) -> pd.Series:
    """Imposto sobre (P_T - P_i), só se houver ganho, distribuído ao longo dos anos da amostra.

    CORREÇÕES em relação ao PDF: (1) imposto não pode ser negativo quando há prejuízo
    (clip em zero); (2) o imposto NÃO é jogado inteiro no último dia (isso criaria um
    retorno diário absurdo e destruiria a variância). Ele vira um desconto anual no retorno
    esperado.

    Base do desconto: o imposto como fração do VALOR FINAL, f = aliquota * ganho / (1 + ganho).
    Isso equivale a um retorno de -f no último dia, que é o que o PDF faz (I / P_(T-1)).
    Dividir pelo preço INICIAL superestima o desconto de quem multiplicou de valor
    (ex.: ganho de +880% daria um desconto 10x maior que o correto).
    """
    anos = (len(precos_comuns) - 1) / DIAS_UTEIS_ANO
    ganho = precos_comuns.iloc[-1] / precos_comuns.iloc[0] - 1

    aliquota = pd.Series(
        {t: ALIQUOTA_GANHO[classe_do_ativo(t)] for t in precos_comuns.columns}
    )
    if APLICAR_ISENCAO_20K:
        for t in precos_comuns.columns:
            if classe_do_ativo(t) == "acao":
                aliquota[t] = 0.0

    imposto_sobre_valor_final = aliquota * ganho.clip(lower=0) / (1 + ganho)
    return imposto_sobre_valor_final / anos


def custos_tesouro_selic(taxa_diaria: pd.Series) -> dict:
    """IOF + IR + custódia do Tesouro Selic, em fração do valor aplicado.

    CORREÇÕES em relação ao PDF:
      (1) IOF e IR não se somam: o IOF sai primeiro e o IR incide sobre (ganho - IOF).
      (2) As alíquotas dependem do prazo (IR regressivo, IOF só nos 30 primeiros dias).
      (3) R_liq = R_bruto - (I + C)/P_i. O PDF subtrai valores em R$ de uma taxa (%).
    """
    datas = taxa_diaria.index
    dias_corridos = (datas[-1] - datas[0]).days
    anos = (len(datas) - 1) / DIAS_UTEIS_ANO

    acumulado = (1 + taxa_diaria).cumprod()         # evolução de R$ 1 aplicado
    ganho = acumulado.iloc[-1] - 1

    iof = ganho * aliquota_iof(dias_corridos)
    ir = (ganho - iof) * aliquota_ir_renda_fixa(dias_corridos)

    # Custódia (fórmula do PDF): C = t_c * soma_d( max(V_d - L, 0) / D ), por dia corrido
    valor_dia = VALOR_APLICADO_SELIC * acumulado
    dias_passo = datas.to_series().diff().dt.days.fillna(0)
    excedente = np.maximum(valor_dia - LIMITE_ISENCAO_CUSTODIA, 0)
    custodia_reais = TAXA_CUSTODIA_B3 * (excedente * dias_passo / DIAS_ANO_CUSTODIA).sum()
    custodia = custodia_reais / VALOR_APLICADO_SELIC

    return {
        "ganho_bruto": ganho,
        "iof": iof,
        "ir": ir,
        "custodia": custodia,
        # custos como fração do valor final, para ficar na mesma base dos demais ativos
        "reducao_anual": (iof + ir + custodia) / (1 + ganho) / anos,
    }


# ----------------------------------------------------------------------
# 6. Estatísticas anuais
# ----------------------------------------------------------------------

def estatisticas_anuais(ret_bruto, ret_liquido, reducao_anual):
    """media_bruta, media_liquida e covariância (sobre os retornos BRUTOS, como no PDF)."""
    media_bruta = ret_bruto.mean() * DIAS_UTEIS_ANO
    media_liquida = (ret_liquido.mean() * DIAS_UTEIS_ANO
                     - reducao_anual.reindex(ret_liquido.columns))
    cov = ret_bruto.cov() * DIAS_UTEIS_ANO
    return media_bruta, media_liquida, cov


def tabela_bruto_vs_liquido(media_bruta: pd.Series, media_liquida: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "classe": [classe_do_ativo(t) for t in media_bruta.index],
        "retorno_bruto": media_bruta,
        "retorno_liquido": media_liquida,
        "perda_imposto": media_bruta - media_liquida,
    })


def validar(ret_bruto: pd.DataFrame, media_liquida: pd.Series, cov: pd.DataFrame) -> None:
    assert ret_bruto.notna().all().all(), "sobrou NaN nos retornos"
    assert media_liquida.notna().all(), "retorno líquido com NaN (falta classe ou provento?)"
    assert np.allclose(cov, cov.T), "covariância não simétrica"


def escolher_taxa_livre_risco(ret_bruto: pd.DataFrame, media_liquida: pd.Series) -> tuple[float, float]:
    """Devolve (rf do Sharpe líquido, rf do Sharpe bruto), sobre o MESMO período da amostra.

    - Sharpe BRUTO: sempre a Selic bruta anualizada (taxa_livre_risco.py).
    - Sharpe LÍQUIDO: a Selic bruta, ou, se RF_USA_SELIC_LIQUIDA = True e o Tesouro Selic
      estiver na carteira, o retorno líquido dele.
    """
    inicio, fim = ret_bruto.index.min(), ret_bruto.index.max()
    rf_bruta = taxa_livre_risco_anual(inicio, fim)
    print(f"Selic bruta anualizada na amostra ({inicio.date()} a {fim.date()}): {rf_bruta:.2%}")

    rf_liquida = rf_bruta
    if RF_USA_SELIC_LIQUIDA and "SELIC" in media_liquida.index:
        rf_liquida = float(media_liquida["SELIC"])
        print(f"rf do Sharpe líquido: Tesouro Selic LÍQUIDO = {rf_liquida:.2%}")
    elif RF_USA_SELIC_LIQUIDA:
        print("AVISO: RF_USA_SELIC_LIQUIDA = True, mas o Tesouro Selic não está na carteira. "
              "Usando a Selic bruta.")
    else:
        print(f"rf do Sharpe líquido: Selic BRUTA = {rf_liquida:.2%}")
    print(f"rf do Sharpe bruto:   Selic BRUTA = {rf_bruta:.2%}\n")
    return rf_liquida, rf_bruta


# ----------------------------------------------------------------------
# 7. Carteiras
# ----------------------------------------------------------------------

def desempenho_carteiras(pesos: np.ndarray, media: pd.Series, cov: pd.DataFrame):
    """Retorno e risco de várias carteiras de uma vez. Risco = raiz(w' · Σ · w)."""
    retornos = pesos @ media.to_numpy()
    variancias = np.einsum("ij,jk,ik->i", pesos, cov.to_numpy(), pesos)
    return retornos, np.sqrt(variancias)


def simular_carteiras(media_bruta, media_liquida, cov, taxa_livre_risco: float,
                      taxa_livre_risco_bruta: float, n_carteiras: int = N_CARTEIRAS):
    """Sorteia carteiras e avalia cada uma nas duas bases (mesmos pesos, mesmo risco).

    retorno_bruto / sharpe_bruto : antes do imposto (retorno total, com proventos)
    retorno / sharpe             : depois do imposto
    """
    rng = np.random.default_rng(SEMENTE)
    pesos = rng.dirichlet(np.ones(len(media_liquida)), size=n_carteiras)
    retorno_liq, risco = desempenho_carteiras(pesos, media_liquida, cov)
    retorno_bruto = pesos @ media_bruta.to_numpy()

    resultado = pd.DataFrame(pesos, columns=media_liquida.index)
    resultado["retorno_bruto"] = retorno_bruto
    resultado["retorno"] = retorno_liq          # retorno LÍQUIDO
    resultado["risco"] = risco
    resultado["sharpe_bruto"] = (retorno_bruto - taxa_livre_risco_bruta) / risco
    resultado["sharpe"] = (retorno_liq - taxa_livre_risco) / risco
    return resultado


# ----------------------------------------------------------------------
# 8. Gráfico
# ----------------------------------------------------------------------

def amostra_para_grafico(carteiras: pd.DataFrame) -> pd.DataFrame:
    return carteiras.sample(min(PONTOS_NO_GRAFICO, len(carteiras)), random_state=SEMENTE)


def plotar_fronteira(carteiras: pd.DataFrame, caminho_saida: Path,
                     coluna_retorno: str = "retorno", coluna_sharpe: str = "sharpe",
                     base: str = "líquido") -> None:
    """Fronteira de uma base. base = 'líquido' (coluna retorno) ou 'bruto' (retorno_bruto)."""
    melhor_sharpe = carteiras.loc[carteiras[coluna_sharpe].idxmax()]
    menor_risco = carteiras.loc[carteiras["risco"].idxmin()]
    amostra = amostra_para_grafico(carteiras)

    fig, ax = plt.subplots(figsize=(10, 6))
    pontos = ax.scatter(amostra["risco"], amostra[coluna_retorno],
                        c=amostra[coluna_sharpe], cmap="viridis", s=6, alpha=0.6)
    ax.scatter(melhor_sharpe["risco"], melhor_sharpe[coluna_retorno],
               marker="*", s=300, c="red", label="Máximo Sharpe")
    ax.scatter(menor_risco["risco"], menor_risco[coluna_retorno],
               marker="D", s=100, c="orange", label="Mínima variância")
    fig.colorbar(pontos, label=f"Índice de Sharpe ({base})")
    ax.set_xlabel("Risco (desvio padrão anual)")
    ax.set_ylabel(f"Retorno esperado {base} anual")
    ax.set_title(f"Fronteira Eficiente de Markowitz - retorno {base}")
    ax.legend(loc="upper left")   # loc fixo: "best" é lento com muitos pontos
    fig.tight_layout()

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


def plotar_comparacao(carteiras: pd.DataFrame, caminho_saida: Path) -> None:
    """Sobrepõe as duas bases. Cada carteira aparece duas vezes, no MESMO risco:
    a nuvem cinza (bruto) fica acima da azul (líquido), e a distância é o imposto."""
    melhor_bruto = carteiras.loc[carteiras["sharpe_bruto"].idxmax()]
    melhor_liquido = carteiras.loc[carteiras["sharpe"].idxmax()]
    # idxmin() devolve o índice da linha de menor risco -> .loc pega a carteira inteira.
    # O risco é o mesmo nas duas bases, então é UMA carteira só.
    min_var = carteiras.loc[carteiras["risco"].idxmin()]
    amostra = amostra_para_grafico(carteiras)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(amostra["risco"], amostra["retorno_bruto"], s=4, alpha=0.25,
               c="tab:gray", label="Bruto (antes do imposto)")
    ax.scatter(amostra["risco"], amostra["retorno"], s=4, alpha=0.25,
               c="tab:blue", label="Líquido (depois do imposto)")
    ax.scatter(melhor_bruto["risco"], melhor_bruto["retorno_bruto"],
               marker="*", s=300, c="black", label="Máximo Sharpe bruto")
    ax.scatter(melhor_liquido["risco"], melhor_liquido["retorno"],
               marker="*", s=300, c="red", label="Máximo Sharpe líquido")

    # NOVO: mínima variância, um marcador por base (mesmo x, y diferente)
    ax.scatter(min_var["risco"], min_var["retorno_bruto"],
               marker="D", s=110, c="gold", edgecolors="black", linewidths=1.2, zorder=5)
    ax.scatter(min_var["risco"], min_var["retorno"],
               marker="D", s=110, c="darkorange", edgecolors="black", linewidths=1.2, zorder=5)

    ax.set_xlabel("Risco (desvio padrão anual)")
    ax.set_ylabel("Retorno esperado anual")
    ax.set_title("Fronteira Eficiente: bruto x líquido (mesmas carteiras)")
    legenda = [
        Line2D([], [], marker="o", ls="", c="tab:gray", markersize=7, label="Bruto (antes do imposto)"),
        Line2D([], [], marker="o", ls="", c="tab:blue", markersize=7, label="Líquido (depois do imposto)"),
        Line2D([], [], marker="*", ls="", c="black", markersize=14, label="Máximo Sharpe bruto"),
        Line2D([], [], marker="*", ls="", c="red", markersize=14, label="Máximo Sharpe líquido"),
        # NOVO
        Line2D([], [], marker="D", ls="", mfc="gold", mec="black", markersize=9, label="Mínima variância bruta"),
        Line2D([], [], marker="D", ls="", mfc="darkorange", mec="black", markersize=9, label="Mínima variância líquida"),
    ]
    ax.legend(handles=legenda, loc="lower right")
    fig.tight_layout()

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_saida, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 9. Execução
# ----------------------------------------------------------------------

def main() -> None:
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    precos = carregar_precos(ARQUIVO_ENTRADA)
    print("Histórico por ativo:")
    print(diagnosticar_historico(precos), "\n")
    avisar_tickers_sem_classe(precos.columns)

    # Proventos: bruto e líquido de imposto (cada tipo tem sua alíquota)
    prov = carregar_proventos(ARQUIVO_PROVENTOS)
    if not PROVENTOS_JA_AJUSTADOS:
        prov = ajustar_proventos_eventos(prov, EVENTOS_CORPORATIVOS)
    if not prov.empty:
        prov["valor_liquido"] = prov["valor"] * (1 - aliquota_provento(prov))
    else:
        prov["valor_liquido"] = pd.Series(dtype=float)
    prov_bruto = proventos_por_dia(prov, precos.index, precos.columns, "valor")
    prov_liquido = proventos_por_dia(prov, precos.index, precos.columns, "valor_liquido")

    ret_bruto, ret_liquido, precos_comuns = calcular_retornos_totais(precos, prov_bruto, prov_liquido)

    saltos = detectar_saltos(precos_comuns)
    if not saltos.empty:
        print(f"SALTOS DE PREÇO SUSPEITOS (acima de {LIMITE_SALTO:.0%} em um dia):")
        print(saltos.round({"retorno": 4, "razao_preco_antes_depois": 2}).to_string(index=False), "\n")
        if PARAR_SE_HOUVER_SALTOS:
            raise SystemExit(
                "Parei: confira a tabela acima. Se for desdobramento/grupamento, preencha "
                "EVENTOS_CORPORATIVOS (use a razão sugerida, arredondada). Se o salto for real, "
                "aumente LIMITE_SALTO ou ponha PARAR_SE_HOUVER_SALTOS = False.")
    reducao = reducao_anual_imposto_ganho(precos_comuns)

    # Tesouro Selic (opcional): entra como mais uma coluna
    taxa_selic = carregar_selic(ARQUIVO_SELIC, ret_bruto.index)
    if taxa_selic is not None:
        custos = custos_tesouro_selic(taxa_selic)
        print("Tesouro Selic na amostra:",
              {k: round(v, 4) for k, v in custos.items()}, "\n")
        ret_bruto["SELIC"] = taxa_selic
        ret_liquido["SELIC"] = taxa_selic
        reducao["SELIC"] = custos["reducao_anual"]

    # LCI/LCA (opcional): % do CDI, isenta de IR p/ PF -> bruto = líquido, sem custos
    if INCLUIR_LCI_LCA:
        if taxa_selic is None:
            print("AVISO: sem selic_diaria.csv a LCI/LCA não entra (o CDI é aproximado pela Selic).\n")
        else:
            ret_bruto["LCI_LCA"] = taxa_selic * PERCENTUAL_CDI_LCI_LCA
            ret_liquido["LCI_LCA"] = ret_bruto["LCI_LCA"]
            reducao["LCI_LCA"] = 0.0
            print(f"LCI/LCA: {PERCENTUAL_CDI_LCI_LCA:.2%} do CDI, isenta de IR, sem IOF e sem custódia.\n")

    print(f"Amostra comum: {ret_bruto.index.min().date()} a {ret_bruto.index.max().date()} "
          f"({len(ret_bruto)} retornos diários)\n")

    media_bruta, media_liquida, cov = estatisticas_anuais(ret_bruto, ret_liquido, reducao)
    validar(ret_bruto, media_liquida, cov)

    # Taxa livre de risco do Sharpe, no mesmo período da amostra
    rf, rf_bruta = escolher_taxa_livre_risco(ret_bruto, media_liquida)

    comparacao = tabela_bruto_vs_liquido(media_bruta, media_liquida)
    print("Retorno esperado anual: bruto x líquido")
    print(comparacao.round(4), "\n")

    comparacao.to_csv(PASTA_RESULTADOS / "retorno_bruto_vs_liquido.csv")
    cov.to_csv(PASTA_RESULTADOS / "matriz_covariancia_liquido.csv")
    media_liquida.to_csv(PASTA_RESULTADOS / "retorno_esperado_liquido.csv", header=["retorno_liquido"])

    carteiras = simular_carteiras(media_bruta, media_liquida, cov, rf, rf_bruta)

    melhor_liq = carteiras.loc[carteiras["sharpe"].idxmax()]
    print("Carteira de máximo Sharpe (LÍQUIDO):")
    print(f"  retorno líquido {melhor_liq['retorno']:.2%} (bruto {melhor_liq['retorno_bruto']:.2%}) "
          f"| risco {melhor_liq['risco']:.2%} | sharpe {melhor_liq['sharpe']:.2f}")
    print((melhor_liq[media_liquida.index] * 100).round(1).sort_values(ascending=False).to_string())

    melhor_bru = carteiras.loc[carteiras["sharpe_bruto"].idxmax()]
    print("\nCarteira de máximo Sharpe (BRUTO, antes do imposto):")
    print(f"  retorno bruto {melhor_bru['retorno_bruto']:.2%} (líquido {melhor_bru['retorno']:.2%}) "
          f"| risco {melhor_bru['risco']:.2%} | sharpe {melhor_bru['sharpe_bruto']:.2f}")
    print((melhor_bru[media_liquida.index] * 100).round(1).sort_values(ascending=False).to_string())

    plotar_fronteira(carteiras, PASTA_RESULTADOS / "fronteira_eficiente_liquido.png")
    plotar_fronteira(carteiras, PASTA_RESULTADOS / "fronteira_eficiente_bruto.png",
                     coluna_retorno="retorno_bruto", coluna_sharpe="sharpe_bruto", base="bruto")
    plotar_comparacao(carteiras, PASTA_RESULTADOS / "comparacao_bruto_liquido.png")
    print(f"\nResultados salvos em: {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()