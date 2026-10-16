"""
Fronteira Eficiente de Markowitz a partir do cotahist_limpo.csv (PI-MARKOWITZ).

Fluxo:
    CSV limpo -> tabela de preços (1 coluna por ativo) -> retornos log diários
    -> retorno esperado + matriz de covariância (anualizados)
    -> simulação de carteiras aleatórias -> fronteira eficiente

Adaptado da lógica do artigo "Fronteira Eficiente de Markowitz" (Edsson Alonso),
trocando yfinance pelos dados da B3 e a simulação em loop por operações matriciais.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ARQUIVO_ENTRADA = BASE_DIR / "dados" / "dados_limpos" / "cotahist_limpo.csv"
PASTA_RESULTADOS = BASE_DIR / "resultados"

DIAS_UTEIS_ANO = 252      # o artigo usa 246; o importante é ser consistente
N_CARTEIRAS = 2_000_000
TAXA_LIVRE_RISCO = 0.0    # ao ano (ex.: 0.10 = 10%). Aqui entraria o CDI/LCI
SEMENTE = 42              # fixa o sorteio para o resultado ser reproduzível


# ----------------------------------------------------------------------
# 1. Carregar e montar a tabela de preços
# ----------------------------------------------------------------------

def carregar_precos(caminho: Path) -> pd.DataFrame:
    """Lê o CSV limpo e devolve preços de fechamento: linhas = datas, colunas = tickers."""
    df = pd.read_csv(
        caminho,
        parse_dates=["data_pregao"],
        dtype={"tpmerc": str, "codbdi": str},   # preserva o zero à esquerda ("010")
    )

    RENOMEAR_TICKERS = {
    "ELET3": "AXIA3",   # código antigo -> código atual de cada ativo
    "EMBR3": "EMBJ3",
    }

    df["codneg"] = df["codneg"].replace(RENOMEAR_TICKERS)   #isso faz a mudança dos nomes dos ativos


    # pivot (e não pivot_table) dá erro se houver ticker repetido no mesmo dia,
    # o que serve de alarme caso o filtro de mercado à vista tenha falhado.
    precos = df.pivot(index="data_pregao", columns="codneg", values="preult")
    return precos.sort_index()


def diagnosticar_historico(precos: pd.DataFrame) -> pd.DataFrame:
    """Mostra desde quando cada ativo tem preço. Ativos novos encurtam a amostra comum."""
    return pd.DataFrame({
        "primeira_data": precos.apply(pd.Series.first_valid_index),
        "ultima_data": precos.apply(pd.Series.last_valid_index),
        "dias_com_preco": precos.notna().sum(),
    }).sort_values("primeira_data")


# ----------------------------------------------------------------------
# 2. Retornos, retorno esperado e covariância
# ----------------------------------------------------------------------

def calcular_retornos_log(precos: pd.DataFrame) -> pd.DataFrame:
    """Retorno logarítmico diário: ln(preço_hoje / preço_ontem).
    Usa só os dias em que TODOS os ativos têm preço (covariância exige datas alinhadas)."""
    precos_comuns = precos.dropna()
    return np.log(precos_comuns / precos_comuns.shift(1)).dropna()


def estatisticas_anuais(retornos: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Retorno esperado (média) e covariância, ambos anualizados."""
    media = retornos.mean() * DIAS_UTEIS_ANO # media de todos os retornos 
    cov = retornos.cov() * DIAS_UTEIS_ANO # cov é covariancia
    return media, cov


# ----------------------------------------------------------------------
# 3. Carteiras
# ----------------------------------------------------------------------

def desempenho_carteiras(pesos: np.ndarray, media: pd.Series, cov: pd.DataFrame):
    """Retorno e risco de várias carteiras de uma vez.
    pesos: matriz (n_carteiras, n_ativos). Risco = raiz(w' · Σ · w)."""
    retornos = pesos @ media.to_numpy()
    variancias = np.einsum("ij,jk,ik->i", pesos, cov.to_numpy(), pesos)
    return retornos, np.sqrt(variancias)


def simular_carteiras(media: pd.Series, cov: pd.DataFrame, n_carteiras: int = N_CARTEIRAS) -> pd.DataFrame:
    """Sorteia carteiras aleatórias (pesos >= 0 somando 1, sem venda a descoberto)."""
    rng = np.random.default_rng(SEMENTE)
    pesos = rng.dirichlet(np.ones(len(media)), size=n_carteiras)
    retorno, risco = desempenho_carteiras(pesos, media, cov)

    resultado = pd.DataFrame(pesos, columns=media.index)
    resultado["retorno"] = retorno
    resultado["risco"] = risco
    resultado["sharpe"] = (retorno - TAXA_LIVRE_RISCO) / risco
    return resultado


# ----------------------------------------------------------------------
# 4. Gráfico
# ----------------------------------------------------------------------

def plotar_fronteira(carteiras: pd.DataFrame, caminho_saida: Path) -> None:
    melhor_sharpe = carteiras.loc[carteiras["sharpe"].idxmax()]
    menor_risco = carteiras.loc[carteiras["risco"].idxmin()]

    fig, ax = plt.subplots(figsize=(10, 6))
    pontos = ax.scatter(carteiras["risco"], carteiras["retorno"],
                        c=carteiras["sharpe"], cmap="viridis", s=6, alpha=0.6)
    ax.scatter(melhor_sharpe["risco"], melhor_sharpe["retorno"],
               marker="*", s=300, c="red", label="Máximo Sharpe")
    ax.scatter(menor_risco["risco"], menor_risco["retorno"],
               marker="D", s=100, c="orange", label="Mínima variância")
    fig.colorbar(pontos, label="Índice de Sharpe")
    ax.set_xlabel("Risco (desvio padrão anual)")
    ax.set_ylabel("Retorno esperado anual")
    ax.set_title("Fronteira Eficiente de Markowitz")
    ax.legend()
    fig.tight_layout()

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho_saida, dpi=150)


def main() -> None:
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    precos = carregar_precos(ARQUIVO_ENTRADA)
    print("Histórico por ativo:")
    print(diagnosticar_historico(precos), "\n")

    retornos = calcular_retornos_log(precos)
    print(f"Amostra comum: {retornos.index.min().date()} a {retornos.index.max().date()} "
          f"({len(retornos)} retornos diários)\n")

    media, cov = estatisticas_anuais(retornos)
    cov.to_csv(PASTA_RESULTADOS / "matriz_covariancia.csv")
    media.to_csv(PASTA_RESULTADOS / "retorno_esperado.csv", header=["retorno_esperado"])
    print("Matriz de covariância anualizada:")
    print(cov.round(4), "\n")

    carteiras = simular_carteiras(media, cov)
    melhor = carteiras.loc[carteiras["sharpe"].idxmax()]
    print("Carteira de máximo Sharpe:")
    print(f"  retorno {melhor['retorno']:.2%} | risco {melhor['risco']:.2%} | sharpe {melhor['sharpe']:.2f}")
    print((melhor[media.index] * 100).round(1).sort_values(ascending=False).to_string())

    plotar_fronteira(carteiras, PASTA_RESULTADOS / "fronteira_eficiente.png")
    print(f"\nResultados salvos em: {PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()