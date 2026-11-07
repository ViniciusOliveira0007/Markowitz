# PI-MARKOWITZ

Fronteira Eficiente de Markowitz aplicada a ativos negociados na B3, em três versões: **bruta (log)**, **aritmética** e **líquida de impostos**. Projeto integrador do curso de Ciência de Dados (Fatec Rubens Lara, Centro Paula Souza).

> **Status: concluído** (novembro de 2026).

---

## 1. Objetivo

Comparar o desempenho de carteiras antes e depois da tributação, usando dados oficiais da B3. A versão líquida segue o documento **"PI - Ativos"**, que define, para cada classe de ativo, como calcular o retorno bruto e o retorno após impostos.

## 2. Ativos e período

| Classe | Ativos |
|---|---|
| Ações | PETR4, ITUB4, B3SA3, AXIA3 (antiga ELET3), EMBJ3 (antiga EMBR3) |
| FIIs | MXRF11, XPML11, HGLG11, KNRI11, TGAR11 |
| ETFs | BOVA11, SMAL11, IVVB11, GOLD11, DIVO11 |
| Renda fixa | Tesouro Selic e LCI/LCA (entram só na versão líquida) |

- Período: 2020 a 2025. A **amostra comum** vai de **15/12/2020 a 30/12/2025**, porque o GOLD11 só passa a ter preço em 14/12/2020 e a covariância exige datas alinhadas.
- **LCI/LCA** não existem no COTAHIST. Elas são modeladas como **88% do CDI**, isentas de IR para pessoa física, usando a Selic diária como aproximação do CDI. A taxa fixa é uma premissa: o valor de hoje é aplicado em toda a amostra.

## 3. Estrutura de pastas

```
PI-MARKOWITZ/
├── codigos/
│   ├── limpeza.py                        # COTAHIST bruto -> cotahist_limpo.csv
│   ├── baixar_selic.py                   # Selic diária (BCB, série 11)
│   ├── taxa_livre_risco.py               # Selic acumulada e anualizada (rf do Sharpe)
│   ├── montar_proventos_b3.py            # manual (B3 + Status Invest) -> proventos.csv
│   ├── validar_proventos.py              # conferência do manual
│   ├── markowitz_calculo_bruto.py        # retornos logarítmicos
│   ├── markowitz_calculo_aritmetico.py   # retornos simples
│   └── markowitz_calculo_liquido.py      # retorno total líquido de impostos
├── dados/
│   ├── dados_brutos/                     # COTAHIST_A*.TXT (B3)
│   ├── dados_zipados/
│   ├── dados_limpos/                     # cotahist_limpo.csv, proventos.csv
│   ├── proventos_b3_manual.csv           # proventos coletados à mão (FIIs e ações)
│   └── selic_diaria.csv
└── resultados/                           # bruto e aritmético; líquido em resultados/liquido/
```

## 4. Fontes de dados

| Dado | Fonte | Observação |
|---|---|---|
| Preços | **B3**, arquivos COTAHIST anuais | Apenas mercado à vista (`TPMERC = 010`) |
| Selic diária | **Banco Central** (SGS, série 11) | % ao dia; salva em decimal (`selic_dia`) |
| Proventos | **B3** (últimos 12 meses) + **Status Invest** (histórico) | Coluna `fonte` registra a origem de cada linha; coluna `tipo` (Dividendo, JCP) nas ações |

Decisão do projeto: **não usar o Yahoo Finance**. A B3 é a fonte primária. O Status Invest, agregador de terceiros, só completa os meses antigos que a tela de Eventos Corporativos da B3 não mostra (ela guarda só os últimos 12 meses).

## 5. Como rodar (ordem)

Dependências: `pandas`, `numpy`, `matplotlib`, `polars`, `requests`.

```
python codigos/limpeza.py                  # 1. gera dados/dados_limpos/cotahist_limpo.csv
python codigos/baixar_selic.py             # 2. gera dados/selic_diaria.csv
python codigos/montar_proventos_b3.py      # 3. gera dados/dados_limpos/proventos.csv
python codigos/validar_proventos.py        # 4. confere o manual
python codigos/markowitz_calculo_bruto.py
python codigos/markowitz_calculo_aritmetico.py
python codigos/markowitz_calculo_liquido.py
```

## 6. Metodologia

**Retornos.** Série de preços diários, só nos dias em que todos os ativos têm preço.

| Versão | Retorno diário | Proventos | Imposto |
|---|---|---|---|
| Bruta | logarítmico, `ln(P_t / P_t-1)` | não | não |
| Aritmética | simples, `P_t / P_t-1 - 1` | não | não |
| Líquida | total, `(P_t + D_t) / P_t-1 - 1` | sim | sim |

Em todas: retorno esperado = média diária × 252; covariância = covariância diária × 252. Carteiras: 2.000.000 de pesos sorteados (distribuição de Dirichlet, sem venda a descoberto, semente 42). Sharpe = (retorno − rf) / risco. Nos gráficos são marcadas a carteira de **máximo Sharpe** e a de **mínima variância**. Os gráficos desenham uma amostra de 400.000 pontos; os cálculos usam todas as carteiras.

Para o Markowitz, o retorno simples é o mais adequado: o retorno da carteira é a soma ponderada dos retornos simples. Por período, `média aritmética ≈ média log + σ²/2`, então a versão aritmética sempre mostra retornos esperados maiores que a logarítmica, e a diferença cresce com a volatilidade.

**Taxa livre de risco (rf).** Selic composta no mesmo período da amostra de retornos (`taxa_livre_risco.py`): 10,90% a.a. na amostra comum. Chave `RF_USA_SELIC_LIQUIDA`: `False` usa a Selic bruta (padrão), `True` usa o retorno líquido do Tesouro Selic.

**Impostos modelados (pessoa física; confirmar a legislação antes de uso oficial).**

| Item | Regra no código |
|---|---|
| Ganho de capital, ações e ETFs | 15% sobre o ganho, na venda no fim do horizonte |
| Ganho de capital, FIIs | 20% |
| Isenção de R$ 20 mil/mês nas ações | desativada (cenário conservador) |
| Dividendos | isentos |
| JCP | 15% até 31/12/2025; 17,5% a partir de 01/01/2026 |
| Rendimento de FII | isento |
| Tesouro Selic | IR regressivo (22,5% / 20% / 17,5% / 15%), IOF regressivo nos primeiros 30 dias, custódia B3 de 0,20% a.a. acima de R$ 10 mil |
| LCI/LCA | isentas de IR e IOF, sem custódia; 88% do CDI |

**Correções feitas em relação ao PDF "PI - Ativos".**
1. Retorno total: o provento entra no numerador e o divisor é o preço do dia anterior. Somar o provento ao preço só no dia `t` faria o retorno "devolver" o ganho no dia seguinte.
2. Imposto de ganho de capital: nunca negativo (sem ganho, não há imposto) e distribuído como desconto anual no retorno esperado, em vez de lançado inteiro no último dia (o que criaria um retorno diário absurdo e destruiria a variância). A base é o valor final, não o preço inicial.
3. Tesouro Selic: IOF e IR não se somam (o IR incide sobre o ganho já descontado do IOF), as alíquotas dependem do prazo, e os custos em reais são convertidos em fração do valor aplicado.
4. A covariância é calculada sobre os retornos **brutos**, como no PDF. Por isso o risco de uma carteira é igual nas bases bruta e líquida; o imposto só desloca o retorno.

## 7. Resultados

Resultados da versão líquida em `resultados/liquido/`:

| Arquivo | Conteúdo |
|---|---|
| `fronteira_eficiente_bruto.png` | fronteira com retorno bruto |
| `fronteira_eficiente_liquido.png` | fronteira com retorno líquido |
| `comparacao_bruto_liquido.png` | as duas nuvens sobrepostas, com máximo Sharpe e mínima variância de cada base |
| `retorno_bruto_vs_liquido.csv` | retorno esperado por ativo, nas duas bases |
| `retorno_esperado_liquido.csv`, `matriz_covariancia_liquido.csv` | insumos da simulação |

Leitura do gráfico de comparação: cada carteira aparece duas vezes, **no mesmo risco**. A nuvem cinza (bruto) fica acima da azul (líquido) e a distância vertical é o imposto. A carteira de máximo Sharpe fica em torno de 13% de risco, com retorno de cerca de 26,5% a.a. bruto e 25% a.a. líquido (valores lidos do gráfico). A de mínima variância fica na borda esquerda da nuvem, perto de 3,5% a 4% de risco, onde o efeito do imposto é pequeno.

Conclusões:
- O ganho da versão líquida em relação à aritmética vem dos **proventos** e do **Tesouro Selic / LCI-LCA**.
- O imposto só subtrai: reduz o retorno esperado entre 0 e cerca de 2,7 p.p. ao ano, dependendo da carteira.

## 8. Histórico do projeto

1. **Entendimento do código base.** Explicação didática de `limpeza.py` (leitura posicional do COTAHIST com polars) e de `markowitz_calculo_bruto.py` (pandas, numpy, matplotlib).
2. **Limpeza do COTAHIST.** Leitura do layout oficial da B3, filtro dos 15 tickers e do mercado à vista. O mercado fracionário (`TPMERC 020`) usa o mesmo ticker no mesmo dia e duplicaria as linhas.
3. **Markowitz bruto (log).** Primeira fronteira eficiente, com simulação matricial em vez de laço.
4. **Ajustes de dados.** Renomeação de tickers (ELET3 → AXIA3, EMBR3 → EMBJ3); desdobramento 1 para 3 da B3SA3 em 17/05/2021; bonificações de ITUB4 (18/03/2025 e 26/12/2025) e AXIA3 (22/12/2025); detecção de saltos diários acima de 25%.
5. **Markowitz aritmético.** Mesma estrutura com retornos simples e verificações (saltos, retornos acima de 50%, NaN, pesos que somam 1).
6. **Versão líquida.** Implementação do PDF "PI - Ativos", com as correções da seção 6, proventos por tipo e alíquotas por classe.
7. **Selic.** `baixar_selic.py` baixou 1.507 dias (02/01/2020 a 31/12/2025). Selic acumulada de 72,80% no período, 9,58% ao ano. Conferência por ano bateu com a Selic efetiva (2020: 2,76%; 2021: 4,42%; 2022: 12,39%; 2023: 13,04%; 2024: 10,88%; 2025: 14,32%).
8. **Taxa livre de risco.** `taxa_livre_risco.py` ligado ao Sharpe do líquido, sobre o mesmo período da amostra.
9. **Proventos.** A primeira tentativa usou o Yahoo Finance, descartada por ser dado de terceiros e incompleta (FIIs só com pagamentos a partir de 2022; ações divididas em 50% dividendo e 50% JCP por suposição). A base passou a ser a B3, com o Status Invest para os meses antigos. O `proventos_b3_manual.csv` passou a cobrir os 5 FIIs (jan/2020 a set/2026) e as 5 ações, com a coluna `tipo`.
10. **Chave `PROVENTOS_JA_AJUSTADOS`.** Adicionada ao cálculo líquido para evitar ajustar o desdobramento da B3SA3 duas vezes nos proventos.
11. **LCI/LCA.** Incluídas na versão líquida, ao lado do Tesouro Selic.
12. **Gráfico de comparação.** Sobreposição das fronteiras bruta e líquida (mesmos pesos, mesmo risco), com máximo Sharpe e mínima variância nas duas bases.

## 9. Limitações conhecidas

- O retorno esperado é a média histórica anualizada: carteiras de máximo Sharpe tendem a concentrar nos ativos que mais subiram no período (por exemplo EMBJ3 e GOLD11).
- Fronteira obtida por sorteio de carteiras, não por otimização; aproxima a fronteira, mas não a encontra exatamente.
- O imposto de ganho de capital supõe venda no fim do horizonte, sem vendas intermediárias nem compensação de prejuízos.
- O percentual da LCI/LCA (88% do CDI) é fixo em toda a amostra, e a Selic efetiva serve de proxy do CDI.
- O imposto sobre ações depende da divisão entre dividendo e JCP informada no manual.
- Parte dos proventos dos FIIs vem do Status Invest (terceiro), e não da B3. Os valores do MXRF11 (cerca de R$ 0,10 por cota) estão arredondados a R$ 0,01 em boa parte da série, o que pode distorcer o rendimento de alguns meses.
- Alguns pagamentos semestrais do HGLG11 podem incluir ganho de capital (20% para pessoa física), hoje tratados como rendimento isento.
- Os scripts bruto e aritmético repetem constantes (`CLASSE_ATIVO`, `EVENTOS_CORPORATIVOS`, `RENOMEAR_TICKERS`); um `config.py` único seria o próximo passo de organização.

## 10. Autor

Vinícius Oliveira, estudante de Ciência de Dados na Fatec Rubens Lara.
[LinkedIn](https://linkedin.com/in/vinicius-oliveira0007) · [GitHub](https://github.com/ViniciusOliveira0007)