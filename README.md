# PI-MARKOWITZ

Fronteira Eficiente de Markowitz aplicada a ativos negociados na B3, em três versões: **bruta (log)**, **aritmética** e **líquida de impostos**. Projeto integrador do curso de Ciência de Dados (Fatec Rubens Lara).

> Última atualização: 01/11/2026. Os resultados da versão líquida são **provisórios** até a base de proventos das ações ser refeita (veja [Pendências](#pendências)).

---

## 1. Objetivo

Comparar o desempenho de carteiras antes e depois da tributação, usando dados oficiais da B3. A versão líquida segue o documento **"PI - Ativos"**, que define, para cada classe de ativo, como calcular o retorno bruto e o retorno após impostos.

## 2. Ativos e período

| Classe | Ativos |
|---|---|
| Ações | PETR4, ITUB4, B3SA3, AXIA3 (antiga ELET3), EMBJ3 (antiga EMBR3) |
| FIIs | MXRF11, XPML11, HGLG11, KNRI11, TGAR11 |
| ETFs | BOVA11, SMAL11, IVVB11, GOLD11, DIVO11 |
| Renda fixa | Tesouro Selic (entra só na versão líquida) |

- LCI e LCA estão previstas no PDF, mas **ainda não foram incluídas** (não são negociadas na B3 e não existem no COTAHIST).
- Período: 2020 a 2025. A **amostra comum** vai de **15/12/2020 a 30/12/2025**, porque o GOLD11 só passa a ter preço em 14/12/2020 e a covariância exige datas alinhadas.

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
│   ├── markowit_calculo_liquido(versao_01).py   # retorno total líquido de impostos
│   ├── baixar_dados.py                   # LEGADO (Yahoo); montar_proventos_b3 ainda importa constantes dele
│   ├── completar_fiis.py                 # LEGADO; montar_proventos_b3 ainda importa funções dele
│   └── teste.py                          # scripts soltos de conferência
├── dados/
│   ├── dados_brutos/                     # COTAHIST_A*.TXT (B3)
│   ├── dados_zipados/
│   ├── dados_limpos/                     # cotahist_limpo.csv, proventos.csv
│   ├── proventos_b3_manual.csv           # proventos coletados à mão (única fonte dos proventos)
│   └── selic_diaria.csv
└── resultados/                           # bruto e aritmético; líquido em resultados/liquido/
```

## 4. Fontes de dados

| Dado | Fonte | Observação |
|---|---|---|
| Preços | **B3**, arquivos COTAHIST anuais | Apenas mercado à vista (`TPMERC = 010`) |
| Selic diária | **Banco Central** (SGS, série 11) | % ao dia; salva em decimal (`selic_dia`) |
| Proventos | **B3** (últimos 12 meses) + **Status Invest** (histórico) | Coluna `fonte` registra a origem de cada linha |

Decisão do projeto: **não usar o Yahoo Finance**. A B3 é a fonte primária. O Status Invest, que é um agregador de terceiros, só completa os meses antigos que a tela de Eventos Corporativos da B3 não mostra (ela guarda só os últimos 12 meses).

## 5. Como rodar (ordem)

Dependências: `pandas`, `numpy`, `matplotlib`, `polars`, `requests`.

```
python codigos/limpeza.py                  # 1. gera dados/dados_limpos/cotahist_limpo.csv
python codigos/baixar_selic.py             # 2. gera dados/selic_diaria.csv
python codigos/montar_proventos_b3.py      # 3. gera dados/dados_limpos/proventos.csv
python codigos/validar_proventos.py        # 4. confere o manual
python codigos/markowitz_calculo_bruto.py
python codigos/markowitz_calculo_aritmetico.py
python codigos/markowit_calculo_liquido(versao_01).py
```

Não rode o `baixar_dados.py`: ele sobrescreve o `proventos.csv` com dados do Yahoo.

## 6. Metodologia

**Retornos.** Série de preços diários, só nos dias em que todos os ativos têm preço.

| Versão | Retorno diário | Proventos | Imposto |
|---|---|---|---|
| Bruta | logarítmico, `ln(P_t / P_t-1)` | não | não |
| Aritmética | simples, `P_t / P_t-1 - 1` | não | não |
| Líquida | total, `(P_t + D_t) / P_t-1 - 1` | sim | sim |

Em todas: retorno esperado = média diária × 252; covariância = covariância diária × 252. Carteiras: 2.000.000 de pesos sorteados (distribuição de Dirichlet, sem venda a descoberto, semente 42). Sharpe = (retorno − rf) / risco. Marcados no gráfico: carteira de máximo Sharpe e de mínima variância.

Para o Markowitz, o retorno simples é o mais adequado: o retorno da carteira é a soma ponderada dos retornos simples. Por período, `média aritmética ≈ média log + σ²/2`, então a versão aritmética sempre mostra retornos esperados maiores que a logarítmica, e a diferença cresce com a volatilidade.

**Taxa livre de risco (rf).** Selic composta no mesmo período da amostra de retornos (`taxa_livre_risco.py`). Chave `RF_USA_SELIC_LIQUIDA`: `False` usa a Selic bruta, `True` usa o retorno líquido do Tesouro Selic.

**Impostos modelados (pessoa física, conforme o código; confirmar a legislação antes de uso oficial).**

| Item | Regra no código |
|---|---|
| Ganho de capital, ações e ETFs | 15% sobre o ganho, na venda no fim do horizonte |
| Ganho de capital, FIIs | 20% |
| Isenção de R$ 20 mil/mês nas ações | desativada (cenário conservador) |
| Dividendos | isentos |
| JCP | 15% até 31/12/2025; 17,5% a partir de 01/01/2026 |
| Rendimento de FII | isento |
| Tesouro Selic | IR regressivo (22,5% / 20% / 17,5% / 15%), IOF regressivo nos primeiros 30 dias, custódia B3 de 0,20% a.a. acima de R$ 10 mil |

**Correções feitas em relação ao PDF "PI - Ativos".**
1. Retorno total: o provento entra no numerador e o divisor é o preço do dia anterior. Somar o provento ao preço só no dia `t` faria o retorno "devolver" o ganho no dia seguinte.
2. Imposto de ganho de capital: nunca negativo (sem ganho, não há imposto) e distribuído como desconto anual no retorno esperado, em vez de lançado inteiro no último dia (o que criaria um retorno diário absurdo e destruiria a variância). A base é o valor final, não o preço inicial.
3. Tesouro Selic: IOF e IR não se somam (o IR incide sobre o ganho já descontado do IOF), as alíquotas dependem do prazo, e os custos em reais são convertidos em fração do valor aplicado.
4. A covariância é calculada sobre os retornos **brutos**, como no PDF.

## 7. Histórico do que foi feito

1. **Entendimento do código base.** Explicação didática de `limpeza.py` (leitura posicional do COTAHIST com polars) e de `markowitz_calculo_bruto.py` (pandas, numpy, matplotlib).
2. **Limpeza do COTAHIST.** Leitura do layout oficial da B3, filtro dos 15 tickers e do mercado à vista. O mercado fracionário (`TPMERC 020`) usa o mesmo ticker no mesmo dia e duplicaria as linhas.
3. **Markowitz bruto (log).** Primeira fronteira eficiente, com simulação matricial em vez de laço.
4. **Ajustes de dados.** Renomeação de tickers (ELET3 → AXIA3, EMBR3 → EMBJ3); desdobramento 1 para 3 da B3SA3 em 17/05/2021 (a B3 não ajusta preços antigos); detecção de saltos diários acima de 25%.
5. **Markowitz aritmético.** Mesma estrutura com retornos simples e verificações (saltos, retornos acima de 50%, NaN, pesos que somam 1).
6. **Versão líquida.** Implementação do PDF "PI - Ativos", com as correções da seção 6, proventos por tipo e alíquotas por classe. Resultados em `resultados/liquido/`.
7. **Selic.** `baixar_selic.py` baixou 1.507 dias (02/01/2020 a 31/12/2025). Selic acumulada de 72,80% no período, 9,58% ao ano. Conferência por ano bateu com a Selic efetiva (2020: 2,76%; 2021: 4,42%; 2022: 12,39%; 2023: 13,04%; 2024: 10,88%; 2025: 14,32%).
8. **Taxa livre de risco.** `taxa_livre_risco.py` ligado ao Sharpe do líquido, sobre o mesmo período da amostra.
9. **Proventos.** A primeira tentativa usou o Yahoo Finance (`baixar_dados.py`), descartada por ser dado de terceiros e incompleta. A base passou a ser a B3, com o Status Invest para os meses antigos. O `proventos_b3_manual.csv` cobre os 5 FIIs de jan/2020 a set/2026.
10. **Diagnósticos.** Comparação entre as fronteiras aritmética e líquida mostrou que o ganho do líquido vem dos proventos e do Tesouro Selic, e que o imposto só subtrai (efeito entre 0 e ~2,7 p.p. ao ano). Esse teste também revelou os problemas de proventos descritos abaixo.

## 8. Problemas encontrados

- **`proventos.csv` vindo do Yahoo.** Os FIIs só tinham pagamentos a partir de mar/abr-2022 (44 a 46 por fundo, contra ~60 esperados na amostra), subestimando seus retornos.
- **Tipo de provento das ações é uma suposição.** O Yahoo não separa dividendo de JCP. O `baixar_dados.py` divide cada pagamento em 50% dividendo e 50% JCP (`FRACAO_JCP_PADRAO = 0.5`). O total pago está correto, mas o imposto sobre as ações é aproximado, e não se sabe se o valor do JCP vem bruto ou líquido.
- **O `montar_proventos_b3.py` não lia o manual** (esperava `codneg`, `tipo`, `valor_bruto`; o arquivo tem `ticker`, `valor_por_cota`, sem `tipo`). Corrigido em `ler_b3()`.
- **Risco de ajustar o desdobramento da B3SA3 duas vezes** nos proventos (uma no `montar`, outra no cálculo líquido). A chave `PROVENTOS_JA_AJUSTADOS` está documentada no `montar`, mas ainda não existe no cálculo líquido.
- **Markowitz bruto sem ajuste do desdobramento** da B3SA3: o dia 17/05/2021 vira um retorno de cerca de −110% e distorce média e covariância.
- **Constantes duplicadas** (`CLASSE_ATIVO`, `EVENTOS_CORPORATIVOS`, `RENOMEAR_TICKERS`) em mais de um arquivo.

## 9. Pendências

**Dados**
- [ ] Incluir no `proventos_b3_manual.csv` os proventos das 5 ações, com a coluna `tipo` (`Dividendo` ou `JCP`), valores **brutos**, de dez/2020 em diante.
- [ ] Rodar `montar_proventos_b3.py` e `validar_proventos.py`; conferir a contagem de pagamentos por ativo (~60 por FII na amostra).
- [ ] Conferir se o Status Invest mostra os proventos da B3SA3 anteriores a 17/05/2021 no valor da época, porque o `montar` os divide por 3.
- [ ] Verificar se parte dos pagamentos semestrais do HGLG11 (R$ 1,50 a 3,30) é ganho de capital, tributado a 20% para pessoa física (hoje tratados como rendimento isento).
- [ ] Conferir por amostragem alguns meses do Status Invest contra a B3.

**Código**
- [ ] Adicionar `PROVENTOS_JA_AJUSTADOS = True` ao cálculo líquido e só chamar `ajustar_proventos_eventos` quando for `False`.
- [ ] Criar um `config.py` com `CLASSE_ATIVO`, `EVENTOS_CORPORATIVOS` e `RENOMEAR_TICKERS`, usado por todos os scripts; depois arquivar `baixar_dados.py` e `completar_fiis.py` e apagar `proventos_yahoo.csv`.
- [ ] Renomear `markowit_calculo_liquido(versao_01).py` para `markowitz_calculo_liquido.py` (os parênteses impedem `import`).
- [ ] Corrigir o bruto e o aritmético: ajuste do desdobramento no bruto, remover a `plotar_fronteira` duplicada do aritmético, usar a Selic como rf nos dois.
- [ ] Ajustar o `validar_proventos.py` para contar repetições por `ticker` + `tipo` (ações pagam dividendo e JCP no mesmo mês).

**Análise**
- [ ] Rodar de novo o líquido com os proventos corrigidos e refazer a tabela `efeito_proventos` / `efeito_imposto`.
- [ ] Gráfico que sobrepõe as fronteiras bruta e líquida (mesmos pesos e mesmo risco, só o imposto desce o retorno).
- [ ] Decidir se o rf do Sharpe será a Selic bruta ou a líquida.
- [ ] Incluir LCI e LCA.

## 10. Limitações conhecidas

- O retorno esperado é a média histórica anualizada: carteiras de máximo Sharpe tendem a concentrar nos ativos que mais subiram no período (por exemplo EMBJ3 e GOLD11).
- Fronteira obtida por sorteio de carteiras, não por otimização; aproxima a fronteira, mas não a encontra exatamente.
- O imposto de ganho de capital supõe venda no fim do horizonte, sem vendas intermediárias nem compensação de prejuízos.
- Os valores do MXRF11 (cerca de R$ 0,10 por cota) estão arredondados a R$ 0,01 em boa parte da série, o que pode distorcer o rendimento de alguns meses.
- Valores de proventos dos FIIs, em parte do período, vêm do Status Invest (terceiro), e não da B3.
