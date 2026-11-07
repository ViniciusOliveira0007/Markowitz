import pandas as pd
df = pd.read_csv("dados/dados_limpos/cotahist_limpo.csv", parse_dates=["data_pregao"])
df["codneg"] = df["codneg"].replace({"ELET3": "AXIA3", "EMBR3": "EMBJ3"})
p = df.pivot(index="data_pregao", columns="codneg", values="preult")
for (t, d), f in {("B3SA3", "2021-05-17"): 3, ("ITUB4", "2025-03-18"): 1.10,
                  ("ITUB4", "2025-12-26"): 1.03, ("AXIA3", "2025-12-22"): 1.2628}.items():
    d = pd.Timestamp(d)
    antes = p.loc[p.index < d, t].iloc[-1]
    depois = p.loc[p.index >= d, t].iloc[0]
    print(t, d.date(), f"{depois / (antes / f) - 1:+.2%}")