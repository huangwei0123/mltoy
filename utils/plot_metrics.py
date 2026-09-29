import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "forecasts/verification_metrics.csv"
)

plt.figure(figsize=(8,5))

plt.plot(
    df["lead_hour"],
    df["rmse"],
    marker="o"
)

plt.grid(True)

plt.xlabel("Lead Time (hours)")
plt.ylabel("RMSE (K)")
plt.title("RMSE vs Forecast Lead Time")

plt.savefig(
    "forecasts/rmse_vs_leadtime.png",
    dpi=150
)

plt.show()
plt.close()
