import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path



# Load the XLSX file from the /content folder, selecting only the desired columns
xlsx_file_path = 'content/Slinky_Data.xlsx'
df_xlsx = pd.read_excel(xlsx_file_path, usecols=['Stretch (m)', 'Force (N) - zero adjusted'])
data_expt = df_xlsx[['Stretch (m)', 'Force (N) - zero adjusted']].values.astype(np.float32)

idx = np.where(data_expt[:, 1] > 2)
data_expt = data_expt[idx]

data_expt = data_expt - data_expt[0, :]
data_expt[:, 1] = data_expt[:, 1] 
# data_expt[:, 0] = data_expt[:, 0] * 100
# data_expt[:, 1] = data_expt[:, 1] * 100

plt.plot(data_expt[:,0], data_expt[:, 1])
plt.show()

print("Shape of the array:", data_expt.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
from utils import ConvexEnergyMLP, train_true_deq, validate, plot_results

# --- Stage 1: train DEQ energy model ---
eps_max = data_expt[:, 0].max() # Identify the maximum stretch
eps_max = 0.06

model, history = train_true_deq(
    data_np=data_expt, hidden=64, lr=1e-2, epochs=3000,
    max_iter=80, tol=1e-8, eps_min=0.0, eps_max=eps_max,
    reg=1e-8, verbose=True,
)

# Save a reproducible checkpoint (recommended over saving the full module object).
ckpt = {
    "model_state_dict": model.state_dict(),
    "history": history,
    "data_expt": data_expt,
    "config": {
        "hidden": 64,
        "lr": 1e-2,
        "epochs": 3000,
        "max_iter": 80,
        "tol": 1e-8,
        "eps_min": 0.0,
        "eps_max": float(eps_max),
        "reg": 1e-8,
    },
}
ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / "true_deq_spring.pt"
torch.save(ckpt, ckpt_path)
print(f"Saved checkpoint to: {ckpt_path}")

# validate(model, data_expt)
plot_results(model, history, data_expt, eps_min=0.0, eps_max=eps_max, num=400)