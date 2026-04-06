import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from utils import TrueDEQSpring, neural_control_rhc, plot_neural_control
from utils import plot_results

# load model from check point
ckpt_path = Path(__file__).resolve().parent / "checkpoints" / "true_deq_spring.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

cfg = ckpt.get("config", {})
model = TrueDEQSpring(
    hidden=cfg.get("hidden", 64),
    max_iter=cfg.get("max_iter", 80),
    tol=cfg.get("tol", 1e-8),
    eps_min=cfg.get("eps_min", 0.0),
    eps_max=cfg.get("eps_max", 2.0),
)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

history = ckpt.get("history", [])
data_expt = ckpt.get("data_expt", None)
if data_expt is None:
    # Fallback: rebuild the dataset the same way as in test_DEQ.py
    xlsx_file_path = Path(__file__).resolve().parent / "content" / "Slinky_Data.xlsx"
    df_xlsx = pd.read_excel(xlsx_file_path, usecols=['Stretch (m)', 'Force (N) - zero adjusted'])
    data_expt = df_xlsx[['Stretch (m)', 'Force (N) - zero adjusted']].values.astype(np.float32)
    idx = np.where(data_expt[:, 1] > 2)
    data_expt = data_expt[idx]
    data_expt = data_expt - data_expt[0, :]
else:
    data_expt = np.array(data_expt, dtype=np.float32)

eps_max = data_expt[:, 0].max() # Identify the maximum stretch

plot_results(model, history, data_expt, eps_min=0.0, eps_max=1.0, num=400)



# --- Stage 2: neural control ---
# Target: eps*(lambda) = 0.5*sin(2*pi*lambda) + 0.5
# 1 full periods over [0,1], strain stays in [0, 1]
def eps_target_fn(lam):
    return 0.0125 * np.sin(2.0 * np.pi * lam) + 0.0125

# Initial condition
eps0   = torch.tensor([[eps_target_fn(0.0)]], dtype=torch.float32)
z0_val = float(model.force_value(eps0).item())

eps_max_target = torch.tensor([[0.05]], dtype=torch.float32)
f_max_needed   = float(model.force_value(eps_max_target).item())
# u_max should allow reaching f_max_needed over ~quarter period (K/4 steps)
K = 400 # Number of computation steps
u_max = f_max_needed * K / (K // 4)   # = f_max_needed * 4

u_max = 30

lam_grid, eps_exec, z_exec_arr, eps_tgt, seg_losses = neural_control_rhc(
    spring        = model,
    eps_target_fn = eps_target_fn,
    K             = K,
    H             = 10,
    z0_val        = z0_val,
    n_seg_steps   = 50,
    lr_ctrl       = 1e-2,
    u_max         = u_max,
    verbose       = True,
)

plot_neural_control(lam_grid, eps_exec, z_exec_arr, eps_tgt, seg_losses)