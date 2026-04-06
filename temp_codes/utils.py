import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

class ConvexEnergyMLP(nn.Module):
    """
    Convex 1D energy E(eps), with E(0)=0 and F(0)=0 enforced analytically.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.w1 = nn.Parameter(0.1 * torch.randn(hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2_raw = nn.Parameter(0.1 * torch.randn(hidden))

    def w2(self):
        return torch.nn.functional.softplus(self.w2_raw)

    def energy(self, eps, params=None):
        if params is None:
            w1, b1, w2_raw = self.w1, self.b1, self.w2_raw
        else:
            w1, b1, w2_raw = params

        w2 = torch.nn.functional.softplus(w2_raw)
        h = torch.nn.functional.softplus(eps * w1.view(1, -1) + b1.view(1, -1))
        h0 = torch.nn.functional.softplus(b1).view(1, -1)

        E = (h - h0) @ w2.view(-1, 1)

        F0 = (torch.sigmoid(b1) * w1 * w2).sum()
        E = E - F0 * eps
        return E

    def force(self, eps, params=None, create_graph=True):
        E = self.energy(eps, params=params)
        F = torch.autograd.grad(E.sum(), eps, create_graph=create_graph)[0]
        return F

    def force_value(self, eps):
        with torch.enable_grad():
            eps_req = eps.detach().requires_grad_(True)
            E = self.energy(eps_req)
            F = torch.autograd.grad(E.sum(), eps_req, create_graph=False)[0]
        return F.detach()

    def stiffness_value(self, eps):
        with torch.enable_grad():
            eps_req = eps.detach().requires_grad_(True)
            F = self.force(eps_req, create_graph=True)
            K = torch.autograd.grad(F.sum(), eps_req, create_graph=False)[0]
        return K.detach()


class EquilibriumSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, model, max_iter, tol, eps_min, eps_max, *params):
        with torch.no_grad():
            scale = w.abs().max().clamp_min(1.0)
            eps = torch.clamp(w / scale, eps_min, eps_max)

            for _ in range(max_iter):
                F = model.force_value(eps)
                G = F - w
                if G.abs().max().item() < tol:
                    break

                K = model.stiffness_value(eps).clamp_min(1e-8)
                step = G / K

                accepted = False
                base_norm = G.abs()

                for damp in (1.0, 0.5, 0.25, 0.1, 0.05):
                    eps_trial = torch.clamp(eps - damp * step, eps_min, eps_max)
                    G_trial = model.force_value(eps_trial) - w
                    if (G_trial.abs() <= base_norm + 1e-12).all():
                        eps = eps_trial
                        accepted = True
                        break

                if not accepted:
                    eps = torch.clamp(eps - 1e-4 * G, eps_min, eps_max)

        ctx.model = model
        ctx.max_iter = max_iter
        ctx.tol = tol
        ctx.eps_min = eps_min
        ctx.eps_max = eps_max
        ctx.save_for_backward(eps.detach(), w.detach(), *params)
        return eps.detach()

    @staticmethod
    def backward(ctx, grad_output):
        model = ctx.model
        saved = ctx.saved_tensors
        eps_star, w = saved[0], saved[1]
        params = saved[2:]

        with torch.enable_grad():
            eps = eps_star.detach().requires_grad_(True)
            params_req = tuple(p.detach().requires_grad_(True) for p in params)

            F = model.force(eps, params=params_req, create_graph=True)
            G = F - w

            dG_deps = torch.autograd.grad(G.sum(), eps, create_graph=True)[0]
            inv_dG_deps = 1.0 / (dG_deps + 1e-8)

            v = -grad_output * inv_dG_deps

            grad_params = torch.autograd.grad(
                outputs=G,
                inputs=params_req,
                grad_outputs=v,
                allow_unused=True,          # fix: was False
                retain_graph=False,
                create_graph=False,
            )
            grad_params = tuple(
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grad_params, params)
            )

            grad_w = grad_output * inv_dG_deps

        return (grad_w, None, None, None, None, None, *grad_params)


class TrueDEQSpring(nn.Module):
    def __init__(self, hidden=64, max_iter=80, tol=1e-8, eps_min=0.0, eps_max=2.0):
        super().__init__()
        self.energy_net = ConvexEnergyMLP(hidden=hidden)
        self.max_iter = max_iter
        self.tol = tol
        self.eps_min = eps_min
        self.eps_max = eps_max

    def forward(self, w):
        params = tuple(self.energy_net.parameters())
        return EquilibriumSolve.apply(
            w, self.energy_net, self.max_iter, self.tol,
            self.eps_min, self.eps_max, *params,
        )

    def energy(self, eps):          return self.energy_net.energy(eps)
    def force(self, eps, create_graph=True): return self.energy_net.force(eps, create_graph=create_graph)
    def force_value(self, eps):     return self.energy_net.force_value(eps)
    def stiffness_value(self, eps): return self.energy_net.stiffness_value(eps)
    def solve_equilibrium(self, w): return self.forward(w)


def train_true_deq(
    data_np, hidden=64, lr=1e-2, epochs=3000,
    max_iter=80, tol=1e-8, eps_min=0.0, eps_max=2.0,
    reg=1e-8, device="cpu", verbose=True,
):
    assert isinstance(data_np, np.ndarray)
    assert data_np.ndim == 2 and data_np.shape[1] == 2

    eps_data = torch.tensor(data_np[:, 0:1], dtype=torch.float32, device=device)
    w_data   = torch.tensor(data_np[:, 1:2], dtype=torch.float32, device=device)

    model = TrueDEQSpring(
        hidden=hidden, max_iter=max_iter, tol=tol,
        eps_min=eps_min, eps_max=eps_max,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        opt.zero_grad()
        eps_pred  = model(w_data)
        loss_data = ((eps_pred - eps_data) ** 2).mean()
        loss_reg  = reg * sum((p ** 2).sum() for p in model.parameters())
        loss      = loss_data + loss_reg
        loss.backward()
        opt.step()

        history.append(float(loss.detach()))
        if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d} | loss = {loss.item():.6e}")

    return model, history


def validate(model, data_np):
    eps_data = torch.tensor(data_np[:, 0:1], dtype=torch.float32)
    w_data   = torch.tensor(data_np[:, 1:2], dtype=torch.float32)

    with torch.no_grad():
        eps_rec = model.solve_equilibrium(w_data)
        F_rec   = model.force_value(eps_rec)
        K_rec   = model.stiffness_value(eps_rec)

    print("\nValidation:")
    for e_true, w_true, e_rec, f_rec, k_rec in zip(
        data_np[:, 0], data_np[:, 1],
        eps_rec.squeeze(), F_rec.squeeze(), K_rec.squeeze()
    ):
        print(
            f"true eps={e_true:7.4f} | w={w_true:9.4f} | "
            f"rec eps={float(e_rec):7.4f} | F(eps*)={float(f_rec):9.4f} | "
            f"K={float(k_rec):9.4f}"
        )


def plot_results(model, history, data_np, eps_min=0.0, eps_max=1.0, num=400):
    eps_plot = torch.linspace(eps_min, eps_max, num).view(-1, 1)
    with torch.no_grad():
        E_plot = model.energy(eps_plot).squeeze().cpu().numpy()
        F_plot = model.force_value(eps_plot).squeeze().cpu().numpy()
    eps_ax = eps_plot.squeeze().cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(eps_ax, E_plot, lw=2)
    axes[0].set_xlabel("Strain")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Learned energy")
    axes[0].grid(True)

    axes[1].plot(eps_ax, F_plot, lw=2, label="Learned force")
    axes[1].scatter(data_np[:, 0], data_np[:, 1], color="red", zorder=5, label="Data")
    axes[1].set_xlabel("Strain")
    axes[1].set_ylabel("Force")
    axes[1].set_title("Force-strain curve")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Neural Control additions
# ---------------------------------------------------------------------------

class NeuralController(nn.Module):
    """
    u_Theta(lambda): scalar lambda -> scalar force rate.
    Output bounded by u_max via tanh.
    """
    def __init__(self, hidden=64, u_max=500.0):
        super().__init__()
        self.u_max = u_max
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )

    def forward(self, lam):
        return self.u_max * self.net(lam)


def run_segment(spring, controller, z0, lam_start, H, dlam, eps_target_fn):
    """
    Forward rollout for one RHC segment (Section 3 + Appendix A).

    System:
        G(x, z) = F_theta(x) - z = 0,   x=eps (free), z=w (force, controlled)
    Continuation dynamics (Eq. 4):
        dz/dlambda = f(z, u) = u   =>   z_{k+1} = z_k + dlam * u_k
    Sensitivity (Eq. 5):
        S_k = dx/dz|_{eps*_k} = 1 / K_k   (1-D IFT, K_k = dF/deps)

    Returns
    -------
    x_hist      : realized strains (detached)
    z_hist      : forces at each step (detached)
    S_hist      : sensitivities 1/K_k (detached)
    u_hist      : controller outputs (retains grad for backward)
    lam_hist    : lambda values
    z_final     : force at end of segment (detached, for horizon shift)
    """
    z = z0.clone().detach()

    x_hist, z_hist, S_hist, u_hist, lam_hist = [], [], [], [], []

    for k in range(H):
        lam_k = lam_start + k * dlam
        lam_t = torch.tensor([[lam_k]], dtype=torch.float32)

        # Controller: force rate
        u_k = controller(lam_t)                                   # (1,1), keeps grad

        # Equilibrium solve -- no grad through solver (preprint Section 3)
        with torch.no_grad():
            x_k = spring.solve_equilibrium(z)                     # (1,1)
            K_k = spring.stiffness_value(x_k).clamp_min(1e-8)
            S_k = 1.0 / K_k    
                                               # S = dx/dz

        x_hist.append(x_k.detach())
        z_hist.append(z.detach())
        S_hist.append(S_k.detach())
        u_hist.append(u_k)
        lam_hist.append(lam_k)

        # Continuation update: z_{k+1} = z_k + dlam * u_k
        z = z.detach() + dlam * u_k

    return x_hist, z_hist, S_hist, u_hist, lam_hist, z.detach()


def proxy_adjoint_backward(x_hist, z_hist, S_hist, u_hist, lam_hist,
                            dlam, eps_target_fn):
    """
    Proxy-adjoint recursion (Appendix A, Eqs. 34-37).

    For f(z,u)=u:  A_k = df/dz = 0,  B_k = df/du = 1.
    Trajectory loss: L = (1/H) * sum_k ||x_k - eps*(lam_k)||^2

    Recursion (simplified by A_k=0):
        a_k = a_{k+1} + w_k * grad_x ell_k
        g_k = g_{k+1} + w_k * S_k * grad_x ell_k
    Control gradient (Eq. 36):
        dL/du_k = dlam * B_k^T * (g_{k+1} + S_k * a_{k+1})
                = dlam * (g_{k+1} + S_k * a_{k+1})
    """
    H  = len(x_hist)
    wk = 1.0 / H

    # Terminal: no terminal loss phi => a_H = 0, g_H = 0
    a = torch.zeros(1, 1)
    g = torch.zeros(1, 1)

    ctrl_grads = [None] * H

    for k in reversed(range(H)):
        x_k      = x_hist[k]
        S_k      = S_hist[k]
        eps_star = torch.tensor([[eps_target_fn(lam_hist[k])]], dtype=torch.float32)

        # grad_x ell_k = 2*(x_k - eps*_k) / H
        grad_x = 2.0 * (x_k - eps_star) * wk

        # Control gradient at step k (Eq. 36)
        ctrl_grads[k] = dlam * (g + S_k * a)

        # Backward recursion (Eqs. 34-35), A_k=0
        a = a + grad_x
        g = g + S_k * grad_x

    return ctrl_grads

# def proxy_adjoint_backward(x_hist, z_hist, S_hist, u_hist, lam_hist,
#                            dlam, eps_target_fn):
#     H = len(x_hist)
#     ctrl_grads = [None] * H

#     # p stores dL/dz_{k+1} during reverse sweep
#     p = torch.zeros_like(x_hist[0])

#     for k in reversed(range(H)):
#         # u_k changes z_{k+1}, so gradient uses future adjoint p
#         ctrl_grads[k] = dlam * p

#         eps_star = x_hist[k].new_tensor([[eps_target_fn(lam_hist[k])]])
#         grad_x = 2.0 * (x_hist[k] - eps_star) / H

#         # scalar case: S_k = dx/dz
#         p = p + S_hist[k] * grad_x

#     return ctrl_grads


def neural_control_rhc(spring, eps_target_fn,
                        K=400, H=10, z0_val=0.0,
                        n_seg_steps=100, lr_ctrl=1e-1, u_max=500.0,
                        verbose=True):
    assert K % H == 0, "K must be divisible by H"
    M    = K // H
    dlam = 1.0 / K

    for p in spring.parameters():
        p.requires_grad_(False)

    z_cur = torch.tensor([[z0_val]], dtype=torch.float32)

    eps_exec   = []
    z_exec_arr = []
    seg_losses = []

    # controller = NeuralController(hidden=64, u_max=u_max)
    # opt        = torch.optim.Adam(controller.parameters(), lr=lr_ctrl)

    # print(M, n_seg_steps)
    # exit(0)
    
    for m in range(M):
        k_start   = m * H
        lam_start = k_start / K

        controller = NeuralController(hidden=32, u_max=u_max)
        opt        = torch.optim.Adam(controller.parameters(), lr=lr_ctrl)

        best_loss  = float("inf")
        best_state = {k: v.clone() for k, v in controller.state_dict().items()}

        for step_counter in range(n_seg_steps):
            opt.zero_grad()

            x_hist, z_hist, S_hist, u_hist, lam_hist, _ = run_segment(
                spring, controller, z_cur, lam_start, H, dlam, eps_target_fn,
            )


            loss_val = sum(
                (x_hist[k] - torch.tensor([[eps_target_fn(lam_hist[k])]])).pow(2)
                for k in range(H)
            ).squeeze() / H

            ctrl_grads = proxy_adjoint_backward(
                x_hist, z_hist, S_hist, u_hist, lam_hist, dlam, eps_target_fn,
            )

            for u_k, g_k in zip(u_hist, ctrl_grads):
                u_k.backward(g_k, retain_graph=True)

            opt.step()

            lv = float(loss_val.detach())
            # print gradient norm for debugging
            total_grad_norm = 0.0
            for p in controller.parameters():
                if p.grad is not None:
                    param_grad_norm = p.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
            print("M: ", m, " steps: ", step_counter, " loss: ", lv, " grad norm: ", total_grad_norm)

            if lv < best_loss:
                best_loss  = lv
                best_state = {k: v.clone() for k, v in controller.state_dict().items()}

        seg_losses.append(best_loss)
        if verbose:
            print(f"  seg {m:3d}/{M} | lam=[{lam_start:.4f},"
                  f"{(k_start + H)/K:.4f}] | best_loss={best_loss:.4e}")

        controller.load_state_dict(best_state)
        z_run = z_cur.clone().detach()

        for k in range(H):
            # Fix 3: use integer-based lambda to match lam_hist exactly
            lam_k = (k_start + k) / K
            lam_t = torch.tensor([[lam_k]], dtype=torch.float32)
            with torch.no_grad():
                x_k = spring.solve_equilibrium(z_run)
                u_k = controller(lam_t)
            eps_exec.append(x_k.item())
            z_exec_arr.append(z_run.item())
            z_run = z_run + dlam * u_k.detach()

        z_cur = z_run.clone().detach()

    lam_grid = np.array([(k_start + k) / K
                          for m in range(M)
                          for k_start, k in [(m * H, k) for k in range(H)]])
    lam_grid = np.linspace(0.0, 1.0, K, endpoint=False)   # consistent with above
    eps_tgt  = np.array([eps_target_fn(l) for l in lam_grid])

    return lam_grid, np.array(eps_exec), np.array(z_exec_arr), eps_tgt, seg_losses


def plot_neural_control(lam_grid, eps_exec, z_exec_arr, eps_tgt, seg_losses):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(lam_grid, eps_tgt,  "k--", lw=1.5, label="Target $\\varepsilon^*(\\lambda)$")
    axes[0].plot(lam_grid, eps_exec, "b-",  lw=2.0, label="Realized $\\varepsilon(\\lambda)$")
    axes[0].set_xlabel("$\\lambda$")
    axes[0].set_ylabel("Strain")
    axes[0].set_title("Strain tracking")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(lam_grid, z_exec_arr, "r-", lw=2.0)
    axes[1].set_xlabel("$\\lambda$")
    axes[1].set_ylabel("Force $z(\\lambda)$")
    axes[1].set_title("Control input (force trajectory)")
    axes[1].grid(True)

    axes[2].semilogy(seg_losses, "o-", lw=1.5, ms=4)
    axes[2].set_xlabel("Segment index")
    axes[2].set_ylabel("Best loss (log scale)")
    axes[2].set_title("Segment-wise optimization loss")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    mse = np.mean((eps_exec - eps_tgt) ** 2)
    print(f"Trajectory MSE: {mse:.6e}")