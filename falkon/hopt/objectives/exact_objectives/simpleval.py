from typing import Optional, Dict

import torch

import falkon.kernels
from falkon.hopt.objectives.exact_objectives.utils import jittering_cholesky
from falkon.hopt.objectives.objectives import HyperoptObjective
from falkon.hopt.utils import get_scalar


class SimpleVal(HyperoptObjective):
    def __init__(
            self,
            kernel: falkon.kernels.DiffKernel,
            centers_init: torch.Tensor,
            penalty_init: torch.Tensor,
            opt_centers: bool,
            opt_penalty: bool,
            centers_transform: Optional[torch.distributions.Transform] = None,
            pen_transform: Optional[torch.distributions.Transform] = None,
    ):
        super(SimpleVal, self).__init__(kernel, centers_init, penalty_init,
                                      opt_centers, opt_penalty,
                                      centers_transform, pen_transform)
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.losses: Optional[Dict[str, torch.Tensor]] = None

    def forward(self, X, Y, Xv, Yv):
        self.x_train, self.y_train = X, Y
        self.x_val, self.y_val = Xv, Yv

        kmval = self.kernel(self.centers, Xv)
        alpha = self._calc_intermediate(X, Y)
        val_preds = kmval.T @ alpha
        # Replaced the MSE by an MAE on the validation set
        loss_fn = torch.nn.L1Loss()
        loss = loss_fn(Yv, val_preds)

        self._save_losses(loss)
        return loss

    def predict(self, X):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call forward at least once before calling predict.")
        with torch.autograd.no_grad():
            alpha = self._calc_intermediate(self.x_train, self.y_train)
            kms = self.kernel(self.centers, X)
            return kms.T @ alpha

    def _calc_intermediate(self, X, Y):
        variance = self.penalty * X.shape[0]
        sqrt_var = torch.sqrt(variance)

        kmn = self.kernel(self.centers, X)
        kmm = self.kernel(self.centers, self.centers)
        L = jittering_cholesky(kmm)  # L @ L.T = kmm
        # A = L^{-1} K_mn / (sqrt(n*pen))
        A = torch.triangular_solve(kmn, L, upper=False).solution / sqrt_var
        AAT = A @ A.T
        # B = A @ A.T + I
        B = AAT + torch.eye(AAT.shape[0], device=X.device, dtype=X.dtype)
        LB = jittering_cholesky(B)  # LB @ LB.T = B
        AYtr = A @ Y
        c = torch.triangular_solve(AYtr, LB, upper=False).solution / sqrt_var

        tmp1 = torch.triangular_solve(c, LB, upper=False, transpose=True).solution
        alpha = torch.triangular_solve(tmp1, L, upper=False, transpose=True).solution
        return alpha

    def _save_losses(self, simpleval):
        self.losses = {
            "simple-val": simpleval.detach(),
        }

    def __repr__(self):
        return f"SimpleVal(" \
               f"kernel={self.kernel}, " \
               f"penalty={get_scalar(self.penalty)}, " \
               f"num_centers={self.centers.shape[0]})" 
