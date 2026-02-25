import unittest

import torch
from torch import nn

from galore_torch import GaLoreAdamW
from galore_torch.diagnostics import galore_projector_nbytes, optimizer_state_tensor_nbytes


class TestDiagnostics(unittest.TestCase):
    def test_galore_optimizer_state_smaller_than_adamw(self):
        torch.manual_seed(0)
        x = torch.randn(4, 128)

        model_adamw = nn.Linear(128, 128, bias=False)
        opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=1e-3)
        loss_adamw = model_adamw(x).pow(2).mean()
        loss_adamw.backward()
        opt_adamw.step()
        opt_adamw.zero_grad(set_to_none=True)
        adamw_state_bytes = optimizer_state_tensor_nbytes(opt_adamw)

        torch.manual_seed(0)
        model_galore = nn.Linear(128, 128, bias=False)
        param_groups = [
            {"params": []},
            {
                "params": [model_galore.weight],
                "rank": 8,
                "update_proj_gap": 1,
                "scale": 1.0,
                "proj_type": "std",
            },
        ]
        opt_galore = GaLoreAdamW(param_groups, lr=1e-3, no_deprecation_warning=True)
        loss_galore = model_galore(x).pow(2).mean()
        loss_galore.backward()
        opt_galore.step()
        opt_galore.zero_grad(set_to_none=True)
        galore_state_bytes = optimizer_state_tensor_nbytes(opt_galore)

        self.assertLess(galore_state_bytes, adamw_state_bytes)

    def test_projector_memory_non_zero(self):
        torch.manual_seed(0)
        x = torch.randn(4, 128)
        model = nn.Linear(128, 128, bias=False)
        param_groups = [
            {
                "params": [model.weight],
                "rank": 8,
                "update_proj_gap": 1,
                "scale": 1.0,
                "proj_type": "std",
            }
        ]
        opt = GaLoreAdamW(param_groups, lr=1e-3, no_deprecation_warning=True)
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()

        self.assertGreater(galore_projector_nbytes(opt), 0)


if __name__ == "__main__":
    unittest.main()

