import pandas as pd
import numpy as np
from pathlib import Path
import torch

from src.models import Lit4dVarNet


class FastRec4dVarNet(Lit4dVarNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def test_quantities(self):
        return ['out']

    def on_test_epoch_end(self):
        # test_data as gpu tensor
        self.test_data = torch.cat(self.test_data).cuda()
        super().on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)

        if batch_idx == 0:
            self.test_data = []
        out = self(batch=mask_batch)
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                #mask_batch.input.cpu() * s + m,
                #mask_batch.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))