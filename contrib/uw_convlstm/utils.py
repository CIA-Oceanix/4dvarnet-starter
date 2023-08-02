import torch

def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(lit_mod.parameters(), weight_decay=weight_decay, lr=lr)
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

