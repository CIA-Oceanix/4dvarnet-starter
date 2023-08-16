import torch

def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(lit_mod.parameters(), weight_decay=weight_decay, lr=lr)
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def scotts_lr(lit_mod, lr=5e-4, weight_decay=0., **kwargs):
    opt = torch.optim.Adam(lit_mod.parameters(), weight_decay=weight_decay, lr=lr)
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.3, patience=8, verbose=1, cooldown=0, min_lr=1e-5,
        ),
        "monitor": "val_loss"
    }

