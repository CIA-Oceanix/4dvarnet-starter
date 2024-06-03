import torch.nn.functional as F

def grad_perp(f, dx, dy):
    """Orthogonal gradient"""
    return (f[...,:-1] - f[...,1:]) / dy, (f[...,1:,:] - f[...,:-1,:]) / dx


def interp_TP(f):
    return 0.25 *(f[...,1:,1:] + f[...,1:,:-1] + f[...,:-1,1:] + f[...,:-1,:-1])


def laplacian_h(f, dx, dy):
    return F.pad(
        (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2*f[...,1:-1,1:-1]) / dx**2 \
      + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1]) / dy**2,
        (1,1,1,1), mode='constant', value=0.)
