"""
Helmholtz equation solver with type-I discrete sine transform
and capacitance matrix method.
Louis Thiry, 2023.
"""
import torch
import torch.nn.functional as F

def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return torch.fft.irfft(-1j*F.pad(x, (1,1)), dim=-1, norm=norm)[...,1:x.shape[-1]+1]


def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1,-2), norm=norm).transpose(-1,-2)


def compute_laplace_dst(nx, ny, dx, dy, arr_kwargs):
    """Discrete sine transform of the 2D centered discrete laplacian
    operator."""
    x, y = torch.meshgrid(torch.arange(1,nx, **arr_kwargs),
                          torch.arange(1,ny, **arr_kwargs),
                          indexing='ij')
    return 2*(torch.cos(torch.pi/nx*x) - 1)/dx**2 + 2*(torch.cos(torch.pi/ny*y) - 1)/dy**2


def solve_helmholtz_dst(rhs, helmholtz_dst):
    return F.pad(dstI2D(dstI2D(rhs.type(helmholtz_dst.dtype)) / helmholtz_dst),
                 (1,1,1,1)
                ).type(torch.float64)


def compute_capacitance_matrices(helmholtz_dst, bound_xids, bound_yids):
    nl  = helmholtz_dst.shape[-3]
    M = bound_xids.shape[0]

    # compute G matrices
    G_matrices = torch.zeros((nl, M, M), dtype=torch.float64, device='cpu')
    rhs = torch.zeros(helmholtz_dst.shape[-3:], dtype=torch.float64,
                      device=helmholtz_dst.device)
    for m in range(M):
        rhs.fill_(0)
        rhs[..., bound_xids[m], bound_yids[m]] = 1
        sol = dstI2D(dstI2D(rhs) / helmholtz_dst.type(torch.float64))
        G_matrices[:,m] = sol[...,bound_xids, bound_yids].cpu()

    # invert G matrices to get capacitance matrices
    capacitance_matrices = torch.zeros_like(G_matrices)
    for l in range(nl):
        capacitance_matrices[l] = torch.linalg.inv(G_matrices[l])

    return capacitance_matrices.to(helmholtz_dst.device)


def solve_helmholtz_dst_cmm(rhs, helmholtz_dst,
                            cap_matrices, bound_xids, bound_yids,
                            mask):
    sol_rect = dstI2D(dstI2D(rhs.type(helmholtz_dst.dtype)) / helmholtz_dst)
    alphas = torch.einsum(
        '...ij,...j->...i',
        cap_matrices,
        -sol_rect[..., bound_xids, bound_yids].type(torch.float64))
    rhs_2 = rhs.clone()
    rhs_2[..., bound_xids, bound_yids] = alphas
    sol = dstI2D(dstI2D(rhs_2.type(helmholtz_dst.dtype)) / helmholtz_dst).type(torch.float64)
    return F.pad(sol, (1,1,1,1)) * mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.rcParams.update({'font.size': 24})
    plt.ion()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    # grid
    N = 8
    nx, ny = 2*2**(N-1), 2*2**(N-1)
    shape = (nx, ny)
    L = 2000e3
    xc = torch.linspace(-L, L, nx+1, dtype=dtype, device=device)
    yc = torch.linspace(-L, L, ny+1, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(xc, yc, indexing='ij')
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]

    # Helmholtz eq.
    lambd = torch.DoubleTensor(1,1,1).type(dtype).to(device)
    helmholtz = lambda f, dx, dy, lambd: \
        (   (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2*f[...,1:-1,1:-1])/dx**2
          + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1])/dy**2
          - lambd * f[...,1:-1,1:-1]
        )
    helmholtz_dst = compute_laplace_dst(
            nx, ny, dx, dy, {'dtype':dtype, 'device': device}) \
            - lambd


    # Rectangular domain
    frect = torch.zeros(1, nx+1, ny+1, dtype=dtype, device=device)
    frect[...,1:-1,1:-1].normal_()
    Hfrect = helmholtz(frect, dx, dy, lambd)
    frect_r = solve_helmholtz_dst(Hfrect, helmholtz_dst)
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].set_title('f')
    fig.colorbar(ax[0].imshow(frect[0].cpu().T, origin='lower'), ax=ax[0])
    ax[1].set_title('|f - f_r|')
    fig.colorbar(ax[1].imshow(torch.abs(frect - frect_r)[0].cpu().T, origin='lower'), ax=ax[1])
    fig.suptitle('Inverting Helmholtz equation on square domain with DST')
    fig.tight_layout()


    # Circular domain
    mask = (1 > ((xx/L)**2 + (yy/L)**2)).type(dtype)
    mask[[0,-1],:] = 0
    mask[[0,-1]] = 0
    domain_neighbor = \
        F.avg_pool2d(mask.reshape((1,1)+mask.shape), kernel_size=3, stride=1, padding=0)[0,0] > 0
    irrbound_xids, irrbound_yids = torch.where(
            torch.logical_and(mask[1:-1,1:-1] < 0.5, domain_neighbor))
    cap_matrices = compute_capacitance_matrices(
            helmholtz_dst, irrbound_xids,
            irrbound_yids)

    fcirc = mask * torch.zeros_like(mask).normal_().unsqueeze(0)
    Hfcirc = helmholtz(fcirc, dx, dy, lambd) * mask[1:-1,1:-1]
    fcirc_r = solve_helmholtz_dst_cmm(Hfcirc, helmholtz_dst,
                    cap_matrices, irrbound_xids,
                    irrbound_yids, mask)

    palette = plt.cm.bwr.with_extremes(bad='grey')
    fig, ax = plt.subplots(1,2, figsize=(18,9))
    ax[0].set_title('$f$')
    vM = fcirc[0].abs().max().cpu().item()
    fcirc_ma = np.ma.masked_where((1-mask).cpu().numpy(), fcirc[0].cpu().numpy())
    fig.colorbar(ax[0].imshow(fcirc_ma.T, vmin=-vM, vmax=vM, origin='lower', cmap=palette), ax=ax[0])
    ax[1].set_title('$f - f_{\\rm inv}$')
    diff =( fcirc - fcirc_r)[0].cpu().numpy()
    vM = np.abs(diff).max()
    diff_ma = np.ma.masked_where((1-mask).cpu().numpy(), diff)
    fig.colorbar(ax[1].imshow(diff_ma.T, vmin=-vM, vmax=vM, origin='lower', cmap=palette), ax=ax[1])
    fig.suptitle('Inverting Helmholtz eq. $\\Delta f - f = r$ on circular domain with CMM and DST')
    ax[0].set_xticks([]), ax[1].set_xticks([]), ax[0].set_yticks([]), ax[1].set_yticks([])
    fig.tight_layout()

    plt.show()
