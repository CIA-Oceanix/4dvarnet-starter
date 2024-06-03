import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from src.QG_code.masks import Masks

matplotlib.rcParams.update({'font.size': 18})

laplacian = lambda f, dx, dy: (f[...,2:,1:-1] + f[...,:-2,1:-1] - 2 *f[...,1:-1,1:-1]) / dx**2 \
                            + (f[...,1:-1,2:] + f[...,1:-1,:-2] - 2*f[...,1:-1,1:-1]) / dy**2
grad_perp = lambda f, dx, dy: ((f[...,:-1] - f[...,1:])/dy, (f[...,1:,:] - f[...,:-1,:])/dx)

f0 = 9.375e-5 # mean coriolis (s^-1)
dx, dy = 20000, 20000
nx, ny = 256, 256
mask = torch.ones(nx, ny)
for i in range(nx//4):
    for j in range(ny//4):
        if i+j < min(nx//4, ny//4):
            mask[i,j] = 0.
            mask[i,-1-j] = 0.
            mask[-1-i,j] = 0.
            mask[-1-i,-1-j] = 0.
masks = Masks(mask)
mask_psi = masks.psi.cpu().numpy()[0,0]
mask_q = masks.q.cpu().numpy()[0,0]


psi = np.load('run_outputs/256x256_dt4000/psi_049y_355d.npy')

plt.ion()
plt.figure()
omega = laplacian(psi, dx, dy) * mask_psi[...,1:-1,1:-1]
f1, a1 = plt.subplots(1, 2)
palette = plt.cm.bwr.with_extremes(bad='grey')

H1 = 400
psi_up = np.ma.masked_where(1 - mask_psi, (H1*psi*1e-6)[0,0])
pM = np.abs(psi_up).max()

omega_up = np.ma.masked_where(1 - mask_psi[...,1:-1,1:-1], omega[0,0])

f1.colorbar(a1[0].imshow(psi_up.T, cmap=palette, vmin=-pM, vmax=pM), ax=a1[0])
a1[0].set_title('Stream-function (Sv)')
f1.colorbar(a1[1].imshow(omega_up.T/ f0, cmap=palette, vmin=-0.3, vmax=0.3), ax=a1[1])
a1[1].set_title('Relative vorticity (units of f0)')
a1[0].set_xticks([]), a1[0].set_yticks([])
a1[1].set_xticks([]), a1[1].set_yticks([])
f1.suptitle('Upper-layer after 50 years of spin-up')
plt.pause(0.01)

import glob
files = sorted(glob.glob('run_outputs/256x256_dt4000/psi_*.npy'))

psi_m = 0.
u_m = 0.
u2_m = 0.
v_m = 0.
v2_m = 0.

for i, f in enumerate(files):
    print(f) if not i % 10 else None
    psi = np.load(f)
    psi_m += psi
    u, v = grad_perp(psi, dx, dy)
    u, v = 0.5*(u[...,1:,:]+u[...,:-1,:]), 0.5*(v[...,1:]+v[...,:-1])
    u_m += u
    u2_m += u**2
    v_m += v
    v2_m += v**2

psi_m /= len(files)
u_m /= len(files)
v_m /= len(files)
u2_m /= len(files)
v2_m /= len(files)

mke = 0.5 * (u_m**2 + v_m**2)
eke = 0.5 * ((u2_m - u_m**2) + (v2_m - v_m**2))


f, a = plt.subplots(1,3, figsize=(18,6))

mask = psi_m[0,0] == 0
psi = np.ma.masked_where(1 - mask_psi, H1*psi_m[0,0]*1e-6)
pM = np.abs(psi).max()

a[0].contour(psi.T, origin='lower', colors='grey', levels=np.linspace(-pM, pM, 10)[1:-1])
f.colorbar(a[0].imshow(psi.T, origin='lower', cmap=palette, vmin=-pM, vmax=pM), ax=a[0])
a[0].set_xticks([]), a[0].set_yticks([])
a[0].set_title('Mean stream-function (Sv)')

palette = plt.cm.Reds.with_extremes(bad='grey')
mke_up = np.ma.masked_where(1 - mask_q, mke[0,0])
f.colorbar(a[1].imshow(mke_up.T, origin='lower', cmap=palette), ax=a[1])
a[1].set_title('Mean kin. energy density (m$^2$s$^{-2}$)')
a[1].set_xticks([]), a[1].set_yticks([])
eke_up = np.ma.masked_where(1 - mask_q, eke[0,0])
f.colorbar(a[2].imshow(eke_up.T, origin='lower', cmap=palette), ax=a[2])
a[2].set_title('Eddy kin. energy density (m$^2$s$^{-2}$)')
a[2].set_xticks([]), a[2].set_yticks([])
plt.tight_layout()
plt.pause(0.1)
