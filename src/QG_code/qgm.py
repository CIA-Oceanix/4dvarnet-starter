import numpy as np
import torch

from src.QG_code.fd import grad_perp, interp_TP, laplacian_h
from src.QG_code.flux import div_flux_3pts, div_flux_5pts, div_flux_3pts_mask, div_flux_5pts_mask
from src.QG_code.helmholtz import compute_laplace_dst, solve_helmholtz_dst, solve_helmholtz_dst_cmm, compute_capacitance_matrices
from src.QG_code.masks import Masks


class QGFV:
    """ Finite volume multi-layer QG solver. """

    def __init__(self, param):
        # physical params
        self.Lx = param['Lx']
        self.Ly = param['Ly']
        self.nl = param['nl']
        self.H = param['H']
        self.g_prime = param['g_prime']
        self.f0 = param['f0']
        self.beta = param['beta']
        self.bottom_drag_coef = param['bottom_drag_coef']
        self.dt = param['dt']

        # ensemble/device/dtype
        self.n_ens = param['n_ens']
        self.device = param['device']
        self.arr_kwargs = {'dtype':torch.float64, 'device': self.device}

        # grid params
        self.ny = param['ny']
        self.nx = param['nx']
        n_ens, nl, nx, ny = self.n_ens, self.nl, self.nx, self.ny
        self.dx = torch.tensor(self.Lx / nx, **self.arr_kwargs)
        self.dy = torch.tensor(self.Ly / ny, **self.arr_kwargs)
        self.flux_stencil = param['flux_stencil']
        self.y = torch.linspace(0.5*self.dy, self.Ly-0.5*self.dy, ny, **self.arr_kwargs).unsqueeze(0)
        self.y0 = 0.5 * self.Ly

        mask = param['mask'] if 'mask' in param.keys()  else torch.ones(nx, ny)
        self.masks = Masks(mask.type(torch.float64).to(self.device))

        # auxillary matrices for elliptic equation
        self.compute_auxillary_matrices()

        # initialize state variables
        self.psi = torch.zeros((n_ens, nl, nx+1, ny+1), **self.arr_kwargs)
        self.compute_q_from_psi()
        self.zeros_inside = torch.zeros((n_ens, nl-2, nx, ny), **self.arr_kwargs) \
                if nl > 2 else None

        # wind forcing
        self.wind_forcing = torch.zeros((1, 1, nx, ny), **self.arr_kwargs)

        # flux computations
        if self.flux_stencil == 5:
            if len(self.masks.psi_irrbound_xids) > 0:
                div_flux = div_flux_5pts_mask
                self.div_flux_args = (
                    self.masks.u_distbound1[...,1:-1,:],
                    self.masks.u_distbound2[...,1:-1,:],
                    self.masks.u_distbound3plus[...,1:-1,:],
                    self.masks.v_distbound1[...,1:-1],
                    self.masks.v_distbound2[...,1:-1],
                    self.masks.v_distbound3plus[...,1:-1]
                )
            else:
                div_flux = div_flux_5pts
                self.div_flux_args = ()
        elif self.flux_stencil == 3:
            if len(self.masks.psi_irrbound_xids) > 0:
                div_flux = div_flux_3pts_mask
                self.div_flux_args = (
                    self.masks.u_distbound1[...,1:-1,:],
                    self.masks.u_distbound2plus[...,1:-1,:],
                    self.masks.v_distbound1[...,1:-1],
                    self.masks.v_distbound2plus[...,1:-1],
                )
            else:
                div_flux = div_flux_3pts
                self.div_flux_args = ()

        # precompile torch functions if torch >= 2.0
        comp =  torch.__version__[0] == '2'
        self.grad_perp = torch.compile(grad_perp) if comp else grad_perp
        self.interp_TP = torch.compile(interp_TP) if comp else interp_TP
        self.laplacian_h = torch.compile(laplacian_h) if comp else laplacian_h
        self.div_flux = torch.compile(div_flux) if comp else div_flux
        if not comp:
            print('Need torch >= 2.0 to use torch.compile, current version '
                 f'{torch.__version__}, the solver will be slower! ')


    def compute_auxillary_matrices(self):
        # A operator
        H, g_prime = self.H.squeeze(), self.g_prime.squeeze()
        self.A = torch.zeros((self.nl,self.nl), **self.arr_kwargs)
        if self.nl == 1:
            self.A[0,0] = 1./(H*g_prime)
        else:
            self.A[0,0] = 1./(H[0]*g_prime[0]) + 1./(H[0]*g_prime[1])
            self.A[0,1] = -1./(H[0]*g_prime[1])
            for i in range(1, self.nl-1):
                self.A[i,i-1] = -1./(H[i]*g_prime[i])
                self.A[i,i] = 1./H[i]*(1/g_prime[i+1] + 1/g_prime[i])
                self.A[i,i+1] = -1./(H[i]*g_prime[i+1])
            self.A[-1,-1] = 1./(H[self.nl-1]*g_prime[self.nl-1])
            self.A[-1,-2] = -1./(H[self.nl-1]*g_prime[self.nl-1])

        # layer-to-mode and mode-to-layer matrices
        ev_A, P = torch.linalg.eig(self.A)
        self.lambda_sq = self.f0**2 * ev_A.real.reshape((1, self.nl, 1, 1))
        self.Cl2m = torch.linalg.inv(P.real)
        self.Cm2l = P.real
        self.rossby_radii = self.lambda_sq.squeeze().pow(-0.5)
        with np.printoptions(precision=1):
            print(f'Rossby rad.: {self.rossby_radii.cpu().numpy()/1e3} km')

        # For Helmholtz equations
        nx, ny, nl = self.nx, self.ny, self.nl
        laplace_dst = compute_laplace_dst(
                nx, ny, self.dx, self.dy, self.arr_kwargs).unsqueeze(0).unsqueeze(0)
        self.helmholtz_dst =  laplace_dst - self.lambda_sq


        # homogeneous Helmholtz solutions
        cst = torch.ones((1, nl, nx+1, ny+1), **self.arr_kwargs)
        if len(self.masks.psi_irrbound_xids) > 0:
            self.cap_matrices = compute_capacitance_matrices(
                self.helmholtz_dst, self.masks.psi_irrbound_xids,
                self.masks.psi_irrbound_yids)
            sol = solve_helmholtz_dst_cmm(
                    (cst*self.masks.psi)[...,1:-1,1:-1],
                    self.helmholtz_dst, self.cap_matrices,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            self.cap_matrices = None
            sol = solve_helmholtz_dst(cst[...,1:-1,1:-1], self.helmholtz_dst)

        self.homsol = cst + sol * self.lambda_sq
        self.homsol_mean = (
                interp_TP(self.homsol)*self.masks.q).mean((-1,-2), keepdim=True)
        self.helmholtz_dst = self.helmholtz_dst.type(torch.float32)


    def compute_q_from_psi(self):
        self.q = self.masks.q * (
            interp_TP(self.masks.psi * (
                laplacian_h(self.psi, self.dx, self.dy) \
              - self.f0**2*torch.einsum('lm,...mxy->...lxy', self.A, self.psi)
              )
            )
            + self.beta * (self.y - self.y0))

    def compute_psi_from_q(self):
        """PV inversion."""
        elliptic_rhs = self.interp_TP(self.q - self.beta * (self.y - self.y0))
        helmholtz_rhs = torch.einsum(
                'lm,...mxy->...lxy', self.Cl2m, elliptic_rhs)
        if self.cap_matrices is not None:
            psi_modes = solve_helmholtz_dst_cmm(
                    helmholtz_rhs*self.masks.psi[...,1:-1,1:-1],
                    self.helmholtz_dst, self.cap_matrices,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            psi_modes = solve_helmholtz_dst(helmholtz_rhs, self.helmholtz_dst)

        # Add homogeneous solutions to ensure mass conservation
        alpha = -self.interp_TP(psi_modes).mean((-1,-2), keepdim=True) \
                / self.homsol_mean
        psi_modes += alpha * self.homsol
        self.psi = torch.einsum('lm,...mxy->...lxy', self.Cm2l, psi_modes)


    def set_wind_forcing(self, curl_tau):
        self.wind_forcing = curl_tau / self.H[0]


    def advection_rhs(self):
        u, v = self.grad_perp(self.psi, self.dx, self.dy)
        div_flux = self.div_flux(self.q, u[...,1:-1,:], v[...,1:-1],
                                 self.dx, self.dy, *self.div_flux_args)

        # wind forcing + bottom drag
        omega = self.interp_TP(self.laplacian_h(self.psi, self.dx, self.dy)*self.masks.psi)
        bottom_drag = -self.bottom_drag_coef * omega[...,[-1],:,:]
        if self.nl == 1:
            fcg_drag = self.wind_forcing + bottom_drag
        elif self.nl == 2:
            fcg_drag = torch.cat([self.wind_forcing, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat([self.wind_forcing, self.zeros_inside, bottom_drag], dim=-3)

        return (-div_flux + fcg_drag) * self.masks.q


    def compute_time_derivatives(self):
        dq = self.advection_rhs()

        # Solve Helmholtz equation
        dq_i = self.interp_TP(dq)
        helmholtz_rhs = torch.einsum(
                'lm,...mxy->...lxy', self.Cl2m, dq_i)
        if self.cap_matrices is not None:
            dpsi_modes = solve_helmholtz_dst_cmm(
                    helmholtz_rhs*self.masks.psi[...,1:-1,1:-1],
                    self.helmholtz_dst, self.cap_matrices,
                    self.masks.psi_irrbound_xids,
                    self.masks.psi_irrbound_yids,
                    self.masks.psi)
        else:
            dpsi_modes = solve_helmholtz_dst(helmholtz_rhs, self.helmholtz_dst)
        # Add homogeneous solutions to ensure mass conservation
        alpha = -self.interp_TP(dpsi_modes).mean((-1,-2), keepdim=True) \
                / self.homsol_mean
        dpsi_modes += alpha * self.homsol
        dpsi = torch.einsum('lm,...mxy->...lxy', self.Cm2l, dpsi_modes)

        return dpsi, dq


    def step(self):
        """ Time itegration with SSP-RK3 scheme."""
        dpsi_0, dq_0 = self.compute_time_derivatives()
        self.q += self.dt * dq_0
        self.psi += self.dt * dpsi_0

        dpsi_1, dq_1 = self.compute_time_derivatives()
        self.q += (self.dt/4)*(dq_1 - 3*dq_0)
        self.psi += (self.dt/4)*(dpsi_1 - 3*dpsi_0)

        dpsi_2, dq_2 = self.compute_time_derivatives()
        self.q += (self.dt/12)*(8*dq_2 - dq_1 - dq_0)
        self.psi += (self.dt/12)*(8*dpsi_2 - dpsi_1 - dpsi_0)
