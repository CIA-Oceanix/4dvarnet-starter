"""
Flux computations.
Louis Thiry, 2023
"""
import torch
import torch.nn.functional as F

from src.QG_code.reconstruction import \
        linear2, linear3_left, linear4, linear5_left, \
        weno3, weno3z, weno5, weno5z

def flux_1pts(q, u, dim):
    n = q.shape[dim]

    qi_left = q.narrow(dim, 0, n-1)
    qi_right = q.narrow(dim, 1, n-1)

    # positive and negative parts of velocity
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # upwind flux computation
    flux = u_pos * qi_left + u_neg * qi_right

    return flux

def flux_3pts(q, u, dim):
    """
    Flux computation for staggerded variables q and u, with solid boundaries.
    Upwind-biased stencil:
      - 3 points inside domain.
      - 1 point near boundaries.

    Args:
        q: tracer field to interpolate, torch.Tensor, shape[dim] = n
        u: transport velocity, torch.Tensor, shape[dim] = n-1
        dim: dimension along which computations are done

    Returns:
        flux: tracer flux computed on u points, torch.Tensor, shape[dim] = n-1
    """
    n = q.shape[dim]

    # q-interpolation: 3-points inside domain
    qm, q0, qp = q.narrow(dim, 0, n-2), q.narrow(dim, 1, n-2), q.narrow(dim, 2, n-2)
    qi_left_in = linear3_left(qm, q0, qp)
    qi_right_in = linear3_left(qp, q0, qm)

    # q-interpolation: 2-points on boundaries
    qi_0 = linear2(q.narrow(dim, 0, 1), q.narrow(dim, 1, 1))
    qi_m1 = linear2(q.narrow(dim, -2, 1), q.narrow(dim, -1, 1))

    qi_left = torch.cat([
        qi_0, qi_left_in.narrow(dim, 0, n-3), qi_m1], dim=dim)
    qi_right = torch.cat([
        qi_0, qi_right_in.narrow(dim, 1, n-3), qi_m1], dim=dim)

    # positive and negative parts of velocity
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # upwind flux computation
    flux = u_pos * qi_left + u_neg * qi_right
    return flux


def div_flux_3pts(q, u, v, dx, dy):
    q_flux_y = F.pad(flux_3pts(q, v, dim=-1), (1,1,0,0))
    q_flux_x = F.pad(flux_3pts(q, u, dim=-2), (0,0,1,1))

    return  torch.diff(q_flux_x, dim=-2) / dx + \
            torch.diff(q_flux_y, dim=-1) / dy


def flux_5pts(q, u, dim):
    """
    Flux computation for staggerded variables q and u, with solid boundaries.
    Upwind-biased stencil:
      - 5 points inside domain.
      - 1 or 3 points near boundaries.

    Args:
        q: tracer field to interpolate, torch.Tensor, shape[dim] = n
        u: transport velocity, torch.Tensor, shape[dim] = n-1
        dim: dimension along which computations are done

    Returns:
        flux: tracer flux computed on u points, torch.Tensor, shape[dim] = n-1
        qi: tracer field interpolated on u points, torch.Tensor, shape[dim] = n-1
    """

    n = q.shape[dim]

    # 5-points inside domain
    qmm, qm, q0, qp, qpp = q.narrow(dim, 0, n-4), q.narrow(dim, 1, n-4), \
                           q.narrow(dim, 2, n-4), q.narrow(dim, 3, n-4), \
                           q.narrow(dim, 4, n-4)
    qi_left_in = linear5_left(qmm, qm, q0, qp, qpp)
    qi_right_in = linear5_left(qpp, qp, q0, qm, qmm)
    # qi_left_in = weno5z(qmm, qm, q0, qp, qpp)
    # qi_right_in = weno5z(qpp, qp, q0, qm, qmm)

    # 3pts-2pts near boundary
    qm, q0, qp = torch.cat([q.narrow(dim, 0, 1), q.narrow(dim, -3, 1)], dim=dim), \
                torch.cat([q.narrow(dim, 1, 1), q.narrow(dim, -2, 1)], dim=dim), \
                torch.cat([q.narrow(dim, 2, 1), q.narrow(dim, -1, 1)], dim=dim)
    qi_left_b = weno3z(qm, q0, qp)
    qi_right_b = weno3z(qp, q0, qm)

    qi_0 = linear2(q.narrow(dim, 0, 1), q.narrow(dim, 1, 1))
    qi_m1 = linear2(q.narrow(dim, -2, 1), q.narrow(dim, -1, 1))

    qi_left = torch.cat([qi_0,
                         qi_left_b.narrow(dim, 0, 1),
                         qi_left_in,
                         qi_left_b.narrow(dim, -1, 1)
                        ], dim=dim)
    qi_right = torch.cat([qi_right_b.narrow(dim, 0, 1),
                          qi_right_in,
                          qi_right_b.narrow(dim, -1, 1),
                          qi_m1
                         ], dim=dim)

    # positive and negative parts of velocity
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # upwind flux computation
    flux = u_pos * qi_left + u_neg * qi_right

    return flux


def div_flux_5pts(q, u, v, dx, dy):
    q_flux_y = F.pad(flux_5pts(q, v, dim=-1), (1,1,0,0))
    q_flux_x = F.pad(flux_5pts(q, u, dim=-2), (0,0,1,1))

    return  torch.diff(q_flux_x, dim=-2) / dx + \
            torch.diff(q_flux_y, dim=-1) / dy


def flux_3pts_mask(q, u, dim, mask_u_d1, mask_u_d2plus):
    n = q.shape[dim]
    pad1 = () if dim == -1 else (0,0)
    pad2 = (0,0) if dim == -1 else ()
    qm, q0, qp = q.narrow(dim, 0, n-2), q.narrow(dim, 1, n-2), \
                 q.narrow(dim, 2, n-2)
    qi3_left = F.pad(weno3z(qm, q0, qp), pad1+(1,0)+pad2)
    qi3_right = F.pad(weno3z(qp, q0, qm), pad1+(0,1)+pad2)
    # qi2 = linear2(q.narrow(dim, 0, n-1), q.narrow(dim, 1, n-1))

    u_pos = F.relu(u)
    u_neg = u - u_pos
    # flux = u * (mask_u_d1 * qi2) \
    flux = mask_u_d1 * (u_pos*q.narrow(dim, 0, n-1) + u_neg*q.narrow(dim, 1, n-1)) \
         + mask_u_d2plus * (u_pos*qi3_left + u_neg*qi3_right)

    return flux


def div_flux_3pts_mask(
        q, u, v, dx, dy,
        mask_u_d1, mask_u_d2plus,
        mask_v_d1, mask_v_d2plus):
    q_flux_y = flux_3pts_mask(q, v, -1, mask_v_d1, mask_v_d2plus)
    q_flux_x = flux_3pts_mask(q, u, -2, mask_u_d1, mask_u_d2plus)
    return  torch.diff(F.pad(q_flux_x, (0,0,1,1)), dim=-2) / dx + \
            torch.diff(F.pad(q_flux_y, (1,1)), dim=-1) / dy


def flux_5pts_mask(
        q, u, dim, mask_u_d1,
        mask_u_d2, mask_u_d3plus):
    n = q.shape[dim]
    pad1 = () if dim == -1 else (0,0)
    pad2 = (0,0) if dim == -1 else ()
    qmm, qm, q0, qp, qpp = q.narrow(dim, 0, n-4), q.narrow(dim, 1, n-4), \
                           q.narrow(dim, 2, n-4), q.narrow(dim, 3, n-4), \
                           q.narrow(dim, 4, n-4)
    # qi5_left = F.pad(weno5z(qmm, qm, q0, qp, qpp), pad1+(2,1)+pad2)
    # qi5_right = F.pad(weno5z(qpp, qp, q0, qm, qmm), pad1+(1,2)+pad2)
    qi5_left = F.pad(linear5_left(qmm, qm, q0, qp, qpp), pad1+(2,1)+pad2)
    qi5_right = F.pad(linear5_left(qpp, qp, q0, qm, qmm), pad1+(1,2)+pad2)

    # qi4 = F.pad(
        # linear4(q.narrow(dim, 0, n-3), q.narrow(dim, 1, n-3),
                # q.narrow(dim, 2, n-3), q.narrow(dim, 3, n-3)),
        # pad1+(1,1)+pad2)

    qm, q0, qp = q.narrow(dim, 0, n-2), q.narrow(dim, 1, n-2), \
                 q.narrow(dim, 2, n-2)
    qi3_left = F.pad(linear3_left(qm, q0, qp), pad1+(1,0)+pad2)

    qi3_right = F.pad(linear3_left(qp, q0, qm), pad1+(0,1)+pad2)

    qi2 = linear2(q.narrow(dim, 0, n-1), q.narrow(dim, 1, n-1))

    u_pos = F.relu(u)
    u_neg = u - u_pos
    flux = u * mask_u_d1 * qi2 + \
           mask_u_d2 * (u_pos*qi3_left + u_neg*qi3_right) \
         + mask_u_d3plus * (u_pos*qi5_left + u_neg*qi5_right)

    return flux


def div_flux_5pts_mask(q, u, v, dx, dy,
        mask_u_d1, mask_u_d2, mask_u_d3plus,
        mask_v_d1, mask_v_d2, mask_v_d3plus):
    q_flux_y = flux_5pts_mask(
        q, v, -1, mask_v_d1, mask_v_d2, mask_v_d3plus)
    q_flux_x =  flux_5pts_mask(
        q, u, -2, mask_u_d1, mask_u_d2, mask_u_d3plus)

    return torch.diff(F.pad(q_flux_x, (0,0,1,1)), dim=-2) / dx + \
           torch.diff(F.pad(q_flux_y, (1,1)), dim=-1) / dy
