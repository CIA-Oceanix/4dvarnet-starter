"""Lead-time output weights for the 21-day 4DVarNet forecast model (Plus4dVarNetForecastPatchGPU_latent_dim).

That model observes days 0..13 of its window and forecasts days 14..20 (Lit4dVarNetForecast_only1leadtime.mask_batch),
but contrib.forecast_weights.forecast_weights_reconstruction_soft_edges puts the weight of lead i on day dT//2 + i
(= 10 + i for dT = 21), so its files test_data_14..17 are reconstructions of the observed days t0-4..t0-1.
This copy puts the weight on day n_obs + i (= 14 + i): file test_data_{14+i} is then the forecast of day i+1, as for the
29-day UNet (dT//2 = 14). Same spatial weights and soft edge as the original function.
"""
import numpy as np


def forecast_weights_reconstruction_soft_edges_after_obs(i, dT, dims, rec_weight, soft_edge=0.1, n_obs=14):
    center_wei = rec_weight[dT // 2].copy()
    center_wei[center_wei != 0] = 1.
    center_wei[center_wei == 0] = soft_edge
    return np.concatenate(
        (np.zeros((n_obs + i, dims[1], dims[2])),
         np.expand_dims(center_wei, axis=0),
         np.zeros((dT - n_obs - i - 1, dims[1], dims[2]))),
        axis=0,
    )
