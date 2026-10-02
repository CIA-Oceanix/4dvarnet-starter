#!/bin/bash
# regrid the CLS 2024 nadir and nadir+SWOT SLA files onto the SST input grid (cell centres)
cd /Odyssey/private/d21botvy/OCEAN_FORECAST/4dvarnet-starter-glorys12
PY=/Odyssey/private/d21botvy/miniconda3/envs/4dvar_v2/bin/python
SST=/Odyssey/public/ODYSSEA_IFREMER_SST/L3_NRT/ANOMALY/SST-L3-NRT-2024_anomaly_climato_sla_grid.nc
OUT=/Odyssey/public/swot_traces/cls/sst_grid
$PY contrib/data_prep/regrid_sla_to_sst_grid.py /Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_nadirs_2024_without_swon.nc $SST $OUT/cls_nadirs_2024_sst_grid.nc && \
$PY contrib/data_prep/regrid_sla_to_sst_grid.py /Odyssey/public/swot_traces/cls/gridded_obs_sla_0.25_swot_2024_reformated.nc $SST $OUT/cls_nadir_swot_2024_sst_grid.nc
echo "REGRID EXIT $?"
