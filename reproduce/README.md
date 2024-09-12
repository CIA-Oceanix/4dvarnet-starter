# REPRODUCING GLORYS RESULTS

This folder contains functions used in the two notebooks situated at the root of this github that can be used to reproduce testing results on OSE NRT (Near Real Time) data.

### dependencies

run these commands:

```
conda install pyinterp
pip install copernicusmarine
pip install "git+https://github.com/jejjohnson/ocn-tools.git"
```

## 1st step: Downloading and Pre-Processing OSE data:

Use [the data pipeline notebook](../ose_data_pipeline.ipynb) in order to **download** and **pre-process** your desired data.

## 2nd step: Using your trained 4DVarNet model and computing results

Use [the results reproducing notebook](../reproduce_glorys_ose_results.ipynb) in order to apply your **trained model** on the OSE data obtained during the **1st step**.
The reconstructed outputs will be compared with the OSE reference data and **metrics will be computed** for each present and future **leadtime**.


---
*You're gonna carry that .pth*
