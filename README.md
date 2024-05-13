## VAE and AE for anomaly detection

This repo contains [autoencoder](https://arxiv.org/pdf/2311.01452) and variational autoencoder models implementation for anomaly detection on multivariate time
series.

### Used data

All experiments were conducted on [SWaT](https://itrust.sutd.edu.sg/itrust-labs_datasets/) dataset. To preprocess and visualize data, use `swat_preproc.ipynb` notebook.

### Model training and evaluation

To train both AE or VAE choose necessary model in `models.py` and train them using `run_model.ipynb`.
You can find code for model evaluation and it's latent space t-SNE visualization in the same `run_model.ipynb` file.

Reconstructed time series can be visualized using `training.plot_output` function.  

### Metrics 

$F1_{K}-AUC$ and $ROC_{K}-AUC$ metrics (see `metrics_eval.py` or https://arxiv.org/pdf/2109.05257 ) are used to evaluate models.
