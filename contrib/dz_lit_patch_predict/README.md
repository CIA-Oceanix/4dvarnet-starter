# my-ocb modules

This module links 4DVarNet to OceanBench for predictions. It performs the following pipeline:
- A `xarray.DataArray` with spatial-temporal observations data is given as input to 4DVarNet;
- 4DVarNet performs the predictions on each batch;
- Each predictions are stored in individual netCDF4 files in a directory.

## Todolist

- [x] Ajouter `src` lors de l'installation dans les dépendances (packages)
- [x] Paramètres :
    - Chemin vers obs
    - Chemin vers dossier de sortie
    - chemin vers config 4dvarnet d'entrainement
    - chemin vers une config 4dvarnet d'override ? (stride)
    - Chemin vers les poids
    - Stats de normalisation à utiliser
- [ ] Specifier les requirement du dataarray d'entrée:
    - dimensions lat lon time
    - Tailles > patch size (check full coverage du patcher ?)
- [x] à faire dans XrPatcher : récupérer les coordonnées de chaque batch après chaque prédiction avant de l'écrire dans un fichier
- [x] pendant du TrainingItem (qui prend tgt et inp) : écrire le PredictItem (qui prend que le inp)
    - [x] en conséquence, les norm_stats sont soit à spécifier en paramètres, soit à calculer depuis les inp
