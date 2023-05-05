# deep_learning_SNOM
This repository gives the core codes for the paper: https://doi.org/10.1063/5.0139517.

## Data generation
- We follow the codes provided in https://doi.org/10.1063/1.4941343 to generate all training data. We recommend the reader check the corresponding Supplementary Materials for all of the information.
- Meanwhile, the full signal simulation for any tip aspect ratio is also implemented in `iHNN_prediction_spheroid.py`, `FindBetaR.m`, and `PolesResidues.m`.
- `iHNN_prediction_spheroid.py` calls `FindBetaR.m` and `FindBetaR.m`calls `PolesResidues.m`.
- For a special tip aspect ratio $L=25a$, the signal can be rapidly generated using ratioanl approximation [[1]](#1). This is implemented in `iHNN_prediction_rational.py`

## Network construction and training
The following file implement the network depicted in the `Figure 3` of the main text.
- `train_image_predictor.py` constructs and trains the deep neural network that extracts the optical constants from the corresponding s-SNOM signal. 

## Network prediction
- `iHNN_prediction_spheroid.py` allows one to change the tip aspect ratio during optimization.
- `iHNN_prediction_rational.py` predicts faster for the fixed tip geometry $L=25a$.

## References
<a id="1">[1]</a> 
B.-Y. Jiang, L. M. Zhang, A. H. Castro Neto, D. N. Basov, and
M. M. Fogler, “Generalized spectral method for near-field optical
microscopy,” Journal of Applied Physics **119**, 054305 (2016).
