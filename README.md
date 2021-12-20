# NERD_code_UNCECOMP2021
Code for the UNCECOMP2021 paper: "Efficient Discrimination between Biological Populations via Neural-based Estimation of  Renyi Divergence".
https://generalconferencefiles.s3.eu-west-1.amazonaws.com/uncecomp_2021_proceedings.pdf

Tested with:
python               3.6
tensorflow           1.15.0
scipy                1.4.1
numpy                1.18.5
matplotlib           3.3.4



ITE (version 0.63) computations are provided in the form of binary matlab files (.mat).

Real datasets are too big (larger than 15mb) for this repository, so we suggest that they are downloaded from: 
https://community.cytobank.org/cytobank/experiments/46098/illustrations/121588
and prepared, as mentioned in the main text.
For Figure 6, we provide notebooks for visualization purposes based on the output data upon computations (.csv files). 
We have included the main python code along with a sample script that can be used for multiple instances of the same parameters over different input files. 
