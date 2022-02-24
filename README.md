# Estimating CATE with Missing Treatment Information

This is a repository for the research paper "Estimating Conditional Average Treatment Effects with Missing Treatment Information". The repository includes the following scripts:

main.py (implementation of the proposed method)

benchmarks.py (implementation of benchmark models)

ihdp_exp.py (experiments with IHDP dataset)

twins_exp.py (experiments with Twins dataset)

jobs_exp.py (experiments with Jobs dataset)

# Requirements

python 3.6 

pytorch 1.7

# Experiments

We conduct experiments with three datasets: IHDP, Twins and Jobs. 

For IHDP we use a subset of 10 datasets from 100 simulated datasets provided by Shalit (2017). 

For Twins, we use the original dataset and simulate treatment assignment as in Yoon (2018). 

For Jobs we use the original dataset but modify the 7th covariate which represents the income before treatment. This covariate is heavily right-skewed so we apply the log-transformation. 

The method and the benchmarks are implemented as follows: MTRNet using pytorch; Linear model using scikit-learn; Causal forest (Athey, 2019) using econml; and TARNet and CFRMMD (Shalit, 2017) using pytorch.

# References

Shalit, U., Johansson, F. D., and Sontag, D. (2017). Estimating individual treatment effect: generalization bounds and algorithms. ICML.

Yoon, J., Jordon, J., and van der Schaar, M. (2018). GANITE: Estimation of individualized treatment effects using generative adversarial nets. ICLR.

Athey, S., Tibshirani, J., and Wager, S. (2019). Generalized random forests. The Annals of statistics.
