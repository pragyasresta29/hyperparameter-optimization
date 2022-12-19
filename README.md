# hyperparameter-optimization

This project is part of my semester project where I am performing Hyperparameter Optimization for a ranges of training sizes. I have used 3 datasets from OpenML and implementing 3 machine learning algorithms. 

# Setup Instructions
1. Install Python version 3.8
2. The libraries used in this project can be found in src/req.txt. To install this using pip: `pip install -r src/req.txt`

# Using Scripts
There are two scripts in projects 
1. `src/hpo.py` - This scripts performs hyperparameter optimization on ranges of scripts on 3 datasets from OpenML on 3 machine learning algorithms. You can change the ranges within script if you want. But I have already ran all this in for ranges of sizes and stored the results in `src/info.json`. 
2. `src/analysis.py` - This script implement ML algorithms using the parameters generated from hyperparameter optimization which is stored in `src/info.json` and stores the accuracies in `src/hpo_data.csv`. The scripts then plots and stores the graps from the data. To run the script: `python analysis.py`. 

