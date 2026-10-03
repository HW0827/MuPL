# MuPL
This repository contains the implementation and dataset for the "Multiscale Physics-Guided Learning" (MuPL) model. The code aims to bridge the gap between material-level properties of heat exchangers and system-level performance prediction. 
Core Features 
1. Physics-Guided Machine Learning with Consistency Constraints 
A hybrid neural network is implemented, integrating material parameters (e.g., contact angle, thermal conductivity) with operating conditions. Cross-scale physical laws are encoded as consistency constraints embedded within the loss function, rather than relying solely on known governing equations. These physics-based constraints ensure physically plausible predictions and significantly enhance the model's generalization capability beyond training data. 
2. Multi-Source Data Fusion and Preprocessing 
Experimental data from both air-cooled and water-cooled heat exchanger configurations are processed. Key input features and target performance metrics are carefully handled. Normalization and feature engineering procedures are applied separately for different coolant types. 
3. Model Training 
First, a purely data-driven baseline model is trained for performance comparison. Then, the main MuPL model is trained. A Monte Carlo ensemble approach is integrated to predict key microscale features—liquid volume fraction δV and non-condensable gas mass fraction φm—from simulation data, providing uncertainty estimates. 
4. Core Outputs 
The trained MuPL model is saved. Error metrics (RMSE, MAPE, R²) for both the data-driven and MuPL models are printed.
5. Usage Running 
the main() function executes the complete workflow. This framework can be flexibly adapted to other material-device-system combinations by modifying input features and the formulation of core physical constraints.
