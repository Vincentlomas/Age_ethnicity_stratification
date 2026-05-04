# A method for including socio-demographic factors in social contact matrices for compartment-based epidemic models
A repository containg the code for the analysis presented in:

**Lomas, V. X., Chambers, T., Watson, L. M., Plank, P. (2026).** A method for including socio-demographic factors in social contact matrices for compartment-based epidemic models. [Journal TBD]

# Required python packages

numpy

matplotlib

scipy

seaborn

cycler


# Summary
These files contain code to extend a social contact matrix with an additional socio-demographic factor. This extended social contact matrix is then used to run various SEIR simulations which the final epidemic states (and reproductive number) is then plotted for use in the above paper. The R code file "POLYMOD_projection_contact_matrix.R" projects the POLYMOD study's social contact matrix onto NZ's age sturcute. Further details are present in the paper.

# File structure
```
- Age_ethnicity_stratification/    
  - README.md # This file
  - fitting_real_world_matrix.py                # Code to extend the POLYMOD projected matrix and run analysis/provide plots
  - numerical_analysis.py                       # Code to analyse the synthetic population example and run/plot single parameter variation
  - two_parameter_variation_heatplots.py        # Code to run two parameter variation and plot a healplot
  - data/
    - age_ethnicity_population_structure.csv    # the number of people in each age-ethnic group in NZ used in the analysis
    - contact_matrix_NZ.csv                     # The contact matrix found by projecting the POLYMOD results
    - popsize_2018.csv                          # The file used to estimate the age-ethnic population
  - generated_results/
    - # generated numpy arrays are stored here for plotting
  - images/
    - # misc images are stored here
    - epsilon_variation/
      - # plots of the how variation in epsilon affects epidemic outcome
    - heatplots/
      - rel_contact_rate_vs_socio_demo_epsilon/
        - # Two parameter variation heatplots
    - relative_contact_rate_variation/
      - # plots of the how variation in relative contact rate affects epidemic outcome
  - multi_factor_matrix_modules/
    - modules.py                                # modules used in multiple other files (currently or in the past) and others that are integral to the method
  - POLYMOD_projection/
    - POLYMOD_projection_contact_matrix.R       # R code used to project the POLYMOD results on NZ to get a social contact matrix
```

# Contact
If you have any question, ploease email vincent.lomas@pg.canterbury.ac.nz

# Citation
If you cite this, please cite it as:
```
Lomas, V. X., Chambers, T., Watson, L. M., Plank, M. (2026). A method for including socio-demographic
factors in social contact matrices for compartment-based epidemic models [Journal TBD].
```
