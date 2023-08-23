# Decadal Predictability of Ocean Carbon Sink
 
Predicting changes in the ocean carbon flux is of relevance to climate policy, informing the Global Stocktake, and the setting of emission targets. Long term increases in ocean carbon uptake driven by increasing atmospheric CO$_2$ concentrations are well understood and predictable. However, inter-annual to decadal changes in the ocean carbon flux produced by climate variability challenges the predictability on shorter timescales. Climate models initialized using observations have proven skillful for near term predictability of the key physical climate variables. By comparison, predictions of biogeochemical fields such as the ocean carbon flux are in their infancy. Initial studies have indicated that skillful predictions are possible for lead times up to six years at the global scale for some of the CMIP6 Earth System Models. However, models diverge largely in their prediction skill while indirect initialization and extensive parametrization of ocean biogeochemistry in Earth System Models introduce a source of uncertainty. We propose a new approach for improving the skill of decadal ocean carbon flux predictions using obsearbvatioanlly-constrained statistical models as alternatives to the ocean biogeochemistry models. We train a multi-linear and a neural network model based on seven different observation-based products and a set of directly initialized physical variables plus surface chlorophyl concentration. We apply this method using input predictors derived from the 5$^{th}$ version of the Canadian Earth System Model (CanESM5) decadal prediction system to produce decadal predictions of the ocean carbon flux. Our results indicate that by combining the obsearvationally constrained models with directly initialized physical predictors, we can improve CanESM5 assimilation and hindcast skills on all lead times and above all scores reported previously for CMIP6 models. Using bias corrected CanESM5 forecast predictors, we make forecasts for ocean carbon flux up to year 2029. In close agreement, both statistical models show consistent faster than linear increases in ocean carbon flux that are larger than the changes predicted from CanESM5 decadal forecasts. This highlights the importance of studies of this kind to bring together observations and models for more robust predictions compared to raw ESMs.

Included in this repository:

    Directories: 
        envs: Conda environment file containing dcpp environment with all necessary packages to do the analysis.
        script: Includes .py scripts to design and run the linear and NN models. Also, it contains code for using these models for producing 
                statistical model based historical, assimilation, and hindcast simulations using input predictors from CanESM5. Finally,
                the scripts for loading CanESM5 predictor data and applying bias correction are included in this directory.

    Notebooks:
        Figures.ipynb: Codes for generating the figure for the manuscript, as well as necessary functions and analysis scripts.
        SI_Figures.ipynb: Codes for generating SI figures for the manuscript.

 by Parsa Gooya, M.Sc.
 parsa.g76@gmail.com
 Canadian Centre for Climate Modeling and Analysis (CCCma)
 Victoria, BC, CA