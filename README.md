
# arTIco - Artificial Intelligence and Correlations 

Framework for development of multichannel time series classification from project arTIco.

Project arTIco started as two-year in January 2022 to explore methods for validating digital twins. The proposed approaches build on artificial intelligence, intending to enable a much broader assessment of model validity than current methods.

The application example of crash test dummies for passive vehicle safety should demonstrate the methodology's performance. The focus is on the evaluation and further development of previously used correlation factors. The new approaches are intended to enhance the established ones with expert knowledge, make them more objective, and be established in the future as an alternative to current certification procedures.

The arTIco project investigates an innovative approach to the objective assessment of validation and certification quality using artificial intelligence. Expert knowledge is conserved and used to expand the existing process. This provides a foundation for assessing vehicle safety through virtual crash simulations that surpasses the current state of the art. The validation, calibration, and certification of virtual models are usually based on individual comparisons. For example, a corridor is specified for the data collected by a particular sensor. For the model to be valid, the data must be within this corridor. However, by using only a few sensors and criteria, the actual physics is only considered to a limited extent. This leads to limited usability of the virtual models and to a reduction of confidence in them. A broader database within the validation should significantly increase the model quality. Ultimately, increased confidence in the model quality should lead to crash simulations increasingly replacing expensive hardware tests.




## Acknowledgements

arTIco is a research project in the funding line "Digitalization" of the Bavarian Cooperative Research Programme (BayVFP) of the Free State of Bavaria under the funding number: DIK-2110-0025// DIK0404/02

## Authors

- Franz Plaschkies (Technische Hochschule Ingolstadt)
- Felix Stocker (Applus+ IDIADA Fahrzeugtechnik GmbH)
- Zhengxuan Yuan (Applus+ IDIADA Fahrzeugtechnik GmbH)

## Usage/Examples

```cmd
# generate dummy data
src/load/LoadFromRaw.py
src/load/ExtractedFeatures.py

# run examples
src/experiments/_test_examples.py.py
```


## Documentation

Convention: files starting with _ should not be changed
Store your learners in src/build
Generate your experiments (single and optuna) in src/experiment

### Structure

    ├── LICENSE
	│
    ├── README.md   <- The top-level README for developers using this project.
	│
	├── requirements.txt   <- The requirements file for reproducing the analysis 
	│                         environment, e.g. generated with 
	│						  conda list -e > requirements.txt
    │                         install with 
	│                         conda install -n <env_name> requirements.txt
    ├── data
    │   ├── processed   <- The final, canonical data sets for modeling; use
	│   │   │              src/load/LoadFromRaw.py and 
	│	│	│			   src/load/ExtractedFeatures.py to generate the data
	│	│	│			   from raw
    │   │   └── data_info.json
	│   │   └── data_info2D.json
    │   │   └── feature.npy
    │   │   └── feature2D.npy
    │   │   └── target.npy
    │   │
    │   └── raw   <- The original, immutable data dump.
    │       └── generate_random_data.py <- generate dummy data
	│       └── ratings_experts.csv     <- ratings to be learned
    │       └── ratings_iso18571.csv    <- ISO 18571 objective ratings
    │       └── signals_cae.csv         <- CAE data
    ├── experiments    <- Trained and serialized models, model predictions, or
	│   │                 model summaries
    │   └── xxx    <- example directory of single experiment, name convention:
	│       │         YYYY-MM-DD-hh-mm-ss_description
    │       └── parameters.json      <- parameters of experiment
    │       └── results.json         <- evaluated results of experiment
    │       └── pipeline_dev_fit.pkl <- pickled pipeline
    │       └── results.csv.zip      <- raw experiment's result
    │       └── run.log              <- log of experiment
    │
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── _Pipeline.py   <- run pipeline load -> learn -> evaluate, controlled
		│                     by parameters.json
        │
        ├── load           <- Only need if processed data should be generated
		│                     from raw data
        │
        ├── build          <- ML pipelines with standardized interfaces
        │   └── _BasePipe.py
        │   └── xxx.py
        │
        │── evaluate       <- evaluate by cross fold validation
        │
        │── experiments    <- automated, non standardized calls of pipeline
		│
		│── tuner          <- optuna
        │
        │── utils          <- utility methods
        │
        └── _StandardNames.py



## License

[MIT](https://choosealicense.com/licenses/mit/)


## Feedback

If you have any feedback, please reach out to us at ondrej.vaculin@thi.de

