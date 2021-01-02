# Tabular Playground Series: Jan 2021

Practice competition to try out some new ideas.

## Usage instructions

1. Download competition data
    ```
    cd input
    ./download_data.sh
    ```
2. Create cross validation folds
    ```
    python3 src/create_folds.py --file input/train.csv --folds 5
    ```
3. Train and evaluate the model
    ```
    python3 src/train.py --model sgd_regressor --folds 5
    ```
4. Create submission file using one of the saved models
    ```
    python3 src/inference.py --model models/sgd_regressor_fold_0.pickle
    ```

