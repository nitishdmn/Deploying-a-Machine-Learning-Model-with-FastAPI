# Census Income Classifier Model Card

## Model Details

This project uses a scikit-learn random forest classifier with 150 trees, a minimum leaf size of two, and a fixed random seed of 42. The model predicts whether a person's annual income is above $50,000. A one-hot encoder and label binarizer are fitted with the model and saved alongside it so inference does not require retraining.

## Intended Use

The model is intended for educational demonstration of a classification workflow on the Census Income dataset. It is not suitable for employment, lending, insurance, public-benefit, or other consequential individual decisions.

## Training Data

The model is trained on the publicly available Census Income (Adult) dataset included at `data/census.csv`. The `salary` column is the target. A stratified 80/20 train-test split with random state 42 is used. Categorical features are one-hot encoded; numeric features are kept numeric. Leading and trailing whitespace in string values is trimmed in memory, without changing the source column names.

## Evaluation Data

The held-out 20% test partition from the same dataset is used for evaluation. It is not used to fit preprocessing or the classifier.

## Metrics

The evaluation reports precision, recall, and F1 score for the positive `>50K` class. The exact measured values are printed by `python -m starter.train_model` after training and the education-slice results are written to `slice_output.txt`. The current execution environment does not include pandas or scikit-learn, so measured values and the trained artifacts could not be generated here.

## Ethical Considerations

Income is associated with protected and sensitive attributes and with structural inequities. The dataset may contain historical measurement and representation bias. Model predictions can reproduce or amplify those biases. The model should not be used to make or justify decisions about individuals.

## Caveats and Recommendations

Performance may vary across demographic groups and categorical slices. Review `slice_output.txt` and conduct a broader fairness assessment before any research use. The training dataset is a simplified benchmark and should not be assumed to represent current populations or economic conditions.
