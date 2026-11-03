# food-delivery-cnn
Predicting Fast vs Delayed food deliveries with a CNN on map-image representations, compared with Logistic Regression using cross-validation and randomized search
# Food Delivery Time Prediction – CNN and Model Validation

Predicting whether a food delivery will be **Fast** or **Delayed** using a **Convolutional Neural Network (CNN)** on map-image representations of each delivery, and comparing it with **Logistic Regression** using cross-validation and randomized hyperparameter search.

## Project Workflow

**Phase 1 – Data Preprocessing and Feature Engineering**
- Missing-value handling and one-hot encoding of weather, traffic, vehicle, priority and order time
- Parsing of `(lat, lng)` location text and **Haversine distance**
- Rush-hour vs non-rush-hour feature from the order time
- Normalized Distance and Delivery Time
- Weather and traffic impact analysis (Kruskal-Wallis tests)
- Data-quality check: provided Distance vs. distance computed from coordinates

**Phase 2 – Convolutional Neural Network**
- Each delivery is converted into a small **map image**: restaurant, customer and route channels plus delivery-detail channels
- 2D CNN (convolution, pooling and dense layers) classifying Fast vs Delayed
- Hyperparameter search with `RandomizedSearchCV` over filters, kernel size, activation, learning rate and dropout
- Compared against Logistic Regression and a majority-class baseline

**Phase 3 – Evaluation and Validation**
- 5-fold stratified cross-validation (accuracy, precision, recall, F1, ROC AUC) with a paired t-test
- Confusion matrices and ROC curves on a held-out test set
- Test-set accuracy with 95% confidence intervals and a binomial test against 50%

## Method Notes

- **Label:** the dataset has no Fast/Delayed column, so a delivery is *Delayed* if its delivery time is above the median (balanced classes; chance level = 50%).
- Delivery time is never a model input.
- Imputation, scaling and map extent are learned from training data only; hyperparameters are chosen by cross-validation on the training set; the test set is used once.

## Results

*(Fill in after running the notebook on the real dataset.)*

| Model | CV Accuracy | CV ROC AUC | Test Accuracy | Test F1 |
|---|---|---|---|---|
| Baseline CNN | – | – | – | – |
| Tuned CNN | – | – | – | – |
| Logistic Regression | – | – | – | – |
| Majority class | – | – | – | – |

## Limitations

- With about 200 deliveries a CNN has very little to learn from, and test-set scores on 40 rows are very noisy.
- The map images are schematic straight-line maps, not real map tiles or road routes.
- The dataset has no temperature or humidity columns, so weather is analyzed by category.
- An accuracy near 50% on balanced classes is chance level.

## Dataset

`Food_Delivery_Time_Prediction.csv` should be placed in the project root next to the notebook.
Download link: https://drive.google.com/file/d/1gAJGXBob3zRpJkRWQhvf3a94PS6NhGjx/view?usp=sharing

## How to Run

```bash
git clone https://github.com/ramachandran-s7rs/food-delivery-cnn.git
cd food-delivery-cnn
pip install -r requirements.txt
jupyter notebook food_delivery_cnn.ipynb
```

Python 3.10–3.12 is recommended (required for TensorFlow).

## Tech Stack

Python, pandas, NumPy, SciPy, scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn
