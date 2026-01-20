App Rating Prediction using NLP and Machine Learning
1. Problem Overview
The objective of this project is to predict user ratings (ranging from 1 to 5) for a Google Play Store application based on textual reviews and version metadata. The core challenge lies in extracting sentiment from informal language (slang, Hinglish, emojis) and mapping it to a discrete numerical scale.

Dataset Specifications:
Training Set: 5,693 rows
Test Set: 1,424 rows

Features: Review Text, Review Title (if present), App Version Code, App Version Name.
Target: Star Rating (Integer 1-5).

2. Approach and Modeling Technique
We treated this as a Supervised Multi-class Classification problem. The approach follows a modular pipeline:
Text Cleaning: Standardizing raw reviews into a clean format for vectorization.
Hybrid Feature Engineering: Combining semantic features from text with structural metadata (App Version).
Modeling: We selected the Random Forest Classifier as our primary model. It is particularly effective for this task because:
It handles high-dimensional sparse data (from TF-IDF) well.
It can naturally capture non-linear relationships between a specific "App Version" and a drop in ratings.
The class_weight='balanced' parameter helps mitigate imbalances in rating distributions (e.g., fewer 2-star ratings compared to 5-star).

3. Feature Extraction Strategy
To capture both the context and the specific keywords used by users, we employed a two-pronged strategy:
TF-IDF Vectorization (n-grams): * Used Uni-grams and Bi-grams (e.g., "very bad", "not working") to capture sentiment phrases.
Limited to the top 5,000 features to prevent overfitting and maintain computational efficiency.

Metadata Integration: 
The App Version Code was treated as a numerical feature. This allows the model to learn if specific software versions are correlated with negative reviews (buggy releases).

Text Preprocessing:Removal of special characters and numbers.
Lemmatization: Reducing words to their root form (e.g., "paying" $\rightarrow$ "pay").
Stopword Removal: Filtering out high-frequency words that lack sentiment value.

4. Validation Methodology
To ensure the model generalizes well to unseen reviews, we implemented a robust internal validation framework:Stratified K-Fold Cross-Validation: We used $K=5$ folds.
"Stratified" ensures that each fold maintains the same percentage of 1-5 star ratings as the original dataset, preventing bias.

Hyperparameter Tuning: Used GridSearchCV to optimize:
n_estimators: Number of decision trees.
max_depth: Complexity of the trees.
min_samples_split: Minimum data points required to split a node.
Evaluation Metrics: Beyond simple accuracy, we analyzed the Precision, Recall, and F1-Score per class to ensure the model wasn't just guessing the most frequent rating.

5. Instructions to Run the Code
Prerequisites
Ensure you have Python 3.8+ installed.

Install the required dependencies:
pip install -r requirements.txt

Execution Steps
Place Data: Ensure train.csv and test.csv are in the project directory (or update the paths in the script).
Run Pipeline: Execute the main Python script:
python main.py

Outputs:
The script will print the Best Parameters found during tuning.
An internal Classification Report will be displayed.
A file named final_predictions.csv will be generated containing the predicted ratings for the test set.
