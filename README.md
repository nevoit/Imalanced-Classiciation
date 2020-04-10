# Imbalanced-Classification
In this assignment we are given a dataset of clinical trials. We must predict if the patient died or had a cardiovascular event.
The first thing we noticed is how imbalanced the data is - over 93% of the entries are classified as 'False'.


## Authors
* **Tomer Shahar** - [Tomer Shahar](https://github.com/Tomer-Shahar)
* **Nevo Itzhak** - [Nevo Itzhak](https://github.com/nevoit)


## Machine Learning - Assignment 3 - Requirements
In this exercise, you are given (a part of) a data set from a clinical trial. You will need to use 3 types of prediction models and compare between.
The first model should be a linear classifier.
The second one should be an ensemble model.
The third should be a DL one.

Use the concepts we learn in class: train/validation/test split, grid-search, cross validation etc.
Notice, there is no need to implement the models, you can use scikit ones.

Your submission should be a jupyter notebook that has the next sections:
Loading the data (clean, transform if needed). Then train the three models. Finally, a comparison between the three: quantitative and qualitative (think what is the evaluation method you use).

You are encourage to share your performance on the forum, but please don't share the name of the algorithm you used, nor the parameters that were used.

The data has 30 features:

- INTENSIVE - The clinical arm that was used (intensive or regular)

- NEWSITEID - Site ID in which the participant was treated

- RISK10YRS - Predicted risk for cardio vascular diseases

- INCLUSIONFRS - Binary risk group

- SBP - Systolic blood preassure

- DBP - Diastolic blood preassure

- N_AGENTS - Number of medications prescribed

- NOAGENTS - Participants on no anti-hypertensive agents

- smoke_3cat - Derived: Baseline smoking status

- aspirin - BSL Hist: Daily Aspirin Use

- egfr - Lab: eGFR MDRD (mL/min/1.73m^2)

- screat - Lab: serum creatinine, mg/dL

- sub_ckd - Derived: Subgroup with CKD (eGFR<60)

- race_black - Incl/Excl: Black, African-American

- age - Derived: Age at randomization top-coded at 90 years

- female - Derived: Female gender

- sub_cvd - Derived: subgroup with history of clinical/subclinical CVD

- sub_clinicalcvd - Derived: subgroup with history of clinical CVD

- sub_subclinicalcvd - Derived: subgroup with history of subclinical CVD

- sub_senior - Derived: subgroup ≥75 years old at randomization

- race4 - Derived: Four-level race variable (character)

- CHR - Lab: Cholesterol, mg/dL

- GLUR - Lab: Glucose, mg/dL

- HDL - Lab: HDL-cholesterol direct, mg/dL

- TRR - Lab: Triglycerides, mg/dL

- UMALCR - Lab: mg Urine Alb / (g Creat * 0.01), mg/g Cr

- BMI - Derived: body mass index (kg/m^2)

- statin - Derived: on any statin

- SBPTertile - Derived: Systolic BP tertile

- EVENT_PRIMARY - Outcome (patient died or had a cardiovascular event)

## Our Solution

Original dataset shape Counter({False: 8207, True: 539})
Percentage of True classifications: 0.06162817287903041
Percentage of False classifications: 0.9383718271209696
This imbalance has to be dealt with in the training phase. However, the first thing we must do is parse and preprocess the data. We did several things to clean it up:

1. We shuffled the data to ensure that there is no bias and the rows are not in a certain order
2. We dropped any rows where the class is undefined.
3. We dropped rows where more than 20% of the features are undefined.
4. We encoded the nominal labels.
5. For any row that has <20% missing features, we filled them in with the 10 nearest neighbors using K-nearest neighbors
6. We used a standard scaler in order to scale the data to a standard normal distribution.
7. And finally we used a SMOTE library to generate new rows with the class "True". Note that this is done ONLY on the train, and not on the test. Note that it is incorrect to use SMOTE on anything besides the training data. Using SMOTE to correct the imbalance of the test set results in synthetic entries that shouldn't exist. This is crucial because the imbalance of the test set is precisely why it is so difficult. We then perform a split into test/train.

**Running the classifiers**
Now that the data is all polished and ready to go, we built our classifiers. For each classifier we used gridsearch with Cross Validation in order to find the best hyperparameters, and then we tested the trained model provided by gridsearch on the test set.

Note that at the end of each block of code we print some quantative scores.


**Quantative Comparison**
Because of the imbalance of the test, we cannot simply judge the classifiers by their accuracy. It is quite irrelevant in our case. So instead, we compare them also by their ROC-AUC scores, which show how many false positive we get for each classifier. For example, a classifier that always returns "False" would have a very high accuracy of 93%~, but a terrible ROC because it would get false positives all the time. Additionally, we added the Precision and Recall ROC (PRAUC). These are good indicators for a classifier when we have a greatly imbalanced dataset. They can often detect differences that traditional ROC cannot. This is because ROC takes into account the True Negative Rate which is most of the dataset, while PRAUC doesn't.

Explanations here: http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf

https://www.kaggle.com/lct14558/imbalanced-data-why-you-should-not-use-roc-curve


Note that the PRAUC rate is not comparable to accuracy or ROC and is always substantially lower. However it helps us compare between different classifiers.

**Qualitative Comparison:**
Here are some qualitative comparisons between the different classifiers.

1. Easy Tuning: Logical Regression is the easiest to tune, following by Random Forest and then MLP. MLP is very difficult to tune as it involved a deep neural network architecture which we only have "rules of thumb" to follow in order to design it. MLP might have the potential to be an amazing classifier but that doesn't help us if we never find the right parameters. Random Forest is much harder to tune than LR but much easier than MLP. It is a good compromise.
2. Interpretability: Random Forest is easiest to interpret. Logistic Regression is also fine, since we can peek at the weights given to each feature to compare how important it is. MLP is of course the hardest to interpret since we cannot really explain the reason for the weights of each neuron.
3. Robustness: Logistic Regression and Random Forest are fairly robust. We didn't need to fine-tune them precisely to this dataset. MLP is very domain specific since the number of neurons in each hidden layer was derived from the number of features. The same MLP might work very bad if someone were to suddenly add or remove some features from the same dataset.
4. Data type: Random Forest can work with almost all data types and is not sensitive to this. Logistic Regression and MLP however require numerical data. 5
