# Imbalanced-Classification
In this assignment we are given a dataset of clinical trials. We must predict if the patient died or had a cardiovascular event.
The first thing we noticed is how imbalanced the data is - over 93% of the entries are classified as 'False'.

Submitted by:
Tomer Shahar and Nevo Itzhak

## Exercise 3 - Requirements
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

INTENSIVE - The clinical arm that was used (intensive or regular)

NEWSITEID - Site ID in which the participant was treated

RISK10YRS - Predicted risk for cardio vascular diseases

INCLUSIONFRS - Binary risk group

SBP - Systolic blood preassure

DBP - Diastolic blood preassure

N_AGENTS - Number of medications prescribed

NOAGENTS - Participants on no anti-hypertensive agents

smoke_3cat - Derived: Baseline smoking status

aspirin - BSL Hist: Daily Aspirin Use

egfr - Lab: eGFR MDRD (mL/min/1.73m^2)

screat - Lab: serum creatinine, mg/dL

sub_ckd - Derived: Subgroup with CKD (eGFR<60)

race_black - Incl/Excl: Black, African-American

age - Derived: Age at randomization top-coded at 90 years

female - Derived: Female gender

sub_cvd - Derived: subgroup with history of clinical/subclinical CVD

sub_clinicalcvd - Derived: subgroup with history of clinical CVD

sub_subclinicalcvd - Derived: subgroup with history of subclinical CVD

sub_senior - Derived: subgroup ≥75 years old at randomization

race4 - Derived: Four-level race variable (character)

CHR - Lab: Cholesterol, mg/dL

GLUR - Lab: Glucose, mg/dL

HDL - Lab: HDL-cholesterol direct, mg/dL

TRR - Lab: Triglycerides, mg/dL

UMALCR - Lab: mg Urine Alb / (g Creat * 0.01), mg/g Cr

BMI - Derived: body mass index (kg/m^2)

statin - Derived: on any statin

SBPTertile - Derived: Systolic BP tertile

EVENT_PRIMARY - Outcome (patient died or had a cardiovascular event)
