# Resume Projects

## Handwriting Generation with style

**Datasets:**

- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)

**WIP**

## [Knee Osteoarthritis Detection](./Detection%20of%20knee%20osteoroporosis%20with%20xAI/)

**Dataset:** [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

**Project Overview**

This project focuses on developing deep learning models to detect and classify knee osteoarthritis using medical imaging. By leveraging transfer learning and explainable AI techniques, I aim to create accurate and interpretable models for assisting in medical diagnoses. It demonstrates the application of advanced deep learning techniques in medical imaging, with a focus on balancing model performance with interpretability for real-world medical applications.

**Key Features**

- Classification Task: Detect and classify knee osteoarthritis on a scale of 0-3 (No/Minimal to Advanced)
- Deep Learning Models: Implemented using transfer learning with DenseNet, EfficientNet, EfficientNetV2, and ConvNeXt architectures
- Performance Metrics: Prioritized F2-score and Recall for medical relevance
- Dataset: 8.262 samples with class imbalance (60% in class 0)

**Methodology**

1. Data Preparation:
   - Simplified the original 0-5 scale to 0-3 for improved model performance
   - Addressed class imbalance using undersampling and data augmentation techniques

2. Model Development:
   - Utilized transfer learning with pre-trained models
   - Experimented with fine-tuning entire models vs. classifier layers only

3. Training and Optimization:
   - Implemented learning rate schedulers to mitigate overfitting
   - Explored various hyperparameters to optimize performance

4. Evaluation:
   - Primary metrics: F2-score and Recall
   - Secondary considerations: Confusion matrix and validation loss

**Results**

- Best model performance: 0.77 F2-score, 0.78 Recall
- Improved results observed when fine-tuning the entire model architecture

**Future Work**

- Complete the implementation of EfficientNetV2 and ConvNeXt models in PyTorch
- Further explore techniques to handle class imbalance
- Further experimentation with models.
- Check other models for this project like U-Net.
- Complete the implementation of xAI in PyTorch.

## [Wine Quality Prediction](./Wine%20Quality%20Prediction/)

**Dataset:** [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data)

**Project Overview:**

Experimented with 8 different machine learning models to predict wine quality on a 0-5 scale based on physicochemical properties. This project explores the application of various data science techniques to solve real-world problems in the food and beverage industry.

**Key Challenges:**

- Small dataset size, limiting the model's ability to generalize
- Highly imbalanced classes, potentially leading to biased predictions
- Complex relationship between chemical properties and perceived quality
  
**Methodology:**

1. Exploratory Data Analysis (EDA):

   - Conducted comprehensive statistical analysis to understand data distributions
   - Visualized correlations between features and wine quality
   - Identified potential outliers and their impact on the model

2. Data Preprocessing:
   - Implemented data normalization techniques to ensure consistent feature scaling
   - Applied oversampling methods (e.g., RandomOverSampler) to address class imbalance
   - Performed feature selection to identify the most influential wine characteristics

3. Model Development:

   - Experimented with various machine learning algorithms (e.g., Random Forest, XGBoost, SVM)
   - Utilized cross-validation
   - Implemented hyperparameter tuning through grid search

4. Performance Optimization:

   - Iteratively refined the model based on performance metrics (F1-score, Recall, Precision)
   - Analyzed feature importance to gain insights into key factors affecting wine quality
