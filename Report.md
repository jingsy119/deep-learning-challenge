# Module 21 Report

## Overview of the Analysis

* The purpose of this analysis was to use machine learning and neural networks to help the nonprofit foundation Alphabet Soup select the applicants for funding with the best chance of success in their ventures.
* The model is to predict whether applicants will be successfully funded by Alphabet Soup by creating a binary classifier.
* The information provided include EIN and NAME (identification columns), APPLICATION_TYPE, AFFILIATION (affilated sector of industry), CLASSIFICATION (government organization classification), USE_CASE (use case for funding), ORGANIZATION, STATUS, INCOME_AMT (income classification), SPECIAL_CONSIDERATIONS, ASK_AMT (funding amount requested), IS_SUCCESSFUL (whether the money was effectively used).
* The dataset was preprocessed with the use of scikit-learn's `StandardScaler()`. Categorical variables that are rare were binned together and ecoded using `pd.get_dummies()`. Data was then split into a features array, X, and a target array, y, and the `train_test_split` function was used to split the data into training and testing datasets. Following this, a deep learning neural network model was created by defining the number of neurons and layers in the model. The model was then compiled, trained, and evaluated to calculate its loss and accuracy. Later, some methods were adopted to optimize the model to achieve a target predictive accuracy higher than 75%.  

## Results

* Data Preprocessing
  * The target variable of the model is `IS_SUCCESSFUL` which is a binary classifier of 0 and 1 indicating whether the money was used effectively.
  * The feature variables are all the other columns, including `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`.
  * `EIN` and `NAME` should be removed from the input data because they do not play a contributing role in the analysis.

* Compiling, Training, and Evaluating the Model
  * Considering the large nature of the dataset, initially 2 hidden layers were used with 80 and 30 neutrons for each layer, respectively. The model accuracy was around 72.5%.
![baseline_accuracy](image.png)
  * Later, the `keras-tuner` package was implemented for auto-optimization to determine the best combination for the number of hidden layers and neutrons for each layer along with the desirable activation function. After 443 running trials, it was found that 3 hidden layers with respectively 1, 41, and 1 neurons output the best result when "relu" activation function was used for the hidden layers and "sigmoid" function was used for the output layer for the given preprocessed data. The model accuracy closed in to 73%.
![best_accuracy](image-1.png)
  * Even after 443 trials of auto optimization (around 3 hours of computing time), the accuracy was only able to reach 73%, which is still short of the target accuracy of 75%. The improvements were not significant considering the baseline accuracy was already 72.5%. Hence, performing such extensive iterations with auto-optimization may not be a worthwhile approach.
  * The first step taken to optimize the model was to reduce the number of bins for categorial columns `APPLICATION_TYPE` and `INCOME_AMT`, which did not result in significant improvement in model accuracy. Followed by this, increasing the number of neutrons for both hidden layers was found to improve the model accuracy slightly, by over 0.1%. Lastly, with the auto-optimization method, the best model yielded 73% accuracy, which is still not as significant especially considering that this approach requires vigorous computing power.

## Summary

* After reducing the binning of three categorial columns (`APPLICATION_TYPE`, `CLASSIFICATION`, `INCOME_AMT`), the best overall result is determined using auto-optimization when "relu" function was used as the activation function for the 3 hidden layers with respectively 1, 41, and 1 neutrons while "sigmoid" function used for the output layer. The model accuracy reached 73% which still fell short of the target accuracy of 75%.
* Despite opting for extensive interations of searching for the best combination of neural network setup with auto-optimization, the target accuracy of 75% was not able to be reached. This suggests that the accuracy was not able to be significantly improved unless the data was processed further. The 9 feature variables is likely too much for the model to predict accurately. Therefore, it is recommended for future trials to firstly implement the principle component analysis (PCA) to remove the insignificant feature variables, then perform auto-optimization to determine the best nerual network setup for the model to perform more effectively.  