# Titanic_Challenge_Kaggle

## Description
Script built to try and solve the Titanic Challenge, one of the many competitions that we can find in Kaggle.
The proposal was to build a classification model, using one of the provided data files, to predict who survived the sinking of the Titanic. Then another data file was provided so that we could test our model and analyze its performance.

## Used Technologies
* **Python** - The source code is written in Python
* **Jupyter Notebook** - The platform of choice to implement this project
* **Python libraries** - Numpy, Pandas, Matplotlib, Scikit Learn, Seaborn, Joblib

## Steps followed
1. **Getting the data ready**
    - Cleaning the data (evaluate which attributes are and are not necessary for the analysis)
    - Filling (also called imputing) or disregarding missing values
    - Converting non-numerical values to numerical values (also called feature encoding)
2. **Choosing the right algorithm for the problem**
    - Selecting the algorithms based on the Scikit Learn machine learning map (https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
    - Implementing the models
    - Evaluating the models
    - Choosing the best one, based on the evaluation metrics
3. **Improving the model**
    - Using RandomizedSearchCV to discover the best values for the model hyperparameters
    - Evaluating the improved model using evaluation metrics
4. **Making the predictons**
    - Making the predictions with the improved model
    - Formating the submission dataframe and saving it into a .csv file


**Bonus - Saving the model**: Using the Joblib library

## Preview of the analysis

This is a sample of the dataset before the first step.

![dataset_before_cleaning](https://user-images.githubusercontent.com/46689116/219474902-3aca47d4-59fd-4d59-b2e9-80699086de48.png)

This is the heat map used to evaluate the correlation between attributes, helping in the process of determining which attribute would be used in the analysis.

![heatmap](https://user-images.githubusercontent.com/46689116/219474961-1cb3c736-61d6-4f56-82ef-18619387e7f4.png)

We can see that the attributes that were not numeric do not appear in the heatmap. Some of them, like Name or Ticket, do not impact the analysis. So we could drop them instead of encoding them. Others, like Sex, were relevant to the analysis, so they went through the process of encoding, that is, turning non-numerical attributes into numerical ones.

This is a sample of the dataset after the first step.

![dataset_after_cleaning](https://user-images.githubusercontent.com/46689116/219476038-d3cc4ac3-e177-4433-974b-29877dcd2136.png)

Here we have the dataset cleaned, with all the missing values filled and the non-numerical attributes turned into numerical ones.

**The evaluation of the chosen model.**

The model chosen to make the predictions was the Random Forest Classifier. Although other models could be used to do the analysis, this one was chosen for a few reasons. After testing four differente models — LinearSVC, K-Nearest Neighboor,Decision Tree Classifier, and Random Forest Classifier —, we evaluated those models. The one with the best performance was the Random Forest Classifier, with higher values for the evaluation metrics than the others. Another important factor is that the Random Forest Classifier is a method that ensembles other methods, in order to be more accurated making the predictions. In addition, the Random Forest Classifier presented a higher value for the F1-Score metric, a relevant metric to evaluate classification models, given that it combine two other metrics: recall and precision.

1. The classification report from the Scikit Learn library, _metrics_

![report_choosen_model](https://user-images.githubusercontent.com/46689116/219475102-19f6f9fe-9f85-41f7-8c42-897479a882a7.png)

Here we have the metrics such as Accuracy, Precision and Recall. All of them has values that go above 0.70, higher than the other models tested.

2. And, the confusion matrix, also from _metrics_

![cm_choosen_model](https://user-images.githubusercontent.com/46689116/219475161-450fe757-e48b-430e-a709-8d83c7f610a4.png)

The confusion matrix allows us to analyze the accuracy rate of the model, showing how many times the model got the right and wrong results.

The evaluation of the improved model.
1.  The classification report from the Scikit Learn library, _metrics_

![report_improved_model](https://user-images.githubusercontent.com/46689116/219475467-8555d802-d29d-43dc-a320-b58e283c041c.png)

We can see that most of the evaluation metrics has higher values than the ones obtained from the chosen model.

## Conclusion
In this project, built to solve Kaggle's Titanic Challenge, a lot of data manipulation was possible.
From cleaning and tuning data, to improving a classification model to make more accurate predictions. This challenge required many steps to achieve the results, allowing a lot of practice in data manipulation.
The submissions made using these models (the chosen one and the improved one), obtained scores that varied from 75% to 80% in accuracy rate.
