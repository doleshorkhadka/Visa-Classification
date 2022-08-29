# WORK VISA Classification Problem Statement 

# Abstract
The most sought-after non-immigrant visa that enables foreign employees to work in speciality occupations in the United States is the H1-B visa. More over 1 million people filed for H-1B visas in 2019—this number includes new applications, renewals, and transfers of H-1Bs to other employers. More than 180,000 new candidates for H1-B were received , but only 80,000 of those were chosen in the lottery to move on to the USCIS for approval. A job applicant's employment and legal status are uncertain due to the H1-B visa application procedure, which also results in expensive legal and visa processing costs for the employer over the course of employment. For 2019, we intend to make use of the anonymized dataset that United Status Department of Labor publishes publicly and apply data science techniques to improve predictability of approval.

# Introduction
Each year, the US Department of Labor releases a dataset for several visa types, including L and H visas as well as PERM applications. The dataset contains information about the employer, such as Name, Address, whether it is H1-B dependent, and employment information, such as Job Title and projected Salary.
We begin by rejecting the null hypothesis that there is a relationship between an employer's profile and the success of a visa application. In this work, we intend to model that and forecast whether an application will be accepted or refused. But we restrict the dataset to additional visa kinds in addition to H1-B visas, the most common type of work visa for immigrants.


# Data collection

Parsing the 2019 excel data into, we found 664616 cases for H1-B applications. The dataset contains features 260 that gives information about employer and visa applicant. Out of all the features, due to the contextual important, we selected only 20 features given below:



# Data preprocessing
Dataset needed to be preprocessed before modelling was done.
- Features of the Data: CASE_STATUS,VISA_CLASS,EMPLOYER_NAME,SECONDARY_ENTITY_1,AGENT_REPRESENTING_EMPLOYER,JOB_TITLE,SOC_TITLE,SOC_CODE,NAICS_CODE,CONTINUED_EMPLOYMENT,CHANGE_PREVIOUS_EMPLOYMENT,NEW_CONCURRENT_EMPLOYMENT,CHANGE_EMPLOYER,AMENDED_PETITION,H-1B_DEPENDENT,SUPPORT_H1B,WILLFUL_VIOLATOR,WAGE_RATE_OF_PAY_FROM_1,WAGE_UNIT_OF_PAY_1,TOTAL_WORKER_POSITIONS.
- Removal of all the Unwanted Columns. As we can clearly see there were many irrelevant and reductant columns in the dataset which needed to be removed like PW_NON-OES_YEAR_10,PW_SURVEY_NAME_10, PW_OTHER_SOURCE_10, STATUTORY_BASIS, MASTERS_EXEMPTION, PUBLIC_DISCLOSURE etc. These are all the irrelevant columns There were columns with N/A values more than 96% which were also dropped. This was Done due to the fact that those columns will not help regarding the modelling and would only hinder the overall process.

- Treating NULL values: SECONDARY_ENTITY_1, AGENT_REPRESENTING_EMPLOYER, H-1B_DEPENDENT, SUPPORT_H1B, WILLFUL_VIOLATOR fro these features what we did we apply *mode* method and rest of the null values we dropped them.

- Outliers:  Removal of all the outliers in  'TOTAL_WORKER_POSITIONS','CHANGE_PREVIOUS_EMPLOYMENT' as they can Be clearly seen in the boxplot. IQR Method was used in order to drop them. 

- Finding duplicate values and treat it : many duplicates values were there so we replace the duplicate values with original ones. in CONTINUED_EMPLOYMENT and SOC_TITLE we treated the duplicate values.

- Encoding : We did label encoding on SOC_TITLE and JOB_TITLE. and for manual replacement for YES or NO values we replaced YES with 1 and NO with 0. we use these features: 'FULL_TIME_POSITION','SECONDARY_ENTITY_1','AGENT_REPRESENTING_EMPLOYER','H-1B_DEPENDENT','WILLFUL_VIOLATOR' 

# Modeling

We define our task as a binary classification problem where we predict whether the visa status will certified(1) or denied(0).
  - Logistic Regression : Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded between 0 and 1. In logistic regression, a logit transformation is applied on the odds—that is, the probability of success divided by the probability of failure. 

 - Random forest :Random forest is a commonly-used machine learning algorithm trademarked by Leo Breiman and Adele Cutler, which combines the output of multiple decision trees to reach a single result. Its ease of use and flexibility have fueled its adoption, as it handles both classification and regression problems.

# Data Visualizations

Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data. We have visualize the data check the outliers using boxplot and verified if the data is normalized. Using heatmap ,we found the correlation between each features. These techniques are very much useful when it comes reduce the more correlated variables in the dataset. Inaddition, We have checked the data balance in the dataset. It is clearly indicated that the dataset is imbalanced and the operating class is occred more often as compared to the closed class.


# Results and Discussions

## Model 1: Logistic Regression
Logistic regression was used in the biological sciences in the early twentieth century. It was then used in many social science applications. Logistic Regression is used when the dependent variable(target) is categorical. LR increases precision, But overfits the training data, sometimes to the detriment of accuracy. Combining a high precision model allows us to increase precision without decreasing accuracy. Precision increase is statistically significant, but not very large. A more effective approach for this problem would be to focus on how LR can be tuned to generalise better through more effective over/under-sampling techniques. LR models are high variance and dependent on the output of the classifier. The technique is not limited to LR. We can explore how other models can be combined using this technique. The practical aspect of this approach made it a very suitable one for at least smaller datasets. Accuracy 99%


## Model 2: Random forest
Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model."Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output. The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting. so the accuracy of the models was 99%.


#  Model Deployment
we used pickle file for both the models LR and RF we developed a webpage with html and backend we used flask and Django python framework.


# conclusion
We found the logistic model to be predicting high accuracy but hides the true negatives as it tries to fit the data. So, we believe that visa outcome is not as dependent on employer and job profile as we presumed in our null hypothesis, it has an element of random behavior in the decision. We captured individual company names, job titles and job categories to see if they are useful measures in modelling the accuracy. The result was a drop in total accuracy but higher level of predicting true negatives. We also evaluated random forest and logistic with modelling more features. In future, the wage can be made to fit using kernel trick with a higher dimension to evaluate how good it fits the data to predict the outcome. Neural network and boosting can also be used for stronger learning from logistic, Naïve Bayes, SVM and Random Forest to predict the outcome.
