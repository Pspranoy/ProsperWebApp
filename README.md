# ProsperWebApp

Web App Link - [ProsperWebApp](https://prosperwebapp.streamlit.app/)

# Predictive Modelling Using Social Profile in Online P2P Lending Market

The Prosper dataset contains all the transaction and member data and contains all the loans created in the calendar year 2007 and all the
listings created in calendar year 2007, the bids created for these listings, the subset of members that
created these listings and bids, and finally, the groups these members are affiliated with.
However, our interest is to predict whether the loan will be timely paid or not.


## Understanding the Dataset

The dataset we are working on is the borrower-, loan- and group-related determinants of performance predictability in an
online P2P lending market by conceptualizing financial and social strength to predict borrower rates and whether the loan would be timely paid.
The result of our empirical study, conducted using a database of 9479 completed P2P transactions in calendar year 2007, provides support for the proposed  model in this study.

- There are 81 columns in the dataset.


## Preprocessing and Analysis

Dropped the features that were not relevant to the loan acceptance target or features with more than 50% missing values.
i.e.,"InvestmentFromFriendsAmount","IncomeRange", "IncomeVerifiable", "StatedMonthlyIncome", "LoanCurrentDaysDelinquent", "LoanMonthsSinceOrigination", "LoanOriginalAmount","OpenRevolvingMonthlyPayment", "LP_CustomerPayments", "LP_CustomerPrincipalPayments", "LP_InterestandFees", "LP_ServiceFees", "LP_CollectionFees", "LP_GrossPrincipalLoss", "LP_NetPrincipalLoss", "LP_NonPrincipalRecoverypayments", "PercentFunded", "Recommendations", "InvestmentFromFriendsCount", "MonthlyLoanPayment", "OpenRevolvingAccounts", "Investors", "LoanStatus", "CurrentlyInGroup", "IsBorrowerHomeowner", "LenderYield", "ListingCategory (numeric)", "DateCreditPulled", "Term", "BorrowerRate", "BorrowerAPR", "CreditScoreRangeLower","CreditScoreRangeUpper", "PublicRecordsLast10Years", "InquiriesLast6Months", "FirstRecordedCreditLine", "TotalCreditLinespast7years","CurrentDelinquencies", "DelinquenciesLast7Years","TotalInquiries", "EmploymentStatus","Occupation","BorrowerState", "TradesNeverDelinquent (percentage)","TotalTrades", "AvailableBankcardCredit", "TradesOpenedLast6Months", 'CurrentCreditLines','RevolvingCreditBalance', 'OpenCreditLines', "PublicRecordsLast12Months", "BankcardUtilization", "AmountDelinquent","EmploymentStatusDuration", "DebtToIncomeRatio","EstimatedLoss", "CreditGrade"
     
Now, 57 columns are remaining.

**Missing values**- Imputing Null values in numerical features with their median.
		  - Imputing null values in categorical features with their mode.
		  - Imputing null values in CreditGrade features with NoData.

-Checked different values present in the target column
-Converting Multi-class data to binary class data - df_new['LoanStatus'].unique()
-Values considered for Defaulted status: LoanCurrentDaysDelinquent is more than 180
-Values Considered for Undefaulted status: LoanCurrentDaysDelinquent is less than 180



## EDA 
**Introduction:**

- The dataset comprises of 113,937 rows and 57 columns.
- Dataset loanstatus varaibales are 'Notdefaulted', 'defaulted'.

**What are the most number of borrowers CreditGrade**
	Ignoring the entries where there is no creditgrade available, most common credit grade is D followed by C and B




**Since there are so much low Credit Grades such as C and D , does it lead to a higher amount of delinquency?**
	If we will compare the count of the number of defaulted and Not defaulted loans under each category of CreditGrade and see if there is any relation 		between lower credit grades and default counts
	We can see the highest number of defaults are present in C and D credit grades so we can conclude that a lower creditGrade leads to more chances of 	default




**What is the highest number of BorrowerRate?**
	The highest number of borrower rates are observed between 0.1 and 0.2




**Since the highest number of Borrower Rate is between 0.1 and 0.2, does the highest number of Lender Yield is between 0.1 and 0.2?**
	Yes, the highest Lender Yield is observed between 0.1 and 0.2




**Is the Credit Grade really accurate? Does a higher Credit Grade leads to higher Monthly Loan Payment?As for Higher Credit Grade, we mean from Grade AA to B**
	Yes the highest monthly loan payments are observed for credit grades AA,A and B






**Here we look at the Completed Loan Status and Defaulted Rate to determine the accuracy of Credit Grade**
	As we can see least number of defaulted loans are in AA, A, and B credit grades and the highest number of completed loans are between AA to C hence, 	credit grade is accurate.






**Now we know the Credit Grade is accurate and is a tool that is used by the organization in determining the person’s creditworthiness. Now we need to understand does the ProsperScore, the custom built risk assesment system is being used in determing borrower’s rate?**
	There is a clear trend here where a high Prosper Score leads to a lower Borrower rate





**From a theoretical standpoint, if the higher ProsperScore leads to a lower Borrower Rate and Borrower Annual Percentage Rate that means the Prosper Score is being used alongside the Credit Grade in determining a person’s creditworthiness**
	As expected we see a high negative correlation between credit scores and borrower rate and borrower APR. Similarly, there is a high correlation between ProsperScore, BorrowerRate and BorrowerAPR. This indicates that a higher credit score or Prosperscore leads to a lower borrower rate


## Feature Engineering
**Feature Scaling & Dimensionality Reduction (PCA)**

***Feature Scaling***

We used StandardScalar to scale our data: StandardScaler is used to resize the distribution of the values inside each features. 

⚫So that the mean of the observed values is 0 and the Standard Deviation is 1.

⚫The values will between -1 and 1.

From sklearn.preprocessing import StandardScaler( ) 

scaler =  StandardScaler( ).fit(X_train)

rescaled = scaler.transform(X_train)

ValidationX=scaler.transform(X_valid)

• fit_transform( ) is used on the training data so that we can scale the training data and also learn the scaling parameters of that data Here, the model       built by us will learn the mean and variance of the features of the training set.

 These learned parameters are then used to scale our test data.

• transform( ) uses the same mean and variance as it is calculated from our training data to transform our test data. 

  Thus, the parameters learned by our model using the training data will help us to transform our test data.

  As we do not want to be biased with our model, but we want our test data to be a completely new and a surprise set for our model.

**Feature Extraction and Dimensionality-reduction using (PCA)**

Principal component analysis, for PCA is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information.

The idea of PCA is simple — reduce the number of variables of a data set, while preserving as much information as possible.

Due to the nature of Dataset, it was observed that performing PCA negatively affected the accuracy of our models.

So that is why we opt to leave this dimentionality reduction method.

**Splitting Data & Model Deployment**
Spiliting Data into training and testing sets

80% of the dataset is consider as the training data while the remaining is used as testing data for our machine learning models.

from sklearn.nodel selection import train test split X_train, X_test, y_train, y_test train_test_split(x, y, random_state=42, test_size =0.2)


# Model Building

**Algorithms used:**
**For Classifier for Loan default status**

• ***Logistics Regression***

• ***Naive Bayes***

**For multi regressor for Preferred ROI, Eligible Loan Amount, Preferred EMI Calculations**
* **Multi Linear Regressor**


Metrics Considered for Model Evaluation for Classifier

Accuracy, Precision, Recall, and F1 Score

• Accuracy: What proportion of actual positives and negatives is correctly classified?

• Precision: What proportion of predicted positives are truly positive?

• Recall: What proportion of actual positives is correctly classified?

• F1 Score: Harmonic mean of Precision and RecallChoosing the features

Metrics Considered for Model Evaluation for Classifier

• R2 Score:  the R2 coefficient of determination is a statistical measure of how well the regression predictions approximate the real data points

## Pipelining
**Pipelines were made by using modular code by separating numerical and categorical features and preprocessing them through seperate pipelines. Then 2 pipelines were combined and then again split into 2 pipelines for prediction through 2 different models**

**Models were then saved using pickle library**

## Model Deployment

**Streamlit**

• It is a tool that lets you create applications for your machine-learning model by using simple Python code.

• We write a Python code for our app using Streamlit; the app asks the user to enter the data. 
The app runs on a local host.

**Using Render or Streamlit Community Cloud**

• To deploy our Streamlit web application for free on Streamlit Cloud. 


