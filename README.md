# Predicting Future Occupancy at Cascade Vacation Rentals
![image cvr homepage](/img/cvr_homepage.png)

## Confidentiality
As part of this consulting project, I was given access to private data from Cascad Vacation Rentals.
If you need to access the project or the demo of web applications, please contact me at youngsun{@}gmail.com

## Business Understanding
As a small business, Cascade Vacation Rentals (CVR) has the aspiration to use Data Science to drive their business decisions. One such candadate is the prediction of future occupancy its properties. This metric is important especially during the peak seasons as based on the probability threshold, business can decide to modify its daily rate for maximum revenue. Currently, the process to assess whether a property should change its daily rate is manual and based solely on last full calendar year's performance.

## Data Understanding
There are two main sources of data.

#### LiveRez.com
This is a third party vendor site that manages reservations for CVR and it is where sales reports can be obtained. 
Sales reports contains, among other metrics, a record for each rental with start date as well as its duration. This is the only source of occupancy information.
2012 was the last year of historical data retrieved from LiveRez.com

#### [Business Website](http://www.cascadevacationrentals.com):
Using Selenium, property features data was scraped from the business reservation site. 
Information on the website is not easily obtainable via other means and there is no central location where such data is stored by the business. 
Additionally, there is seasons (and its date range), rates and minimum numer of nights information per property that was scraped from the website.
There are total of 155 properties managed by CVR as the time of this writing.

## Data Preparation
From ~20k records where each row contain property name, start of rental date and the duration of rental, it was exploded to ~220k records that account for every calendar day per property.

![image date_engineering](/img/data_pipeline.png)

Postgresql database hosted in RDS was used extensively during data engineering and EDA process.

As part of data preparation, to reflect the historic daily average occupancy, each calendar day was associated with month, weeek and day number from Retail Calendar (aka [4-5-4 calendar](https://en.wikipedia.org/wiki/4–4–5_calendar)). This way, comparison of prediction vs. historical data takes into consideration seasonality and day of the week especially weekend ended up having a huge importance as a feature in the model.

## Modeling
Once data engineering tasks were completed, following classification models were used:
- Logistic Regression
- Random Forest
- Grandient Boosting Classifier

## Evaluation
Using KFold cross-validation evaluated model performance using log-loss and AUC score to compare different models.
It turns out that all three models were not too dissimilar even after tuning RF and GBC via extensive GridSearch.

![image ROC](/img/ROC.png)

However, what tip the scale in favor of GBC can be seen below:

![image model comp](/img/model_comp.png)

When compared to historic daily average occupancy, GBC came closest when compared to other models.

## Deployment

![image architecture and technology](/img/arch_tech.png)
