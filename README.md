# Predicting Future Occupancy at Cascade Vacation Rentals

## Confidentiality
As part of this consulting project, I was given access to private data from Cascad Vacation Rentals.

If you need to access the project or the demo of web applications, please contact me at youngsun{@}gmail.com

## Business Understanding
As a small business, Cascade Vacation Rentals (CVR) has the aspiration to use Data Science to drive their business decisions. One such candadate is prediction of future occupancy for a given property/day. This metric is important especially during the peak seasons as based on the probability threshold, business can decide to change its daily rental rate. Currently, this process is manual and based solely on last full calendar year's performance.

## Data Understanding
There are two main sources of data.

#### LiveRez.com
LiveRez.com is where sales report can be obtained. This report contains when a rental starts and it also contains how many nights the rental will last. This is the only source of occupancy information.
2012 was the last year of historical data retrieved from LiveRez.com

#### Business Website: cascadevacationrentals.com
Business reservation site is where features pertaining to the property such as number of bedrooms, number of bathrooms, number of allowed guests etc... can be scraped. Information on the website is not easily obtainable and there is no central location where such data is stored by the business. Additionally, there is seasons (and date range), rates and minimum numer of nights information that was scraped from this site.
There are total of 155 properties managed by CVR as the time of this writing.

## Data Preparation
From ~20k records where each row contain property code, start of rental and the duration of rental, it was exploded to ~220k records that account for every calendar day per property.

![image date_engineering](/img/data_pipeline.png)

Postgresql database hosted in RDS was used extensively during data engineering and EDA process.

## Modeling

## Evaluation

## Deployment
