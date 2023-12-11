
## Introduction
Due to accelerated urbanization and population growth in cities, more people have cars, increasing air pollution, commute times, accidents, and traffic congestion on major highways. To help traffic engineers, transportation planners, and government make educated decisions about traffic management rules, development of infrastructure, and public transportation planning, accurate traffic flow forecasts can offer useful insights. Traffic flow forecasting is an essential part of transportation management systems that optimize traffic flow and reduce highway congestion. 

  Data plays a vital role in decision-making and mitigating risks by foreseeing the future based on predictions made on the data available. Due to technological advancements and the Internet of Things (IoT), which employs smart devices and sensors to gather data and connect it via the internet, the process of data collection has become much easier. Machine learning algorithms are used to analyze large amounts of data and make predictions for the future which aids to make decisions. There are various ways to forecast traffic flow, including statistical models and algorithms based on artificial intelligence. Machine learning algorithms are gaining popularity in traffic flow forecasting due to their capacity to detect complex patterns and relationships in traffic data. The are two types of machine learning algorithms supervised learning techniques and unsupervised learning techniques. Supervised algorithms are used to fit the models in the project. The models used are Linear regression, Lasso regression, Ridge regression Random Forest and XGBoost.   

The data used for the study is collected from the Caltrans Performance Measurement System (PeMS) website.  The freeway selected for the study Interstate 5 South (I5). The data is divided in to two part that is training data and test data. The Training data is used to fit the models and test data to validate the models.
## Data Source
The data source for this project is extracted from the Caltrans Performance Measurement System(PMS). Caltrans is an organization that monitors California freeways by working with different local cities, tolls, and different public organizations to collect the data like traffic flow, freeway accidents, and vehicle speeds. The Caltrans PeMS has implanted around 40,000 detectors along the freeways of California to collect this information. For this project, we have chosen Interstate Freeway 5 (I5) South, one of California's busiest freeways that extends from Siskiyou County in northern California to San Diego in southern California. The entire twelve months of 2022 are used as training data, and January 2023 data is used for test data. The figure 1 below show the I5 freeway route map. 

![I5 Route](/Images-1/traffic/I5_1.jpg)

## Exploratory Data Analysis
The data contains hourly traffic flow on the Interstate 5 freeway collected by the Caltrans PeMS through the detectors. The dataset has five columns, namely, 

Date and time (Hour):- It contains the date and time of the data collected. 

Vehicles Miles Travelled (VMT(Veh_Miles)):- It is the total distance driven by cars in a specific location or at a particular time frame. 

Vehicles Hours Travelled (VHT(Veh_Hours)): - IT is the total amount of time that vehicles have traveled on a specific roadway segment or network. 

Lane Points: - Lane points describe the various lane configurations at a specific sensor station. 

%Observed: - The proportion of time throughout the measurement interval that the traffic sensor station recorded accurate data. 

The data set has been extracted from the website as a .xlsx file and loaded into python and converted into a data frame. Table 1 below shows the raw dataset description from python. 

![](/Images-1/traffic/table-1.png)

Table 2 shows the statistical analysis of the dataset. It shows the columns' total number of rows, mean, and standard deviation. It also contains the minimum, maximum, and interquartile values, which explain the spread of the attributes in the dataset. 

![](/Images-1/traffic/table-2.png)

## Data Cleaning

The cleaning of the dataset is one of the essential parts of data analysis to arrive at correct conclusions. The cleaning process involves checking for duplicate and missing values in the dataset and replacing them with appropriate ones. The table 3 below shows the missing values output of the dataset in python.

![](/Images-1/traffic/table-3.png)

## Data Transformation

The data transformation is the critical step in preparing the data for analysis. It involves many sub-steps like data encoding, data integration, feature extraction, etc. As we can see, the Hour column from the dataset contains Day, month, year, and Time so we need to separate Day and Month and create new columns for the analysis. The data encoding is used to convert the categorical variables into numerical variables. The Table 4 below shows the dataset after all data transformation steps have been performed.

![](/Images-1/traffic/table-4.png)

To understand the correlation between all the attributes in the dataset a correlation plot has been plotted. The figure 2 below shows the correlation plot. 

![](/Images-1/traffic
/Correlation plot.png)

The correlation plot shows that Vehicles Miles Travelled(VMT) and Vehicles Hours Travelled (VHT) have a high positive correlation of 0.9. The lane points and month have a positive correlation of 0.5. The %Observed has a negative correlation with the month and lane points. For analysis, the weekdays, i.e., Monday to Sunday, are mapped to numbers from 1 to 7, respectively. To ensure that both vehicle miles travelled (VMT) and vehicle hours travelled (VHT) are on the same scale, we standardize all data using the data rescaling method. Each data point is multiplied by a constant factor. A plot is plotted to determine on which time of the day the traffic flow is high on the I5 freeway. The figure3 below shows the plot, and the highway is the most congested traffic on Friday 3 to 4 pm. 

## Data Pre-Processing

The data must be split into train and test sets to fit the models. The 13 months of traffic flow data for Interstate Freeway 5 has been extracted from PeMS and loaded into Python. The twelve months' data of 2022 is used as training data to capture all traffic flow data for the entire year, including all traffic irregularities on public holidays, and helps to fit a good model. The January 2023 data is used as testing data.

## Methodology

Tools like Packages and libraries are handy in Python to perform analysis more efficiently. Many Libraries were used to analyze data and fit the models based on machine learning algorithms. NumPy and Pandas libraries are used to import the dataset and convert the data into a data frame to perform the function on entire columns. Seaborn and Matplotlib libraries are imported to plot the graphs and correlation heatmap. OneHotEncoder library has been imported from the sklearn pre-processing package to execute data transformation operations. Three machine learning algorithms have been chosen for the study and suggest a best-performing model for traffic flow prediction. Regression (Linear, Ridge, Lasso), Random Forest, and Extreme Gradient Boost (XGBoost) are the algorithms used to fit models. 


## Results and Conclusion

The five models are fit by following above methodology  and made predictions based on the training data.  The tabure 7  below show graphs of prediction vs actual values of all the models fitted along with the error. As we observe the line  of predicted value and actual value are almost superimposed because the of the models fitted have high accuracy. From the figure we can observe that error rate is very low for the Random Forest and XGBoost models compared to other models. 

![](/Images-1/traffic/trafic graph.png)

Initially, λ regularization parameter of ride and lasso regression is assumed to be one. With the help of a library called GridSearchCV from Sklearn, we executed hyperparameter tuning with 10-fold cross-validation. The optimal value λ regularization parameter for both Ridge and Lasso regression are 100 and 0.1, respectively. The hyper parameter tunning has been performed on the random forest model and the optimal values for the parameters has shown in below table 5. 

![](/Images-1/traffic/table-5.png)

The XGBoost model is also tuned to get higher accuracy and the corresponding parameters are shown in below table 6. The accuracy of the model has been increased from 98.9 to 99.5 percent.

![](/Images-1/traffic/table-6.png)

The table 7 show the indicate the comparison between the performance of different traffic flow prediction models fitted.  

table 7

From the table 7 we can infer the Lasso regression is the least performing model among all. The Ridge and Linear regression models have same R-squared value which tells that both models have almost same prediction accuracy. The Random Forest and XGBoost models are high performing model among all but XGBoost had outperformed random forest model. It has very low MAE(0.2241) , RMSE(0.3368) values and high R-Squared value of 0.9948. XGBoost model is best performing among all the models. 
