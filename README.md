# predicting_telco_churn
In this project we start off by downloading the Telco dataset using the acquire.py file,
for any user with the correct credentials and permissions, the env.py file was used to gain access to the information.
information was split and prepared with code using the prepare file.

we continue by asking ourselves the following.......

1) what are the main drivers of churn?

2) follow the bread crums that drive churn and make visuals?

3) what services have the highest churn levels?

Project goals:

1) Identify Churn drivers

2) Creating a detailed Readme.md file to guide the reader into a further analysis of the data.

3) a CSV file that predicts churn.

4) .py files that go into detail with how the data was prepared and cut.


- initial hypothesis was that month to month prices were above average and that was the reason why customers were leaving, we dug further into the data and realized the true drives of churn.

data dictionary

Gender transformed to gender_Male: 1 = Male, 0 = Female
Churn transforemed to churn_Yes: 1 = has churn, 0 = no churn
Internet Service type: 1 = DSL, 2 = Fiber optic, 3 = None
Contract type: 1 = Month-to-month, 2 = One year, 3 = Two year
Payment Type: 1 = Electronic check, 2 = Mailed check, 3 = Bank transfer (automatic), 4 = Credit card (automatic)
Senior citizen: 0 = Not a senior citizen 1 = is a senior citizen
project planning (lay out your process through the data science pipeline)


instructions or an explanation of how someone else can reproduce your project and findings (What would someone need to be able to recreate your project on their own?)

using seaborn as sns.barplots you can easily visualize the churn based on columns.
next:
  we can use hypothesis testing by thinking of null and alternative hypothesis, then testing them by setting a alpha numeber(recommendeed 0.05)
followed by chi2 and p testing
next:
  we get into modeling, by setting a baseline and aiming for higher acuracy in our predictive models.
  
 RandomForest, Knn, and decicion trees can all be used to get the best decicions



key findings, recommendations, and takeaways from your project.
Conclusion
- 2891 customers stayed and 1046 have left this month, resulting in 26.5% churn.
- contract types, payment types, and internet service types have the highest correlation with churn.
- Random forest tree seems the best way to predict churn.

Recomendations:
1) Push for 1 or two year contracts.

2) Try to eliminate the electronic check option for customers.

3) Fiber optic internet has a high churn rate, lets check up on the cutomers weather its due to service or connectivity.

please take note that there is a predictive model called probability_of_churn_in_telco in following link https://github.com/EribertoContreras/predicting_telco_churn to reduce churn values and predict better outcomes.
