# COVID 19 or common cold? Machine Learning distinguishes the symptoms you’re facing
WIth the flu-season on the horizon, having a runny-nose or simply coughing could scare someone into thinking they caught the mighty COVID-19 virus. 

Since the flu & COVID have somewhat similar symptoms, it is natural for someone to fear for their health. Our goal in this project is to predict whether a patient has COVID-19 based on a list of symptoms one is dealing with.  We will go about this using machine learning classification models. Precisely, we will pass our data set in Naive Bayes, Decision Tree, Random Forest, Linear Regression & Neural Network models. Once the models are trained, we will look at the top 3 most important features of each model & compare them to the government guidelines’ top 3 most common COVID-19 symptoms.   


### Authors
[Mohanad Arafe](https://github.com/mohanadarafe)

[Badreddine Loulidi](https://github.com/bloulidi)

### Setup
Make sure you have conda installed on your machine.
```
conda env create --name covid --file=environment.yml
conda activate covid
```

### Running the code
Once your environment is setup, simply run the following command to execute all models.
```
python run.py
```