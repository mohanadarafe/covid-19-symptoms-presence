# Identify early stages of COVID-19
WIth the flu-season on the horizon, having a runny-nose or simply coughing could scare someone into thinking they caught the mighty COVID-19 virus. 

Since the flu & COVID have somewhat similar symptoms, it is natural for someone to fear for their health. Our goal in this project is to predict whether a patient has COVID-19 based on a list of symptoms one is dealing with.  We will go about this using machine learning classification models. Precisely, we will pass our data set in Naive Bayes (baseline), Decision Tree, Random Forest, SVM & Neural Network models. Once the models are trained, we will look at the top 3 most important features of each model & compare them to the government guidelines’ top 3 most common COVID-19 symptoms.

## Experiment
We will experiment how strong our models are by fully training one dataset & testing our models on another dataset. The only goal here is to learn!

### Authors
[Mohanad Arafe](https://github.com/mohanadarafe)

[Badreddine Loulidi](https://github.com/bloulidi)

### Data
The datasets were collected from the following Kaggle repositories.
[Main dataset](https://www.kaggle.com/hemanthhari/symptoms-and-covid-presence)
[Experiment dataset](https://www.kaggle.com/prakharsrivastava01/covid19-symptoms-dataset)

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

### Running the experiment
If you would like to see the results of the experiment, simply run the following command:
```
python experiment.py
```

### Analysis
You can find an in-depth analysis of our results in the [analysis notebook](https://github.com/mohanadarafe/covid-19-symptoms-presence/blob/main/analysis.ipynb) we created. We explain the dataset used, preprocessing tools, the performance of each model & breakdown of feature importances per model.