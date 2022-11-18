# FYS-STK4155 Project 2

Welcome to our repository for project 2 in Applied Data Analysis and Machine Learning!
The purpose of this repository is to contain code to recreate the results presented in our report. Below is a short guide on which codes to run, corresponding to the tasks in our project description.

	* Part a : 
		Code for the gradient descent / stochastic gradient descent task can be found in file PolynomialRegression.py, which contains both a function
		which performs the regression for a set of parameters, and a main section which reproduces our results when the file is run.
		
	* Part b, c : 
		Parameter and activation function exploration for tasks b and c are done in the very fittingly named files task_b.py and task_c.py respectively.
  
	* Part d, e :
		The classification and logistic regression tasks are handled by the aptly named files Classifiction.py and LogisticRegression.py respectively.
		They set up the neural network and perform the fit to the Wisconsin Breast Cancer data set, imported from sklearn's list of datasets.

Beyond the common dependencies such as numpy and matplotlib, our code also requires the following (more uncommon) libraries:

  * jax     - Automatic differentiation
  * sklearn - Various tools for deep learning
  * seaborn - Wrapper for matplotlib for harder-to-make plots (i.e. heatmaps)
  * tqdm    - Progress bars

Note: Due to time constraints we have two separate networks which did not merge, but they together in unison produce our results :)
