# Fake New Predictor

This is a GitHub repository for deploying a machine learning model for Fake News prediction on Heroku. Data set used for training and testing could be found on <a href="https://www.kaggle.com/competitions/fake-news/data">Kaggle</a>, with the training set with a sample size of 20800 and a testing set with a size of 5200. Several different models has been tested and analyzed for performance (see Jupyter notebooks under the notebooks directory). The best performing model of the batch was selected for deployment.

The deployed model was set up using Flask/Heroku. (see <a href="https://fake-news-predictor-randfor.herokuapp.com/">website</a> in progress)

A test file that could be used for the web app (test.csv) could be found under data_files. The file train.csv could also be used if truncated to under 25 MB.
