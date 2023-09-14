#  Stroke Prediction Model

## Project Overview:
Strokes are a condition that can strike anyone without a warning. It is beneficial to find a way to determine someone's risk of getting a stroke.
My target is to train a model using different features that can be entered into the system by an individual to evaluate their health and assess if they are at risk of having a stroke.

### Dataset:
The dataset used was obtained from Kaggle. The dataset contains 11 columns and 4981 rows.
### Project Description:
This is a supervised machine learning project where I used "Brain Stroke" dataset to create a model that can predict if a patient is at risk of having a stroke. This model is designed to predict strokes in adults only.

### EDA and Data Cleaning
During the EDA, I found out that the dataset contains patients under the age of 18. There was no null values in the data and the data was imbalanced, the number of patients with a stroke made up 6% of the data.

### The preprocessing 
1- Removing patients under 18 years old
2- Balancing the data by repetition technique.
3- Encoding the categorical features using one hot encoding
4- Assigning X and Y values
5- Spliting the data to training and testing data

### Training the Model
Creating a pipeline. The pipeline steps are:
1- Scaling
2- Features which include PCA and SelecKBest
3- Random forest classifier

### Saving the model
I used pickle to save the trained model.

### Cross validation
I performed cross validation of the model using a new set of data that I found on Kaggle.

### Conclusion
The model has good accuracy and precision scores
### Challenges:
I couldn't find a free host for my webpage because of the size of the file, mainly because I was using sklearn.
### Future work
I hope I will be able to collaborate with a biochemist in the hospital where I used to work to get a new data with more features and try to make the model work better in prediction.