
# Handwritten digit recognizer: AI Programming Assignment Overview

- Created a neural network model using USPS dataset in scikit-learn, model application displays:
  - Confusion matrix 
  - Error curve
  - Training time.  
  - Overall accuracy
- the application allows the user to write a digit and test the model. 

## Code and Resources Used

**Packages:** Tkinter, scikit-learn, matplotlib, pyttsx3

**Grid Creation Github:** https://github.com/misbah4064/A_Star_Python  



## Data format 
this application was built using the USPS dataset. Regardless, it will work with any dataset that follows the same format
- Data is a 256 digit binary representation of the image.
- Targets are single integer values 
note that training and testing data has to be in separate files as the program will NOT split the data.
## File selection

the application uses a file dialog to get the file's full path. Therefore, training and testing data don't need to be in the same file as the application.


## Model Building
The application uses a Multi-layer Perceptron classifier with a stochastic gradient descent solver. Also, alpha is set to 1e-4. Moreover, the number of hidden layers and the learning rate can be set by the user.

# Statistics
After the data is accepted and the model has been built, the application displays the Neural Network's confusion matrix, error/epoch curve, training time in seconds, and Overall accuracy.


![statistical graphs](https://i.imgur.com/my3smVj.png)

## User Drawing

this segment is a bonus.

### Grid Creation

modified grid creation to allow the drawing of a digit.

![digit drawn](https://i.imgur.com/FhxmomB.png)

### Prediction 
after the user Draws a digit on the 16x16 grid, it is read in a row by row manner where if the square is black it is a 1, if not then it is a 0. when the model predicts the digit it is spoken out using a pyttsx3 engine.


