# Deep Learning Training Process with Real-Time Visualization
This Python code demonstrates a complete deep learning training process using PyTorch for predicting car park occupancy. The process involves data preprocessing, model creation, training loop execution, and real-time visualization of the training loss and model predictions.

## Data Preprocessing:
__Data Loading__: The code loads data from a CSV file named "parking_1h.csv" using the Pandas library. It filters the dataset to select records for car park ID 51 and sorts them by the ID.

__Feature Engineering__: The "current_carpark_full_total" column values are normalized to ensure consistent input ranges for the neural network.

__Data Splitting__: The preprocessed data is split into training and testing sets. The training set contains 90% of the data, and the testing set contains the remaining 10%.

## Model Definition:
__Neural Network Architecture__: The code defines a neural network model named ModelBase. The model consists of several fully connected layers with different activation functions (sigmoid and tanh) to capture complex patterns in the data.
## Training Process:
1. Training Loop: The model is trained using an Adam optimizer and mean squared error loss. The training loop runs for 200 epochs. For each epoch, the model is trained using batches of data.

2. __Real-Time Visualization__: During training, the code dynamically updates two plots in real-time:

* Loss Plot: Displays the training loss over epochs, showing how the loss decreases during training.
* Prediction Plot: Displays the model predictions vs. actual values for a subset of the testing data. The prediction plot is updated at regular intervals during training.
3. Interactive Mode: The Matplotlib interactive mode is used to enable real-time visualization.

## Key Components:
1. Data Preprocessing: Pandas for data loading and preprocessing, NumPy for numerical operations.
2. Deep Learning: PyTorch for defining and training the neural network.
3. Visualization: Matplotlib for creating real-time plots.
This code provides a comprehensive example of a deep learning training process with live visualization, allowing for a better understanding of how the model evolves and improves over epochs.
