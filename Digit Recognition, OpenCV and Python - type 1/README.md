# https://www.youtube.com/watch?v=CRLyKwSqxJY

Here's a complete example of how to set up the folder structure, preprocess the Kaggle MNIST dataset (in CSV format), train a digit recognition model using K-Nearest Neighbors (KNN), and test it. This will include loading the dataset, organizing it into folders, and writing the Python script to train and test the model.

### Folder Setup

The folder structure should look like this:

```
digit_recognition/
    ├── dataset/
    │   ├── train.csv
    │   └── test.csv
    ├── train_images/        # Optional: You can use this to store any image files for visualization
    ├── test_images/         # Optional: Store test images here
    └── digit_recognition.py # Main Python script for training and testing
```

### Step-by-Step Code Implementation

1. **Download the Kaggle MNIST Dataset**

   - Go to the Kaggle page: [MNIST Handwritten Digit Dataset](https://www.kaggle.com/competitions/digit-recognizer/data).
   - Download `train.csv` and `test.csv` and place them in the `dataset/` folder.

2. **digit_recognition.py**

This Python script will load the dataset, preprocess it, train the model using K-Nearest Neighbors (KNN), and then test the model on the test data.

```python
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the MNIST dataset
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# Split the training data into features (X) and labels (y)
X = train_data.drop('label', axis=1).values  # Image data (pixel values)
y = train_data['label'].values  # Labels (digits)

# Normalize the data by dividing by 255.0 to scale pixel values between 0 and 1
X = X / 255.0

# Split the dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training data
knn.fit(X_train, y_train)

# Predict the labels for the validation data
y_pred = knn.predict(X_val)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model (optional)
import joblib
joblib.dump(knn, 'digit_recognition_model.pkl')

# Testing on a single image from the test set (for example, the first image in test.csv)
def recognize_digit(image_data):
    image_data = image_data.reshape(1, -1)  # Flatten the image data to a 1D array
    image_data = image_data / 255.0  # Normalize pixel values to range [0, 1]
    prediction = knn.predict(image_data)
    return prediction[0]

# Let's recognize a random digit from the test data
random_index = np.random.randint(0, len(test_data))
test_image = test_data.iloc[random_index].values

predicted_digit = recognize_digit(test_image)
print(f"Predicted Digit: {predicted_digit}")

# Optional: Visualize the test image (reshaped to 28x28 pixels)
test_image_reshaped = test_image.reshape(28, 28)
plt.imshow(test_image_reshaped, cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.show()

# Optional: Visualize a few training images
def visualize_train_images(num_images=5):
    for i in range(num_images):
        image_data = X_train[i].reshape(28, 28)
        plt.imshow(image_data, cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.show()

# Uncomment to visualize some training images
# visualize_train_images()
```

### Explanation of Code:

1. **Loading Data**: 
   - `train.csv` contains both the image data (784 pixel values) and the labels (digits 0-9). We split it into `X` (image data) and `y` (labels).
   
2. **Data Preprocessing**: 
   - The pixel values are normalized by dividing them by 255.0 to scale them between 0 and 1.
   
3. **Model Training**:
   - A K-Nearest Neighbors classifier (`KNeighborsClassifier`) is used to train the model on 80% of the data.
   
4. **Model Testing**:
   - After training, the model is tested on 20% of the validation data. The accuracy score is printed.
   
5. **Saving the Model**: 
   - The trained model is saved using `joblib.dump` (optional).
   
6. **Recognition**:
   - A random test image is selected from the `test.csv` file, and the model predicts its label. The image is displayed using `matplotlib`.

7. **Visualizing Training Images**: 
   - Optionally, you can visualize a few images from the training set to verify that the images are correctly loaded.

### Requirements

You need to install the following libraries if you don’t have them already:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### Running the Script

1. **Folder Setup**: Ensure the folder structure is as mentioned above, with the `train.csv` and `test.csv` files in the `dataset/` folder.
   
2. **Running the Script**: 
   - Save the code in `digit_recognition.py`.
   - In your terminal, navigate to the project directory (where `digit_recognition.py` is located) and run:

   ```bash
   python digit_recognition.py
   ```

3. **Output**: 
   - The script will print the accuracy of the model.
   - A predicted digit and corresponding image will be displayed from the test dataset.

### Additional Notes:
- **KNN Performance**: The K-Nearest Neighbors algorithm is simple but might not be as efficient for large datasets. If you want better accuracy or faster training, consider using a more advanced model like a Convolutional Neural Network (CNN).
- **Model Improvement**: You can try hyperparameter tuning for KNN (such as changing `n_neighbors`), or use other classifiers like Support Vector Machines (SVM), Random Forest, etc.

Let me know if you need any additional help!