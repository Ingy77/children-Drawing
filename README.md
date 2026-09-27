Children Drawing Emotion Classification

A deep learning project that classifies children's drawings into four emotion categories: Angry, Fear, Happy, and Sad.

The project explores different transfer learning models, data augmentation techniques, model evaluation, and using the trained model in a simple Streamlit application.

About the Project

The main idea of this project is to investigate whether a deep learning model can learn visual patterns from children's drawings and classify them into predefined emotion categories.

The project covers the complete machine learning workflow:

Dataset
   ↓
Preprocessing
   ↓
Data Augmentation
   ↓
Model Training
   ↓
Validation
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Streamlit Application
Classes

The model predicts one of four classes:

Class	Description
Angry	Anger-related expression
Fear	Fear or worry-related expression
Happy	Happiness-related expression
Sad	Sadness-related expression
Dataset

The original dataset contains 702 images distributed across four classes.

Class	Images
Angry	169
Fear	179
Happy	171
Sad	183
Total	702

Data augmentation was applied to the training data to increase the amount of training data and provide more variation in the images.

The augmented training set was increased to approximately 3,900 images.

A separate test set was used for the final evaluation.

Data Augmentation

The training images were augmented using different transformations, including:

Resize
Random Horizontal Flip
Random Rotation
Brightness adjustment
Contrast adjustment
Saturation adjustment

This was done to expose the models to different variations of the drawings during training.

Models

I experimented with several pretrained deep learning architectures:

ResNet18
EfficientNet-B0
MobileNetV3-Large

The final classification layer of each model was adapted for the four emotion classes.

Results
Model	Accuracy	Macro F1
ResNet18	~68%	~0.68
EfficientNet-B0	63.88%	0.63
MobileNetV3-Large	63.88%	0.63

The results show that the task is challenging, especially when distinguishing between visually similar classes.

Evaluation

The models were evaluated using:

Accuracy
Precision
Recall
F1-score
Macro F1-score
Confusion Matrix

Macro F1 was included to evaluate performance across all four classes rather than relying only on overall accuracy.

Streamlit Application

The trained model was also prepared for use in a Streamlit application.

The application allows the user to upload a drawing and receive:

Predicted emotion
Confidence score
A short observation related to the predicted class

Example:

Predicted Emotion: Happy
Confidence: 82%

The application is designed as a simple demonstration of how a trained computer vision model can be integrated into an interactive application.

Technologies Used
Python
PyTorch
Torchvision
Scikit-learn
NumPy
Matplotlib
PIL
Google Colab
Jupyter Notebook
Streamlit
