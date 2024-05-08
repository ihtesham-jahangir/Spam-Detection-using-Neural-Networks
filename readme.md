**Spam Detection Model**
**Overview**
This repository contains the source code and resources for a spam detection model built using deep learning techniques. The model is designed to accurately classify text messages as either spam or non-spam (ham).

**Features**
Trained using TensorFlow and Keras
Preprocessing pipeline with Tokenizer and Padding
Architecture includes Embedding, Flatten, and Dense layers
Binary classification with sigmoid activation function
Achieves high accuracy on test dataset
**Dataset**
The model is trained on a dataset of text messages sourced from Spam.csv. The dataset contains labeled examples of spam and ham messages, which are used for training and evaluation.

**Installation**
Clone the repository:
**bash**

git clone https://github.com/your_username/spam-detection.git
cd spam-detection
**Install the required dependencies:**

pip install -r requirements.txt
Download the dataset (spam.csv) and place it in the project directory.
**Usage**
Run the train_model.py script to train the spam detection model:

python train_model.py
After training, you can use the predict_message function in predict.py to classify new text messages:

**Code for running**

from predict import predict_message

message = "Your message here"
prediction = predict_message(message)
print(f"Prediction: {prediction}")

**Model Evaluation**
The trained model achieves an accuracy of approximately 95% on the test dataset. You can evaluate the model's performance using the evaluate_model.py script.

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License.

