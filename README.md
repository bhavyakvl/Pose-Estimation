# Fitness Tracker with Post Estimation
This project aims to classify exercises shown in videos using a deep learning model. It involves resizing videos, extracting frames, training a convolutional neural network (CNN), and predicting the exercise shown in a given video.

## Requirements
Before running the code, ensure you have the following Python packages installed:

#### opencv-python-headless: 
A version of OpenCV without GUI dependencies, useful for server environments where display functionality is not required.
#### moviepy: 
A Python library for video editing, which can handle various video file formats and perform operations like cutting, concatenating, and adding effects.
#### imgaug: 
A library for image augmentation, helpful for creating variations of images for machine learning models. It provides numerous techniques for modifying images in diverse ways.
#### boto3: 
The Amazon Web Services (AWS) SDK for Python, allowing Python developers to write software that makes use of services like Amazon S3 and Amazon EC2.
#### tensorflow: 
A deep learning framework by Google for building and training machine learning models.
#### pandas: 
A powerful data manipulation and analysis library for Python, often used for working with structured data such as spreadsheets and databases.# Fitness-Tracker-with-Pose-Estimation

## Steps
#### 1. Resizing Videos
The resize_video function resizes videos to a target resolution of 640x480. It processes all .mp4 files in the Poses directory and saves the resized videos to the ResizedVideos directory while maintaining the original directory structure.

#### 2. Extracting Frames
The extract_frames function extracts frames from the resized videos at a specified frame rate. It saves the frames in the ExtractedFrames directory, again maintaining the directory structure.

#### 3. Training the Model
The model is a convolutional neural network (CNN) with the following architecture:

Three convolutional layers with ReLU activation and max pooling.
A flatten layer to convert the 2D outputs to 1D.
A dense layer with ReLU activation and dropout for regularization.
An output dense layer with softmax activation for classification.
The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.

#### 4. Evaluating the Model
The model is trained on the extracted frames and evaluated on a test set. The accuracy and F1 score are calculated to measure the model's performance.

#### 5. Predicting Exercises
The predict_exercise function predicts the exercise shown in a single frame from a video. It preprocesses the frame, predicts the exercise using the trained model, and calculates the confidence of the prediction.
