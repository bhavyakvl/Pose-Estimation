import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials and other configurations from environment variables
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
bucket_name = os.getenv('AWS_BUCKET_NAME')
region_name = os.getenv('AWS_REGION')

# Initialize the S3 client
s3 = boto3.client('s3', aws_access_key_id=aws_access_key,
                  aws_secret_access_key=aws_secret_key,
                  region_name=region_name)

def upload_to_s3(local_file, bucket, s3_file):
    try:
        s3.upload_file(local_file, bucket, s3_file)
        return True
    except FileNotFoundError:
        print(f"The file {local_file} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

# Upload training videos
training_videos_dir = 'C:/Users/KMS/Desktop/Poses'
success_count = 0
total_count = 0

for subdir, _, files in os.walk(training_videos_dir):
    for filename in files:
        if filename.endswith('.mp4'):
            local_file = os.path.join(subdir, filename)
            s3_file = os.path.relpath(local_file, training_videos_dir)
            if upload_to_s3(local_file, bucket_name, f'training_videos/{s3_file}'):
                success_count += 1
            total_count += 1

print(f"Upload Successful for {success_count} out of {total_count} training videos")

# Upload the model
model_path = "exercise_classification_model.h5"
if upload_to_s3(model_path, bucket_name, 'models/exercise_classification_model.h5'):
    print("Model upload successful")
else:
    print("Model upload failed")
