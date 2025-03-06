# Whisper Audio Transcription Lambda Service

## Overview
This is a serverless application that processes audio files using the Whisper model. The service automatically transcribes audio files uploaded to an S3 bucket using AWS Lambda.

## Project Structure

sam-test/
│── app/
│ ├── lambda_handler.py # Lambda function entry point
│ ├── services.py # Core business logic and audio processing
│ ├── requirements.txt # Python dependencies
│ ├── Dockerfile # Container configuration for Lambda
│── events/
│ ├── event.json # Sample test events
│── test_local.py # Local testing script
│── README.md # Documentation


## Prerequisites
- AWS CLI installed and configured
- Docker installed
- Python 3.8 or later
- An AWS S3 bucket for storing audio files
- AWS ECR repository

## Features
- Audio file processing using Whisper ML model
- Supports multiple audio formats (.mp3, .wav, .m4a, .flac, .ogg, .aac)
- Automatic file validation
- S3 integration for file storage
- Containerized deployment

## Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/Divya09-11/transcript-lambda.git

## Install dependencies
pip install -r app/requirements.txt

## create a .env file with the following variables

BUCKET_NAME=your-bucket-name
AWS_REGION=your-region

Deployment Steps
Build the Docker image:

docker build -t whisper-transcription ./app

Tag and push to Amazon ECR:

# Authenticate Docker to ECR
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# Create ECR repository (if not exists)
aws ecr create-repository --repository-name whisper-transcription

# Tag image
docker tag whisper-transcription:latest <account-id>.dkr.ecr.<region>.amazonaws.com/whisper-transcription:latest

# Push image
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/whisper-tr

## Create Lambda Function:

Go to AWS Lambda Console
Create new function
Choose "Container image" option
Select the pushed container image
Configure environment variables
Set appropriate IAM roles and permissions

## Create S3 Bucket
Using AWS Console:

Open the Amazon S3 console
Click "Create bucket"
Enter a unique bucket name
Select the AWS Region
Configure bucket settings:
    For versioning: Disabled
    For public access: Block all public access (enabled)
    For default encryption: Enable
Click "Create bucket"


## Testing
Local Testing
python test_local.py

## AWS Console Testing
Go to Lambda Console
Select your function
Create a test event using the following template:

{
  "Records": [
    {
      "eventVersion": "2.1",
      "eventSource": "aws:s3",
      "awsRegion": "us-east-1",
      "eventTime": "2023-12-20T10:00:00.000Z",
      "eventName": "ObjectCreated:Put",
      "s3": {
        "bucket": {
          "name": "your-bucket-name"
        },
        "object": {
          "key": "audio/test.mp3"
        }
      }
    }
  ]
}

## Supported Audio Formats
MP3 (.mp3)
WAV (.wav)
M4A (.m4a)
FLAC (.flac)
OGG (.ogg)
AAC (.aac)

## Technical Details
Container Configuration
Base image: Python
FFmpeg included in container
Whisper ML model
Required Python packages installed via requirements.txt
Lambda Configuration
Memory: Configure based on your needs (recommended: 6144MB or higher)
Timeout: Configure based on your needs (recommended: 5-15 minutes)

## Environment variables required:
BUCKET_NAME
AWS_REGION

## Monitoring and Logging
View logs in CloudWatch Logs
Monitor function metrics in CloudWatch
Check container image scanning results in ECR