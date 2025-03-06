import os
import sys
import json
import boto3
import asyncio
# Add both potential python paths
sys.path.append("/opt/python")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ffmpeg
    print("FFmpeg package imported successfully")
except ImportError as e:
    print(f"Error importing ffmpeg: {str(e)}")
    

os.environ['PATH'] = f"/opt/ffmpeg/bin:{os.environ['PATH']}"
# Set CPU configurations
os.environ['TORCH_DEVICE'] = 'cpu'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
torch.set_default_tensor_type('torch.FloatTensor')
from services import validate_audio_file, process_audio_file

s3_client = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')

os.environ['INFERENCE_PROFILE_ARN'] = "arn:aws:bedrock:us-east-1:010526276239:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
INFERENCE_PROFILE_ARN = os.environ.get('INFERENCE_PROFILE_ARN')

async def qa_with_claude(transcript_text, question):
    """
    Use Claude to answer questions based on the transcript context
    Returns the answer directly without storing in S3
    """
    try:
        if not INFERENCE_PROFILE_ARN:
            raise ValueError("INFERENCE_PROFILE_ARN environment variable is not set")

        prompt = f"""You are an AI assistant helping to answer questions about a transcript. 
        Please answer the following question based solely on the information provided in the transcript.

        Transcript:
        <transcript>
        {transcript_text}
        </transcript>

        Question: {question}

        Instructions:
        1. Answer the question specifically based on the information in the transcript
        2. If the answer cannot be found in the transcript, respond with "I cannot answer this question based on the transcript content"
        3. Keep the answer concise and relevant
        4. Do not make assumptions beyond what is explicitly stated in the transcript
        """

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "top_p": 0.9
        }

        response = bedrock_runtime.invoke_model(
            modelId=INFERENCE_PROFILE_ARN,
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']

        return answer

    except Exception as e:
        print(f"Error in Q&A with Claude: {str(e)}")
        raise

async def get_transcript_from_s3(s3_client, bucket, key):
    """
    Retrieve transcript from S3 bucket
    """
    try:
        print(f"Attempting to retrieve transcript from bucket: {bucket}, key: {key}")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        transcript_data = json.loads(content)
        
        # Try to get transcript from different possible keys
        transcript = transcript_data.get('original_transcript',  # Try original_transcript first
                    transcript_data.get('transcript',           # Then try transcript
                    transcript_data.get('text', '')))          # Then try text
        
        if not transcript:
            print(f"Available keys in transcript_data: {list(transcript_data.keys())}")
            print("Transcript is empty or not found in the expected format")
            return None
            
        return transcript
    except Exception as e:
        print(f"Error retrieving transcript from S3: {str(e)}")
        raise

def handler(event, context):
    return asyncio.run(lambda_handler(event, context))

async def lambda_handler(event, context):
    try:
        operation = event.get('operation', 'transcribe')
        
        if operation == 'transcribe':
            # Handle transcription operation
            records = event.get('Records', [])
            if not records:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': 'Missing S3 event records for transcription'
                    })
                }
            
            record = records[0]
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            filename = key.split('/')[-1]

            # Validate the audio file
            if not validate_audio_file(key):
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': 'Invalid file type or location. Please upload audio files to the audio/ folder.'
                    })
                }

            result = await process_audio_file(s3_client, bucket, key, filename)
            return {
                'statusCode': 200,
                'body': json.dumps(result)
            }
            
        elif operation == 'q_n_a':
            # Handle Q&A operation
            transcript_file = event.get('transcript_file')
            transcript_bucket = event.get('transcript_bucket')
            question = event.get('question')
            
            # Validate required parameters
            if not all([transcript_file, transcript_bucket, question]):
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': 'Missing required parameters: transcript_file, transcript_bucket, and question'
                    })
                }
            
            try:
                # Retrieve transcript from S3
                transcript = await get_transcript_from_s3(
                    s3_client, 
                    transcript_bucket, 
                    transcript_file
                )
                
                if not transcript:
                    return {
                        'statusCode': 404,
                        'body': json.dumps({
                            'error': 'Transcript not found or empty'
                        })
                    }
                
                # Get answer from Claude
                answer = await qa_with_claude(transcript, question)
                
                # Return the response directly
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'question': question,
                        'answer': answer,
                        'transcript_file': transcript_file
                    })
                }
                
            except Exception as e:
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'error': f'Error processing Q&A: {str(e)}'
                    })
                }
            
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': f'Invalid operation: {operation}'
                })
            }
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            }),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
