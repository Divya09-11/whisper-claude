import os
import torch
import json
import whisper
import logging
from datetime import datetime
import boto3
from botocore.config import Config

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set inference profile ARN directly for local testing
os.environ['INFERENCE_PROFILE_ARN'] = "arn:aws:bedrock:us-east-1:010526276239:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# Get inference profile ARN
INFERENCE_PROFILE_ARN = os.environ.get('INFERENCE_PROFILE_ARN')

# Initialize Bedrock client
bedrock_config = Config(
    region_name='us-east-1'  # or your preferred region
)
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    config=bedrock_config
)

async def analyze_with_claude(transcript_text):
    """
    Analyze transcript using Claude 3.5 to get classification and summary
    """
    try:
        if not INFERENCE_PROFILE_ARN:
            raise ValueError("INFERENCE_PROFILE_ARN environment variable is not set")

        prompt = f"""You are an expert sales analyst specializing in the EdTech industry, focusing on online training and courses. Your task is to analyze sales call transcripts and generate a JSON summary of each interaction to evaluate lead quality and prioritize follow-ups.

Analyze the following sales call transcript and generate a JSON summary of the interaction:

<transcript>
{transcript_text}
</transcript>

Instructions:
1. Carefully read the transcript.
2. Identify and extract key elements such as:
   - Lead's interest level
   - Specific needs or pain points
   - Budget considerations
   - Decision-making authority
   - Timeline for decision
   - Any objections raised
   - Agreed-upon next steps
3. Based on these elements, assign a lead score between 0 and 100, with higher scores indicating higher potential.
4. Generate a JSON object summarizing the key aspects of the interaction according to the specified structure.

Important guidelines:
- Confidentiality: Omit all specific personal data like names, phone numbers, and email addresses.
- Character limit: Restrict each text field to a maximum of 100 characters.
- Maintain a professional tone in your summary.

Output format:
Generate a JSON object with the following structure:
<json>
{{
  "leadAnalysis": {{
    "interestLevel": "High or Medium or Low",
    "needsOrPainPoints": "Brief description of the lead's needs or pain points",
    "budget": "Lead's budget considerations",
    "decisionAuthority": true or false,
    "decisionTimeline": "Lead's timeline for making a decision",
    "objections": ["List of any objections raised by the lead"],
    "nextSteps": "Agreed-upon next steps"
  }},
  "leadScore": lead_score,
  "status": "COMPLETE",
  "ambiguities": ["List of any unclear or vague points in the conversation"]
}}
</json>

Before generating the JSON, please analyze the transcript within <thinking> tags. Include your identification of the interest level, needs or pain points, budget, decision authority, decision timeline, objections, next steps, and any ambiguities. Then, provide your JSON output within <json> tags.
"""

        # Prepare request for Claude 3.5
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

        # Invoke Claude 3.5 using the inference profile ARN
        response = bedrock_runtime.invoke_model(
            modelId=INFERENCE_PROFILE_ARN,  # Using inference profile ARN
            body=json.dumps(body),
            contentType='application/json',
            accept='application/json'
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        analysis_text = response_body['content'][0]['text']
        
        # Extract JSON from response
        try:
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            json_str = analysis_text[json_start:json_end]
            analysis = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Claude's response as JSON: {str(e)}")
            analysis = {
                "summary": analysis_text[:500],
                "error": "Failed to parse structured response",
                "raw_response": analysis_text
            }

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing with Claude: {str(e)}")
        raise

async def process_audio_file(s3_client, bucket, key, filename):
    """
    Process audio file from S3 and generate transcript with analysis
    """
    temp_dir = "/tmp" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else "."
    temp_audio_path = os.path.join(temp_dir, filename)
    
    try:
        logger.info(f"Downloading file {key} from bucket {bucket}")
        s3_client.download_file(bucket, key, temp_audio_path)
        
        if not os.path.exists(temp_audio_path):
            raise FileNotFoundError(f"File not found: {temp_audio_path}")
        
        file_size = os.path.getsize(temp_audio_path)
        if file_size == 0:
            raise ValueError("Downloaded file is empty")

        logger.info("Loading Whisper model")
        device = "cpu"
        model = whisper.load_model("medium").to(device)
        
        # Transcribe audio
        logger.info(f"Transcribing file: {temp_audio_path}")
        result = model.transcribe(
            temp_audio_path,
            fp16=False,
            language='en',
            task='transcribe',
            verbose=True
        )
        
        # Get analysis from Claude
        logger.info("Analyzing transcript with Claude 3.5")
        analysis = await analyze_with_claude(result["text"])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare complete result with metadata
        complete_result = {
            "id": timestamp,
            "filename": filename,
            "metadata": {
                "processed_at": timestamp,
                "audio_duration": result.get("duration", 0),
                "model_used": "Claude 3.5",
                "whisper_model": "medium"
            },
            "original_transcript": result["text"],
            "analysis": analysis
        }
        
        # Store result in S3
        result_key = f"transcripts/{timestamp}.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=result_key,
            Body=json.dumps(complete_result, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"Result saved to {result_key}")
        return complete_result
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.info(f"Cleaned up temporary file: {temp_audio_path}")
            
def validate_audio_file(file_key):
    """
    Validate if the uploaded file is an audio file and in the correct location
    """
    # Check if file is in the audio/ directory
    if not file_key.startswith('audio/'):
        logger.warning(f"Invalid file location: {file_key}")
        return False

    # List of allowed audio extensions
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
    
    # Get file extension
    file_extension = os.path.splitext(file_key.lower())[1]
    
    # Check if extension is allowed
    if file_extension not in allowed_extensions:
        logger.warning(f"Invalid file type: {file_extension}")
        return False

    return True

def get_all_transcripts():
    try:
        # List all transcript objects in S3
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='transcripts/'
        )
        
        transcripts = []
        for obj in response.get('Contents', []):
            transcript_data = s3.get_object(
                Bucket=BUCKET_NAME,
                Key=obj['Key']
            )
            transcript = json.loads(transcript_data['Body'].read())
            transcripts.append(transcript)
            
        return transcripts
    except Exception as e:
        print(f"Error retrieving transcripts: {str(e)}")
        return []

def get_transcript(transcript_id):
    try:
        transcript_key = f"transcripts/{transcript_id}.json"
        response = s3.get_object(
            Bucket=BUCKET_NAME,
            Key=transcript_key
        )
        return json.loads(response['Body'].read())
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise e

def export_transcript(transcript):
    try:
        export_filename = f"export_{transcript['id']}.txt"
        export_path = os.path.join(TEMP_DIR, export_filename)
        
        with open(export_path, 'w') as f:
            f.write(transcript['text'])
        
        # Upload to S3
        export_key = f"exports/{export_filename}"
        s3.upload_file(export_path, BUCKET_NAME, export_key)
        
        # Generate presigned URL for download
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': export_key},
            ExpiresIn=3600
        )
        
        return url
    finally:
        if os.path.exists(export_path):
            os.remove(export_path)
