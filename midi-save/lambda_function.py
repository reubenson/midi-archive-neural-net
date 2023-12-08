from miditok import Structured, TokenizerConfig, MIDITokenizer
import boto3
import logging
from botocore.exceptions import ClientError
from datetime import datetime

def upload_to_s3(bucket_name, file_name, object_name=None):
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
        response = s3_client.upload_file(file_name, bucket_name, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

# TO DO: export tokenizer along with model, instead of copying over here
def load_tokenizer():
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 4): 8, (4, 12): 4},
        "num_velocities": 2,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": True,
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": False,
        "use_programs": False,
        "num_tempos": 4,  # nb of tempo bins
        "tempo_range": (40, 250),  # (min, max),
        "one_token_stream": True,
        "one_token_stream_for_programs": True,
        "use_programs": True
    }
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    
    tokenizer = Structured(config)
    return tokenizer
    
def save_tokens_to_midi(tokens):
    tokenizer = load_tokenizer()
    result = tokenizer(tokens)
    result.dump('/tmp/midi-sequence.mid')
    current_date = datetime.now().date()
    filepath = f'neural-net/{current_date}_sequence.mid'
    upload_to_s3('midi-archive', '/tmp/midi-sequence.mid', filepath)

def lambda_handler(event, context):
    tokens = event["body"]
    save_tokens_to_midi(tokens)
    
    return {
        'statusCode': 200,
        'body': 'Message sent from the MIDI Archive neural net <3'
    }