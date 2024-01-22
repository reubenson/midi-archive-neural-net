import onnxruntime
import numpy as np
import boto3
from botocore.exceptions import ClientError
import logging
import json

vocab_size = 354 # hard-coding for now, but won't change often

def get_previous_tokens():
    token_filepath = '/tmp/previous_token_sequence.json'
    download_from_s3('neural-net/token_sequence.json', token_filepath)
    with open(token_filepath, 'r') as f:
        prev_tokens = json.load(f)
    
    return prev_tokens

def download_from_s3(file_key, save_path):
    bucket_name = 'midi-archive'


    # Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket_name, file_key, save_path)
    except ClientError as e:
        logging.error(e)
        return False
    return True

# https://medium.com/mlearning-ai/softmax-temperature-5492e4007f71
def softmax_temp(x, temperature):
    return(np.exp(x/temperature)/np.exp(x/temperature).sum())
    
def generate_token(logit, temperature):
    # 0.9999 factor is a hacky solution to probability adding up to > 1 due to float math
    probability = 0.99999 * softmax_temp(logit, temperature)
    val = np.random.multinomial(1, probability)
    token = val.tolist().index(1)
    return token

def generate_tokens(session):
    input_shape = session.get_inputs()[0].shape
    batch_size, block_size = input_shape
    num_tokens = 10000 # translates roughly to 3-4min

    prev_tokens = get_previous_tokens()
    context = prev_tokens[-block_size:]
    # context = np.random.randint(vocab_size, size=block_size)
    outputs = []
    for i in range(num_tokens):
        logits = session.run(None, {'input': [context]})[0]
        last_logit = logits[0, -1, :] # grab last timestep
        token = generate_token(last_logit, 1.15)
        outputs.append(token)
        context = np.append(context, token)
        context = context[-block_size:]
    return outputs

def lambda_handler(event, context):
    session = onnxruntime.InferenceSession('/opt/model.onnx', None)
    tokens = generate_tokens(session)

    return {
        'statusCode': 200,
        'body': tokens
    }