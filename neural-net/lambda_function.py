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
    # current_date = datetime.now().date()
    # to_path = f'neural-net/{current_date}_sequence.mid'

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.download_file(bucket_name, file_key, save_path)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def generate_tokens(session):
    input_shape = session.get_inputs()[0].shape
    batch_size, block_size = input_shape
    num_tokens = 8192 # translates roughly to 3-4min
    # randint should be replaced with some actual MIDI data so it's not gibberish to start
    prev_tokens = get_previous_tokens()
    print(f'prev_tokens {prev_tokens}')

    context = prev_tokens[-block_size:]
    # context2 = np.random.randint(vocab_size, size=block_size)
    # print(f'context1 {context1}')
    # print(f'context2 {context2}')
    # return
    # context = seed_sequence
    outputs = []
    for i in range(num_tokens):
        logits = session.run(None, {'input': [context]})[0]
        last_logit = logits[0, -1, :] # grab last timestep
        # 0.9999 factor is a hacky solution to probability adding up to > 1 due to float math
        probability = 0.99999 * softmax(last_logit)
        val = np.random.multinomial(1, probability)
        token = val.tolist().index(1)
        outputs.append(token)
        context = np.append(context, token)
        context = context[-block_size:]
        print(f'token i: {i}')
    return outputs

def lambda_handler(event, context):
    session = onnxruntime.InferenceSession('/opt/model.onnx', None)
    tokens = generate_tokens(session)

    return {
        'statusCode': 200,
        'body': tokens
    }