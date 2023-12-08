import onnxruntime
import numpy as np

vocab_size = 352 # hard-coding for now, but won't change often

def generate_tokens(session, num_tokens):
    input_shape = session.get_inputs()[0].shape
    batch_size, block_size = input_shape
    # randint should be replaced with some actual MIDI data so it's not gibberish to start
    context = np.random.randint(vocab_size, size=block_size)
    outputs = []
    for _ in range(num_tokens):
        logits = session.run(None, {'input': [context]})[0]
        last_logit = logits[0, -1, :]
        token = int(last_logit.argmax(axis=0))
        outputs.append(token)
        context = np.append(context, token)
        context = context[-block_size:]
    return outputs

def lambda_handler(event, context):
    # number of tokens probably can just be set in the event payload
    num_tokens = 128
    session = onnxruntime.InferenceSession('/opt/model.onnx', None)
    tokens = generate_tokens(session, num_tokens)
    # tokenizer = load_tokenizer()
    
    return {
        'statusCode': 200,
        'body': tokens
    }