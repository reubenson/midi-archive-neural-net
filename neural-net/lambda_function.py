import onnxruntime
import numpy as np

vocab_size = 352 # hard-coding for now, but should update to import preconfigured tokenizer

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def generate_tokens(session):
    input_shape = session.get_inputs()[0].shape
    batch_size, block_size = input_shape
    num_tokens = 64 * block_size
    # randint should be replaced with some actual MIDI data so it's not gibberish to start
    context = np.random.randint(vocab_size, size=block_size)
    # context = seed_sequence
    outputs = []
    for _ in range(num_tokens):
        logits = session.run(None, {'input': [context]})[0]
        last_logit = logits[0, -1, :] # grab last timestep
        # 0.9999 factor is a hacky solution to probability adding up to > 1 due to float math
        probability = 0.99999 * softmax(last_logit)
        val = np.random.multinomial(1, probability)
        token = val.tolist().index(1)
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