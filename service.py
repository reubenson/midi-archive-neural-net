import os
import asyncio
import websockets
import boto3
import torch
import torch.nn as nn

active_connections = []


####################
block_size = 128
dropout = 0.2
device = 'cpu'
vocab_size = 352
n_layers = 4


class Head(torch.nn.Module):
  def __init__(self, note_dimensions, head_size):
    super().__init__()
    self.head_size = head_size
    self.key = torch.nn.Linear(note_dimensions, head_size, bias=False)
    self.query = torch.nn.Linear(note_dimensions, head_size, bias=False)
    self.value = torch.nn.Linear(note_dimensions, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, idx):
    B, T, C = idx.shape # C is head_size
    k = self.key(idx) # (B, T, C)
    q = self.query(idx) # (B, T, C)
    # weights is a weighted sum of information from connected nodes in the graph (attention scores)
    # (in this operation, each batch is kept segregated and doesn't pass information across each other)
    weights = q @ k.transpose(-2, -1) * self.head_size**0.5 # (B, T, T)
    # "decoder" block - nodes in `future` can't talk to `past`
    weights = weights.masked_fill(self.tril==0, float('-inf'))
    # softmax is needed to prevent weights from amplifying into one-hot vectors
    activations = F.softmax(weights, dim=1) # (B, T, T)

    # if not model.training:
    #   model_activations['head'].append(activations.detach())
    #   model_activations['head'] = model_activations['head'][-16:]

    activations = self.dropout(activations)

    # value is information private to the node
    # this operation allows information from adjacent nodes to influence the given node
    out = activations @ self.value(idx) # (B, T, C)
    return out

class MultiHead(torch.nn.Module):
  def __init__(self, note_dimensions, head_count):
    super().__init__()
    head_size = note_dimensions // head_count
    self.heads = torch.nn.ModuleList([Head(note_dimensions, head_size) for _ in range(head_count)])
    # projection layer for residual pathway
    self.proj = torch.nn.Linear(note_dimensions, note_dimensions)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=2)
    out = self.proj(out)
    out = self.dropout(out)

    return out

class FeedForward(torch.nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    f = 4 # factor of 4 added in Attention paper (TO DO: see if this actually makes much difference)
    self.operations = torch.nn.Sequential(
      torch.nn.Linear(n_embed, f * n_embed, device=device),
      torch.nn.ReLU(),
    #   PrintLayer(), # passthrough layer, used for capturing activations from ReLU
      torch.nn.Linear(f * n_embed, n_embed, device=device), # projection layer (feeds back into residual pathway?)
      torch.nn.Dropout(dropout)
    )

  def forward(self, x):
    out = self.operations(x)
    return out

# class PrintLayer(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, x):
#         if not model.training:
#             activation = x.detach()
#             model_activations['layer'].append(activation)
#             model_activations['layer'] = model_activations['layer'][-16:]
#         return x

class Block(torch.nn.Module):
  def __init__(self, note_dimensions, n_heads):
    super().__init__()
    # attention head produces associations between nodes within a neural net
    # head_size = note_dimensions / head_count
    # "communication" achieved through self-attention
    head_count = note_dimensions // n_heads
    self.multi_head = MultiHead(note_dimensions, head_count)

    # "computation" through feed-forward (token-wise computation, no associations with other tokens)
    self.ff = FeedForward(note_dimensions)

    # layer normalizations (to help with regularization, reduce over-fitting)
    self.ln1 = torch.nn.LayerNorm(note_dimensions, device=device)
    self.ln2 = torch.nn.LayerNorm(note_dimensions, device=device)

  def forward(self, x):
    # "x = x +" pattern here implements residual connections
    # TO DO: check to see if that actually comes through progression of training
    x = x + self.multi_head(self.ln1(x)) # (B, T, C)
    x = x + self.ff(self.ln2(x))
    return x

class Model(torch.nn.Module):
  def __init__(self, note_dimensions=16, n_heads = 4, loss_threshold = 0.5):
    super().__init__()
    # token embedding is an association between notes and their vector representation
    self.token_embedding = torch.nn.Embedding(vocab_size, note_dimensions)

    # position embedding is an association between a note and its position in time
    self.position_embedding = torch.nn.Embedding(block_size, note_dimensions)

    # model head produces predictions for the next note in a sequence
    self.model_head = torch.nn.Linear(note_dimensions, vocab_size)

    # layers of self-attention communication and computation
    self.blocks = torch.nn.Sequential(
      *[Block(note_dimensions, n_heads) for _ in range(n_layers)]
    )

    self.ln = torch.nn.LayerNorm(note_dimensions)

    # automatically halt training if loss goes below this threshold
    self.loss_threshold = loss_threshold

    # not used for model training, just for plotting figures
    # self.position_embeddings = []
    print(f"model generated with {sum(p.nelement() for p in self.parameters())} parameters")

  def forward(self, idx, targets=None):
    # B (batch_size) is the number of items in batch
    # T (block_size) is the length of data in a batch item
    # C (note_dimensions) is provided by embedding table
    B, T = idx.shape
    token_emb = self.token_embedding(idx) # (B, T, C) tokens representing note values
    position_emb = self.position_embedding(torch.arange(T, device=device)) # (T, C) token representing note positions
    # the following two operations tie the model together
    # TO DO: develop better understanding/intuition about how/why this works ...
    data = token_emb + position_emb # (B, T, C) represents note values with positions

    # decoder blocks (with self-attention)
    data = self.blocks(data)

    # final layer-norm
    data = self.ln(data)

    # logits represent levels of occurrence, which are used to calculate the predictions for next note
    logits = self.model_head(data) # (B, T, note_options_count)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(-1)
      # cross-entropy performs softmax on logits with respect to targets (expected outputs) to produce loss calculation
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, output_length):
    self.eval()
    output = []
    context = idx # (B, T)
    logits_array = []
    for _ in range(output_length):
      context = context[:, -block_size:] # constrain to block_size length
      logits, loss = self(context)
      logits = logits[:, -1, :] # grab last timestep
      # print(f"logics {logits}")
      probs = F.softmax(logits, dim=-1)
      val = torch.multinomial(probs, num_samples=1)
      # every iteration, context gets longer and the next iteration is influenced by generation history
      context = torch.cat((context, val), dim=1)
      # output.append(i_to_note(val.item()))
      output.append(val.item())
      # logits_array.append(logits)

    return output
#######################################################



def add_connection_to_loop(connection_id):
    active_connections.append(connection_id)
    print(f"active_connections: {active_connections}")
    # loop = asyncio.get_event_loop()
    # loop.create_task(send_message(connection_id))

def remove_connection_from_loop(connection_id):
    active_connections.remove(connection_id)
    print(f"active_connections: {active_connections}")
    # loop = asyncio.get_event_loop()
    # loop.remove_task(send_message(connection_id))

async def repeating_task():
    # print(f"client already defined?: {client}")
    while True:
        print("This is a repeating task.")
        print(f"active_connections: {active_connections}")
        for connection_id in active_connections:
            print(f"Sending message to {connection_id}")
            # wasteful but will it wokr???
            websocket_url = os.getenv('WS_CONNECTIONS_URL').split('@connections')[0]
            client = boto3.client('apigatewaymanagementapi', endpoint_url = websocket_url)
            try:
                client.post_to_connection(
                    Data=f'hello from lambda @@@ {connection_id}',
                    ConnectionId=connection_id
                )
            except Exception as e:
                print(f'Error sending message {e}')
        await asyncio.sleep(5)  # wait for 5 seconds before repeating

async def handle_input(input):
    print(f"Handling input: {input}")
    # handle the input here

def start_loop(client):
    loop = asyncio.get_event_loop()
    loop.create_task(repeating_task(client))
    loop.create_task(handle_input('some input'))
    loop.run_forever()

async def send_message(uri, message):
    async with websockets.connect(uri) as websocket:
        await websocket.send(message)
        response = await websocket.recv()
        print(response)

# from websockets.sync.client import connect

# def hello():
#     with connect("wss://1wtfmfef4k.execute-api.us-east-2.amazonaws.com/production/") as websocket:
#         websocket.send("Hello world!")
#         message = websocket.recv()
#         print(f"Received: {message}")

async def handle_request(eventType, connection_id):
    # handle the request here
    if eventType == 'CONNECT':
        add_connection_to_loop(connection_id)
    elif eventType == 'DISCONNECT':
        remove_connection_from_loop(connection_id)
    
    print(f"Handling request for connection: {connection_id}")
    print(f"active_connections: {active_connections}")

    return {
        'statusCode': 200,
        'body': 'Connected!!'
    }

    await asyncio.sleep(1)  # simulate IO-bound operation with sleep

async def setInterval(func, sec):
    while True:
        await func()
        await asyncio.sleep(sec)

def handler(event, context):
    model = Model().to(device)
    # model.load_state_dict(torch.load('./model.pt', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('./model.pt'))
    # with open('./model.pt', 'r') as f:
    #     content = f.read()
    #     print(f"content: {content}")

    print(f"event: {event}")
    print(f"context: {context}")
    event_type = event["requestContext"]["eventType"]
    connection_id = event["requestContext"]["connectionId"]
    websocket_url = os.getenv('WS_CONNECTIONS_URL').split('@connections')[0]

    print(f"event_type: {event_type}")

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigatewaymanagementapi.html
    client = boto3.client('apigatewaymanagementapi', endpoint_url = websocket_url)
    # start_loop(client)

    connection_id = event['requestContext']['connectionId']
    # loop = asyncio.get_event_loop()
    # loop.create_task(repeating_task(client))
    # loop.create_task(handle_request(event_type, connection_id))
    # loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop)))

    # return {
    #     'statusCode': 200,
    #     'body': 'Request handled'
    # }
    # asyncio.run(setInterval(repeating_task, 5))
    print(f"event_type: {event_type}")

    # hello()
    if event_type == "CONNECT":
        print(f"Connect established with connectionId: {connection_id}")
        add_connection_to_loop(connection_id)
        # try:
        #     asyncio.get_event_loop().run_until_complete(
        #         # send_message(websocket_url, 'Hello, World!')
        #         response = client.post_to_connection(
        #             Data='connected to lambda!',
        #             ConnectionId=connection_id
        #         )
        #     )
        # except Exception as e:
        #     print(f'Error sending message {e}')
        return {
            'statusCode': 200,
            'body': 'Connected'
        }
    elif event_type == "DISCONNECT":
        print(f"disconnected: {connection_id}")
        remove_connection_from_loop(connection_id)
        return {
            'statusCode': 200,
            'body': 'Disconnected'
        }
    else:
        print(f"inside else: {connection_id}")
        print(f"active_connections: {active_connections}")
        message = event['body']
        print(f"message: {message}")
        # add_connection_to_loop(connection_id)
        # need to wait for default event_type (not connect)
        # https://stackoverflow.com/questions/55688632/aws-api-gateway-websocket-unknownerror
        
        # the documentation is not abundantly clear, but the @connections
        # api is needed for sending messages from the lambda back to client
        # https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-websocket-api-data-from-backend.html
        # https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-how-to-call-websocket-api-connections.html
        # response = client.post_to_connection(
        #     Data='hello from lambda',
        #     ConnectionId=connection_id
        # )
        # print(f"active_connections: {active_connections}")
        # try:
        #     asyncio.get_event_loop().run_until_complete(
        #         # send_message(websocket_url, 'Hello, World!')
        #         client.post_to_connection(
        #             Data='hello from lambda',
        #             ConnectionId=connection_id
        #         )
        #     )
        # except Exception as e:
        #     print(f'Error sending message {e}')

    # print(f"response: {response}")

    return {
        'statusCode': 200,
        'body': 'Message sent from the MIDI Archive neural net <3'
    }