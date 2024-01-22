# MIDI-Archive Lambda
This repo is defines two AWS Lambda functions and contains the Jupyter notebook used to train the model. The first Lambda is used to generate prediction tokens from an ML inference model trained on MIDI files hosted by the companion Github repo, [/midi-archive](https://github.com/reubenson/midi-archive). The second Lambda takes those tokens, converts it to MIDI, and saves the file to an S3 Bucket. These two Lambdas are chained together in prod via an AWS Step Function.

The ML model follows a simple decoder-only transformer model, built on the PyTorch library, and is based on Andrej Karpathy's [Neural Net lecture series](https://karpathy.ai/zero-to-hero.html).

Created while I was at [Recurse Center](https://recurse.com/) in the fall of 2023!

## About
I built this project as an exploration of low-cost deployment of ML models. I decided to use Lambda (as opposed to Azure, Sagemaker, EC2, etc) because of the resource limitations and ease of scaling down uptime, and thought it would be interesting to use these limitations to explore tradeoffs in training the model. For example, the more sophisticated I make the model (increased number of parameters), the longer it takes to generate a sequence of tokens. My current plan is to only have the Lambda run once a day (staying within AWS's free tier), and since Lambda functions are limited to 15 minutes of execution time, that means a more sophisticated model may end up generating less than a minute of music. At the moment, I'm still balancing these tradeoffs.

For more sophisticated approaches to MIDI Machine Learning, check out [this article on using GPT-2 with MIDI](https://huggingface.co/blog/juancopi81/using-hugging-face-to-train-a-gpt-2-model-for-musi), or [this huggingface model](https://huggingface.co/krasserm/perceiver-ar-sam-giant-midi)

Also, see [MIDI Archive](https://github.com/reubenson/midi-archive) for more notes on the underlying training set used for this model. While current state of the art is focused on text, there's also already been considerable work done on MIDI, including OpenAI's [Musenet](https://openai.com/research/musenet) and Magenta's [Music Transformer](https://magenta.tensorflow.org/music-transformer). Both of these models use a MIDI training set called [Maestro](https://magenta.tensorflow.org/datasets/maestro) that often comes up as a canonic MIDI archive. However, developing the training set is a critical step in the design of ML systems, and I wanted to take the opportunity to both explore the design decisions that go into developing a ML model alongside an archive.

## Flowchart
The following is a general workflow diagram of the larger MIDI Archive project
```mermaid
flowchart TD
    flowchart TD
    A(Scrape MIDI files from early web) --> Z(Present model outout and interactive MIDI archive on Eleventy static site)
    A --> B(Process MIDI into tokens for training neural net)
    %% B --> L(Process MIDI into tokens for training neural net and upload to S3 Bucket)
    B --> |Switch to Colab|C(Fetch tokens and tokenizer)
    C --> D(Train neural net model)
    D --> E(Export model to ONNX)
    E --> |Done with Colab|F(Deploy to AWS Lambda)
    %% F --> X(Daily Cloudwatch trigger)
    F --> |Daily Cloudwatch Trigger|G(Generate MIDI sequence and save to S3)
    %% B --> D(Load onto MIDI Archive website)
    %% B --> I(Browser client connects via websockets to Lambda to receive MIDI broadcast)
    %% G --> H(Cloudwatch-triggered Lambda reads MIDI file from S3) --> Z
    %% G-->H
    G --> Z
```

### Training Notebook
Notebook for training and exporting the model in the repo [at `/training-notebook`](https://github.com/reubenson/midi-archive-lambda/tree/main/training-notebook), and on [Colab](https://colab.research.google.com/drive/1hpzG6ygsn0Cv44ImhyOn13eHtSo_Lccg#scrollTo=2BEEaoBHBQ1K)


## Workflow
- `MIDI Archive` repo is responsible for furnishing the training set in the form of .midi files and tokenized .json files
- In this repo, run the notebook to generate tokenizer config and tokenized dataset
- Assuming lack of access to local compute resources, run training on Colab, using pre-tokenized dataset
- After training, export the trained model to ONNX
- use the deploy script in `/midi-save` to push the model up to an AWS Lambda Layer
- update the layer version in the Lambda function
- updated model is now deployed!

<!-- ## Installation -->
<!-- Follow instrictions at https://github.com/nficano/python-lambda, which is the repo this project follows. Unfortunately, its releases are lagging, and an [important update](https://github.com/nficano/python-lambda/pull/714) has not made its way into the official package distribution. Until then, I'm running a local version of the repo: `pip install -e ../python-lambda`, in order for `lambda deploy` to work as expected -->

<!-- ## Commands -->
<!-- Zip up and deploy lambda with `lambda deploy --requirements ./requirements.txt`. The requirements flag is useful because I've been having weird issues with the installer pulling in all kinds of things from venv -->

## Misc notes
- Used `docker run --rm -v $(pwd):/package highpoints/aws-lambda-layer-zip-builder miditok` to generate .zip with miditok dependency (https://github.com/HighPoint/aws-lambda-onnx-model)
<!-- - To get the Lambda working properly, I had to fuss with permissions a bunch [here](https://us-east-1.console.aws.amazon.com/iam/home?region=us-east-1#/roles/details/lambda_basic_execution?section=permissions) -->

## References
- [As an alternative to the approach taken here, Docker and ECR could instead be used to manage Lambda deployment](https://www.serverless.com/blog/deploying-pytorch-model-as-a-serverless-service)
- [Example of using an inference model exported to ONNX](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb)
- [PyTorch's example on exporting to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [Short article on the benefits of exporting to ONNX](https://medium.com/tr-labs-ml-engineering-blog/model-deployment-with-onnx-7b45b82da71c)