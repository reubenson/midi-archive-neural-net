#!/bin/bash

set -e

# generate model.onnx from model in colab after training
# python model.py

# zip up lambda handler code
zip -r handler.zip lambda_function.py

# update function code
aws lambda update-function-code \
  --function-name midi-generate \
  --zip-file fileb://handler.zip

exit 0

# zip up the model file and ship it to the lambda layer
zip -r ../models/model.zip ../models/model.onnx
aws lambda publish-layer-version --layer-name onnxmodel \
  --description "MIDI Archive neural net" \
  --license-info "MIT" \
  --zip-file fileb://../models/model.zip


# zip up model.onnx and push to Lambda Layer
aws lambda publish-layer-version --layer-name midimodel \
  --description "MIDI Archive neural net" \
  --license-info "MIT" \
  --zip-file fileb://model.zip




# generate torchscript-compiled version of the pytorch model
python test.py

# rebuild ./torchlambda code
torchlambda template --yaml torchlambda.yaml

# build model execution code
# this is what will get zipped and shipped to the lambda function itself
torchlambda build ./torchlambda --compilation "-Wall -O2"

# maybe need to manually delete torchlambda.zip?

# package up the model, which will get shipped to the lambda layer
torchlambda layer ./model.ptc --destination "model.zip"

# create model (only run once ...)
#  aws lambda create-function --function-name test2 \
#   --role arn:aws:iam::646493684442:role/lambda_basic_execution \
#   --runtime provided --timeout 60 --memory-size 1024 \
#   --handler torchlambda --zip-file fileb://torchlambda.zip

# update function code
aws lambda update-function-code \
  --function-name demo \
  --zip-file fileb://torchlambda.zip

# update function configuration
# aws lambda update-function-configuration \
#   --function-name test2 \
#   --timeout 160 \
#   --layers arn:aws:lambda:us-east-1:646493684442:layer:modell:2

# update layer (contains the actual neural net model)
# (after doing this, need to update the layer version in the lambda function)
aws lambda publish-layer-version --layer-name mmm \
  --description "MIDI Archive neural net another test" \
  --license-info "MIT" \
  --zip-file fileb://model.zip
