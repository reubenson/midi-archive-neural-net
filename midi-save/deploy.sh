#!/bin/bash

set -e

# zip up lambda handler code
zip -r handler.zip lambda_function.py

# update function code
aws lambda update-function-code \
  --function-name test-miditok \
  --zip-file fileb://handler.zip

exit 0