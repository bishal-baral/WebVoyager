#!/bin/bash
# nohup python -u auto_eval.py \
#     --api_key YOUR_OPENAI_API_KEY \
#     --process_dir ../results/examples \
#     --max_attached_imgs 15 > evaluation.log &


#!/bin/bash
nohup python -u auto_eval.py \
    --model_type <model_type> \
    --gcp_project <project_name> \
    --gcp_location us-central1 \
    --process_dir ../results/examples \
    --max_attached_imgs 15 > evaluation.log &
