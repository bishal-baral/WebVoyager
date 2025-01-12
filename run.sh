# #!/bin/bash
# nohup python -u auto_eval.py \
#     --model_type gemini \
#     --gcp_project <> \
#     --gcp_location us-central1 \
#     --process_dir ../results/examples \
#     --max_attached_imgs 15 > evaluation.log &

#!/bin/bash
nohup python3 -u run.py \
    --test_file ./data/tasks_test.jsonl \
    --model_type <model_type> \
    --gcp_project <project_name> \
    --gcp_location us-central1 \
    --headless \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --temperature 1 \
    --fix_box_color \
    --seed 42 > test_tasks.log &