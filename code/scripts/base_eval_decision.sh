# ct
python -m experiments.make_decision_vllm --batch_size 16 --max_new_tokens 50 --inference True --c_type "ct" --seed 0 --data_dir <YOUR_FILE_DIRECTORY> --model_name <MODEL_NAME>
# ct-probe
python -m experiments.make_decision_vllm --batch_size 16 --max_new_tokens 50 --inference True --c_type "ct_probe" --seed 0 --data_dir <YOUR_FILE_DIRECTORY> --model_name <MODEL_NAME>
# ling
python -m experiments.make_decision_vllm --batch_size 16 --max_new_tokens 50 --inference True --c_type "ling" --seed 0 --data_dir <YOUR_FILE_DIRECTORY> --model_name <MODEL_NAME>
# number
python -m experiments.make_decision_vllm --batch_size 16 --max_new_tokens 50 --inference True --c_type "number" --seed 0 --data_dir <YOUR_FILE_DIRECTORY> --model_name <MODEL_NAME>