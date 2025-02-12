# ct
python -m experiments.make_lm_outputs --query_peft_dir <YOUR_MODEL_DIRECTORY> --retrieval_method <RETREIVAL METHOD> --batch_size 16 --max_new_tokens 50 --model_name "Meta-Llama-3.1-8B-Instruct" --inference True --c_type "ct" --dataset "test"
# ct-probe
python -m experiments.make_lm_outputs --query_peft_dir <YOUR_MODEL_DIRECTORY> --retrieval_method <RETREIVAL METHOD> --batch_size 16 --max_new_tokens 50 --model_name "Meta-Llama-3.1-8B-Instruct" --inference True --c_type "ct-probe" --dataset "test"
# number
python -m experiments.make_lm_outputs --query_peft_dir <YOUR_MODEL_DIRECTORY> --retrieval_method <RETREIVAL METHOD> --batch_size 16 --max_new_tokens 50 --model_name "Meta-Llama-3.1-8B-Instruct" --inference True --c_type "number" --dataset "test"
# ling
python -m experiments.make_lm_outputs --query_peft_dir <YOUR_MODEL_DIRECTORY> --retrieval_method <RETREIVAL METHOD> --batch_size 16 --max_new_tokens 50 --model_name "Meta-Llama-3.1-8B-Instruct" --inference True --c_type "ling" --dataset "test"