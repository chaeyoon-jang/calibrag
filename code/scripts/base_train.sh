# ct-probe
python -m experiments.train.train_classifier_tune --model_name="Meta-Llama-3.1-8B-Instruct" --batch_size 2 --gradient_accumulation_steps 1 --max_steps 5000 --lr 1e-04 --seed 0 
# ct 
python -m experiments.train.train_calibration_tune --model_name="Meta-Llama-3.1-8B-Instruct" --batch_size 2 --gradient_accumulation_steps 1 --c_type ct --max_steps 5000 --lr 1e-04 --seed 0
# number 
python -m experiments.train.train_calibration_tune --c_type "number" --model_name "Meta-Llama-3.1-8B-Instruct" --batch_size 2 --gradient_accumulation_steps 1 --max_steps 5000 --lr 1e-04 --seed 0
# limg
python -m experiments.train.train_calibration_tune --c_type "ling" --model_name "Meta-Llama-3.1-8B-Instruct" --batch_size 2 --gradient_accumulation_steps 1 --max_steps 5000 --lr 1e-04 --seed 0