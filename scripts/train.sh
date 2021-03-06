python ./run_biqe.py \
--model_type kg \
--model_name_or_path vanilla \
--config_name bert-base-cased \
--tokenizer_name ./data/fb15k-237/DAGs/indices/kg_index.pkl \
--data_set biqe \
--do_train \
--do_predict \
--train_file ./data/fb15k-237/DAGs/train_dags.json \
--predict_file ./data/fb15k-237/DAGs/dev_dags.json \
--num_train_epochs 50.0 \
--warmup_proportion 0.1 \
--output_dir ./output/ \
--seed 42 \
--valid_gold ./data/fb15k-237/DAGs/dev_dags_filters.json \
--valid_every_epoch \
--plus_classify_tokens 1 \
--predict one_step_greedy \
--token_index_path ./data/fb15k-237/DAGs/indices/ent_index.pkl \
--train_batch_size 256 \
--predict_batch_size 256 \
--gradient_accumulation_steps 1 \
--learning_rate 3e-5 \
--max_part_a 30 \
--max_seq_length 30