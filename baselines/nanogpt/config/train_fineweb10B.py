out_dir = "out-fineweb10B"

dataset = "fineweb10B"

block_size = 1024
n_layer = 6
n_head = 6
n_embd = 288
dropout = 0.0
bias = False

batch_size = 32
gradient_accumulation_steps = 4
max_iters = 5000
eval_interval = 500
log_interval = 10
eval_iters = 100

learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

dtype = "bfloat16"
