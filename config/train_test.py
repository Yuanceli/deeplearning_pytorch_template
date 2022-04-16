import os
from config import common

file_path = dict(
    model_dir=os.path.join(common.repo_dir, 'checkpoints'),
)

train_utils = dict(
    batch_size=1024,
    num_of_workers=2,
    num_of_epochs=5,
    learning_rate=1e-4,
    gpu_ids = '0, 1',
)

test_utils = dict(
    batch_size=256,
    num_of_workers=2,
)
