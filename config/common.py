import os

root_path = os.path.expanduser('~')

# path
repo_dir = os.path.join(root_path, 'deeplearning_pytorch_template')
data_dir = os.path.join(repo_dir, 'data')
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')
