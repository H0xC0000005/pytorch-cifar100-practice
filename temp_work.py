import torch

# Your class labels
class_labels = torch.tensor([1, 3, 2, 1, 4])

# Number of classes
num_classes = 5

# Convert to one-hot encoding
one_hot_encoding = torch.eye(num_classes)[class_labels]

print(one_hot_encoding)
print(torch.eye(num_classes))
