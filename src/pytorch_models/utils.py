import torch
import torch.nn as nn

def modelSplitter(original_model, split_idx):
	original_layer = []
	for layer in original_model.children():
		original_layer.append(layer)
	
	head_sequential = nn.Sequential(*original_layer[:split_idx])
	tail_sequential = nn.Sequential(*original_layer[split_idx:])
	
	return head_sequential, tail_sequential

def configHeadTail(head, tail):
	print("[Head]")
	child_counter = 0
	for child in head().children():
		print(child_counter, ":", child)
		child_counter += 1

	print("")
	print("[Tail]")
	child_counter = 0
	for child in tail().children():
		print(child_counter, ":", child)
		child_counter += 1

numpy2tensor = torch.from_numpy

def tensor2numpy(tensor):
	return tensor.detach().numpy()

def tensorShape(tensor):
	return tensor.size()
