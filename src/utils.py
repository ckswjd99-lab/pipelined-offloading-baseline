import torch

def decodeImageNet(probabilities):
	with open("imagenet_classes.txt", "r") as f:
		categories = [s.strip() for s in f.readlines()]
	# Show top categories per image
	top5_prob, top5_catid = torch.topk(probabilities, 5)
	for i in range(top5_prob.size(0)):
		print(f"  {i+1}: {categories[top5_catid[i]], top5_prob[i].item()}")