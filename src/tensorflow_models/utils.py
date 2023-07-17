import tensorflow as tf

def configHeadTail(head, tail):
	head_model = head()
	tail_model = tail()
	print("[Head]")
	for layer in head_model.layers:
		print(layer)

	print("")
	print("[Tail]")
	for layer in tail_model.layers:
		print(layer)

numpy2tensor = tf.convert_to_tensor

def tensor2numpy(tensor):
	return tensor.numpy()

def tensorShape(tensor):
	return tf.shape(tensor)