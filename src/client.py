import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import tensorflow as tf
import numpy as np

import socket
import pickle
import json
import argparse
from timeit import default_timer as timer

from utils import decodeImageNet


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--framework", dest="framework", action="store")
parser.add_argument("-i", "--input", dest="input", action="store")
parser.add_argument("--ip", dest="ip", action="store")
parser.add_argument("-p", "--port", dest="port", action="store")
parser.add_argument("-m", "--model", dest="model", action="store")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False)
args = parser.parse_args()

# Init: framework
if args.framework == 'pytorch' or args.framework == 'pt':
	FRAMEWORK = 'pytorch'
elif args.framework == 'tensorflow' or args.framework == 'tf':
	FRAMEWORK = 'tensorflow'
else:
	# default
	FRAMEWORK = 'pytorch'

# Init: model & config
if FRAMEWORK == 'pytorch':
	if args.model == 'vgg16':
		from pytorch_models.vgg16 import Vgg16Head, VGG16_RESULT_SIZE
		model_head = Vgg16Head()
		RESULT_SIZE = VGG16_RESULT_SIZE
	elif args.model == 'resnet50':
		from pytorch_models.resnet50 import ResNet50Head, RESNET50_RESULT_SIZE
		model_head = ResNet50Head()
		RESULT_SIZE = RESNET50_RESULT_SIZE
	elif args.model == 'alexnet':
		from pytorch_models.alexnet import AlexNetHead, ALEXNET_RESULT_SIZE
		model_head = AlexNetHead()
		RESULT_SIZE = ALEXNET_RESULT_SIZE
	else:
		# default
		args.model = 'vgg16'
		from pytorch_models.vgg16 import Vgg16Head, VGG16_RESULT_SIZE
		model_head = Vgg16Head()
		RESULT_SIZE = VGG16_RESULT_SIZE
	
	# utils
	from pytorch_models.utils import tensor2numpy, numpy2tensor, tensorShape
	
	# input
	import PIL
	import torchvision.transforms as transforms

	image_path = args.input if args.input else 'test1.jpg'
	image = PIL.Image.open(image_path).resize((224, 224))
	image = transforms.ToTensor()(image)
	image = torch.reshape(image, (1, 3, 224, 224))


elif FRAMEWORK == 'tensorflow':
	if args.model == 'vgg16':
		from tensorflow_models.vgg16 import Vgg16Head, VGG16_RESULT_SIZE
		model_head = Vgg16Head()
		RESULT_SIZE = VGG16_RESULT_SIZE
	elif args.model == 'resnet50':
		from tensorflow_models.resnet50 import ResNet50Head, RESNET50_RESULT_SIZE, inputPreprocessor, outputDecoder
		model_head = ResNet50Head()
		RESULT_SIZE = RESNET50_RESULT_SIZE
	elif args.model == 'alexnet':
		from tensorflow_models.alexnet import AlexNetHead, ALEXNET_RESULT_SIZE
		model_head = AlexNetHead()
		RESULT_SIZE = ALEXNET_RESULT_SIZE
	else:
		# default
		from tensorflow_models.vgg16 import Vgg16Head, VGG16_RESULT_SIZE
		model_head = Vgg16Head()
		RESULT_SIZE = VGG16_RESULT_SIZE
	
	# utils
	from tensorflow_models.utils import tensor2numpy, numpy2tensor, tensorShape
	from tensorflow.keras.preprocessing.image import load_img, img_to_array

	# input
	image_path = args.input if args.input else 'test1.jpg'
	image = load_img(image_path, target_size=(224, 224))
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = inputPreprocessor(image)

# Init: connection
RX_IP = args.ip if args.ip else '127.0.0.1'
RX_WORK_PORT = int(args.port) if args.port else 3786
RX_SYNC_PORT = RX_WORK_PORT + 1

# Init: etc
SYNC_REPEAT = 16
TIMESTAMP_SIZE = 21
VERBOSE = args.verbose

def cprint(string):
	if VERBOSE: print(string)

cprint(f"")
cprint(f"==============================")
cprint(f"RUNNING")
cprint(f"  - Framework: {FRAMEWORK}")
cprint(f"  - Model: {args.model}")
cprint(f"  - RX ip: {RX_IP}")
cprint(f"  - RX work port: {RX_WORK_PORT}")
cprint(f"  - RX sync port: {RX_SYNC_PORT}")
cprint(f"==============================")
cprint(f"")

## FUNCTIONS ##

def synchronize_timestamp_server(sync_sock):
	diff = 0
	for _ in range(SYNC_REPEAT):
		start = timer()
		sync_sock.sendall(pickle.dumps(start, protocol=pickle.HIGHEST_PROTOCOL))
		middle = pickle.loads(sync_sock.recv(TIMESTAMP_SIZE))
		end = timer()
		diff += middle - (end + start) / 2
	
	diff /= SYNC_REPEAT
	cprint(f"Sync diff: {diff} (client timestamp + sync diff = server timestamp)")
	return diff


def send_workload(ip, port):
	cprint("[Synchronize]")
	
	sync_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sync_sock.connect((RX_IP, RX_SYNC_PORT))
	sync_diff = synchronize_timestamp_server(sync_sock)
	sync_sock.close()

	cprint("[Process head model]")

	ts_start = timer() + sync_diff

	outputs1 = model_head(image)

	ts_head_done = timer() + sync_diff

	cprint(f"Send data: {tensorShape(outputs1)}")

	work_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	work_sock.connect((ip, port))

	send_x = tensor2numpy(outputs1)
	
	data_input = pickle.dumps(send_x, protocol=pickle.HIGHEST_PROTOCOL)
	work_sock.sendall(data_input)
	cprint(f"Data sent: {len(data_input)}")

	cprint("[Receive output]")
	data = b""
	target_length = RESULT_SIZE + 2 * TIMESTAMP_SIZE
	while 1:
		output = work_sock.recv(RESULT_SIZE)
		if len(data) >= target_length: break
		data += output
	final_output = pickle.loads(data[:RESULT_SIZE])
	ts_end = timer() + sync_diff
	cprint(f"Data received: {len(data)}")
	ts_server_receive = pickle.loads(data[RESULT_SIZE:RESULT_SIZE+TIMESTAMP_SIZE])
	ts_server_tail_done = pickle.loads(data[RESULT_SIZE+TIMESTAMP_SIZE:])
	work_sock.close()

	cprint(f"Output received")
	if VERBOSE: decodeImageNet(numpy2tensor(final_output).flatten())
	cprint(f"Elapsed time: {ts_end - ts_start}")
	cprint(f"  - Head inference:    {ts_head_done - ts_start}")
	cprint(f"  - Send intermediate: {ts_server_receive - ts_head_done}")
	cprint(f"  - Tail inference:    {ts_server_tail_done - ts_server_receive}")
	cprint(f"  - Send output:       {ts_end - ts_server_tail_done}")



send_workload(RX_IP, RX_WORK_PORT)