import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import socket
import pickle
import threading
import argparse

import tensorflow as tf

from timeit import default_timer as timer


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--framework", dest="framework", action="store")
parser.add_argument("-i", "--input", dest="input", action="store")
parser.add_argument("--ip", dest="ip", action="store")
parser.add_argument("-p", "--port", dest="port", action="store")
parser.add_argument("-m", "--model", dest="model", action="store")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False)
args = parser.parse_args()

images = torch.zeros(1, 3, 224, 224)

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
		from pytorch_models.vgg16 import Vgg16Tail, VGG16_INTER_SIZE
		model_tail = Vgg16Tail()
		BUFFER_SIZE = VGG16_INTER_SIZE
	elif args.model == 'resnet50':
		from pytorch_models.resnet50 import ResNet50Tail, RESNET50_INTER_SIZE
		model_tail = ResNet50Tail()
		BUFFER_SIZE = RESNET50_INTER_SIZE
	elif args.model == 'alexnet':
		from pytorch_models.alexnet import AlexNetTail, ALEXNET_INTER_SIZE
		model_tail = AlexNetTail()
		BUFFER_SIZE = ALEXNET_INTER_SIZE
	else:
		# default
		from pytorch_models.vgg16 import Vgg16Tail, VGG16_INTER_SIZE
		model_tail = Vgg16Tail()
		BUFFER_SIZE = VGG16_INTER_SIZE
	
	# utils
	from pytorch_models.utils import tensor2numpy, numpy2tensor, tensorShape
elif FRAMEWORK == 'tensorflow':
	if args.model == 'vgg16':
		from tensorflow_models.vgg16 import Vgg16Tail, VGG16_INTER_SIZE
		model_tail = Vgg16Tail()
		BUFFER_SIZE = VGG16_INTER_SIZE
	elif args.model == 'resnet50':
		from tensorflow_models.resnet50 import ResNet50Tail, RESNET50_INTER_SIZE
		model_tail = ResNet50Tail()
		BUFFER_SIZE = RESNET50_INTER_SIZE
	elif args.model == 'alexnet':
		from tensorflow_models.alexnet import AlexNetTail, ALEXNET_INTER_SIZE
		model_tail = AlexNetTail()
		BUFFER_SIZE = ALEXNET_INTER_SIZE
	else:
		# default
		from tensorflow_models.vgg16 import Vgg16Tail, VGG16_INTER_SIZE
		model_tail = Vgg16Tail()
		BUFFER_SIZE = VGG16_INTER_SIZE
	
	# utils
	from tensorflow_models.utils import tensor2numpy, numpy2tensor, tensorShape

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

sem_workload_append = threading.Semaphore(1)
sem_workload_pop = threading.Semaphore(0)

workload_queue = []
client_sock_queue = []
ts_receive_queue = []

num_work = 2
remaining_work = 0

recved = 0

cprint(f"")
cprint(f"==============================")
cprint(f"RUNNING")
cprint(f"  - Framework: {FRAMEWORK}")
cprint(f"  - RX ip: {RX_IP}")
cprint(f"  - RX work port: {RX_WORK_PORT}")
cprint(f"  - RX sync port: {RX_SYNC_PORT}")
cprint(f"==============================")
cprint(f"")

## FUNCTIONS ##

def synchronize_timestamp_server(sync_sock):
	conn, addr = sync_sock.accept()
	cprint("[CONT_THREAD] Accepted")
	for _ in range(SYNC_REPEAT):
		conn.recv(21)
		timestamp = pickle.dumps(timer(), protocol=pickle.HIGHEST_PROTOCOL)
		conn.sendall(timestamp)
	cprint("[CONT_THREAD] Done")

def receive_workload(server_sock, workload_queue, client_sock_queue):
	conn, addr = server_sock.accept()
	cprint(f"[NET_THREAD] receive_workload: accepted")
	data = b''
	while 1:
		tensor = conn.recv(BUFFER_SIZE)
		data += tensor
		if len(data) >= BUFFER_SIZE: break
	cprint(f"[NET_THREAD] receive_workload: data received ({len(data)})")

	ts_receive = timer()

	input_ten = pickle.loads(data)
	if FRAMEWORK == 'pytorch':
		input = numpy2tensor(input_ten)
	elif FRAMEWORK == 'tensorflow':
		input = tf.convert_to_tensor(input_ten)
	
	cprint(f"[NET_THREAD] receive_workload: data converted to {tensorShape(input)}")
	
	sem_workload_append.acquire()
	workload_queue.append(input)
	client_sock_queue.append(conn)
	ts_receive_queue.append(ts_receive)
	sem_workload_pop.release()
	cprint(f"[NET_THREAD] receive_workload: workload pushed")

def process_workload(workload_queue, client_sock_queue):
	sem_workload_pop.acquire()
	input = workload_queue.pop(0)
	client_sock = client_sock_queue.pop(0)
	ts_receive = ts_receive_queue.pop(0)
	sem_workload_append.release()
	cprint("[WORK_THREAD] Popped workload")

	output = tensor2numpy(model_tail(input))
	ts_tail_done = timer()
	cprint("[WORK_THREAD] Inference done")

	data_final = pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
	client_sock.sendall(data_final)
	cprint(f"[WORK_THREAD] Returned result ({len(data_final)})")
	client_sock.sendall(pickle.dumps(ts_receive, protocol=pickle.HIGHEST_PROTOCOL))
	client_sock.sendall(pickle.dumps(ts_tail_done, protocol=pickle.HIGHEST_PROTOCOL))
	cprint(f"[WORK_THREAD] Returned timestamp ({len(pickle.dumps(ts_receive, protocol=pickle.HIGHEST_PROTOCOL))})")
	client_sock.close()

def control_thread(num_work):
	cprint("[CONT_THREAD] Start...")
	sync_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sync_sock.bind((RX_IP, RX_SYNC_PORT))
	sync_sock.listen(1)
	cprint(f"[CONT_THREAD] Controller running on {RX_IP}:{RX_SYNC_PORT}")

	remaining_work = num_work
	while (num_work == -1 or remaining_work > 0):
		synchronize_timestamp_server(sync_sock)
		remaining_work -= 1

	sync_sock.close()

def network_thread(num_work):
	cprint("[NET_THREAD] Start...")
	server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_sock.bind((RX_IP, RX_WORK_PORT))
	server_sock.listen(1)

	cprint(f"[NET_THREAD] Server listening at {RX_IP}:{RX_WORK_PORT}")

	remaining_work = num_work
	while (num_work == -1 or remaining_work > 0):
		cprint(f"[NET_THREAD] Remaining work: {remaining_work}")
		receive_workload(server_sock, workload_queue, client_sock_queue)
		remaining_work -= 1
	
	server_sock.close()

def worker_thread(num_work):
	cprint("[WORK_THREAD] Start...")

	remaining_work = num_work
	while (num_work == -1 or remaining_work > 0):
		cprint(f"[WORK_THREAD] Remaining work: {remaining_work}")
		process_workload(workload_queue, client_sock_queue)
		remaining_work -= 1

def main():
	# Ready
	st = threading.Thread(target=control_thread, args=(num_work,))
	nt = threading.Thread(target=network_thread, args=(num_work,))
	wt = threading.Thread(target=worker_thread, args=(num_work,))

	# Start
	st.start()
	nt.start()
	wt.start()


## RUN ##
main()