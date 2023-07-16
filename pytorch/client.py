import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import socket
import pickle
import json
from timeit import default_timer as timer

from models.vgg16 import Vgg16Head, VGG16_RESULT_SIZE

images = torch.zeros(1, 3, 224, 224)

model_head = Vgg16Head()

RX_IP = '127.0.0.1'
RX_WORK_PORT = 3787
RX_SYNC_PORT = RX_WORK_PORT + 1
SYNC_REPEAT = 16
TIMESTAMP_SIZE = 21

def synchronize_timestamp_server(sync_sock):
	diff = 0
	for _ in range(SYNC_REPEAT):
		start = timer()
		sync_sock.sendall(pickle.dumps(start, protocol=pickle.HIGHEST_PROTOCOL))
		middle = pickle.loads(sync_sock.recv(TIMESTAMP_SIZE))
		end = timer()
		diff += middle - (end + start) / 2
	
	diff /= SYNC_REPEAT
	print(f"Sync diff: {diff} (client timestamp + sync diff = server timestamp)")
	return diff


def send_workload(ip, port):
	print("[Synchronize]")
	
	sync_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sync_sock.connect((RX_IP, RX_SYNC_PORT))
	sync_diff = synchronize_timestamp_server(sync_sock)
	sync_sock.close()

	print("[Process head model]")

	ts_start = timer() + sync_diff

	outputs1 = model_head(images)

	ts_head_done = timer() + sync_diff

	print(f"Send data: {outputs1.size()}")

	work_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	work_sock.connect((ip, port))

	
	send_x=outputs1.detach().numpy()
	data_input = pickle.dumps(send_x, protocol=pickle.HIGHEST_PROTOCOL)
	work_sock.sendall(data_input)
	print(f"Data sent: {len(data_input)}")

	print("[Receive output]")
	data = b""
	target_length = VGG16_RESULT_SIZE + 2 * TIMESTAMP_SIZE
	while 1:
		output = work_sock.recv(VGG16_RESULT_SIZE)
		if len(data) >= target_length: break
		data += output
	final_output = pickle.loads(data[:VGG16_RESULT_SIZE])
	ts_end = timer() + sync_diff
	print(f"Data received: {len(data)}")
	ts_server_receive = pickle.loads(data[VGG16_RESULT_SIZE:VGG16_RESULT_SIZE+TIMESTAMP_SIZE])
	ts_server_tail_done = pickle.loads(data[VGG16_RESULT_SIZE+TIMESTAMP_SIZE:])
	work_sock.close()

	print(f"Output received: {len(final_output[0])}")
	print(f"Elapsed time: {ts_end - ts_start}")
	print(f"  - Head inference:    {ts_head_done - ts_start}")
	print(f"  - Send intermediate: {ts_server_receive - ts_head_done}")
	print(f"  - Tail inference:    {ts_server_tail_done - ts_server_receive}")
	print(f"  - Send output:       {ts_end - ts_server_tail_done}")



send_workload(RX_IP, RX_WORK_PORT)