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
RX_PORT = 3787
BUFFER_SIZE = VGG16_RESULT_SIZE

def send_workload(ip, port):
	data = []

	outputs1 = model_head(images)
	print(outputs1.size())

	print("Sending Data to server")
	start=timer()


	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((ip, port))
	send_x=outputs1.detach().numpy()
	data_input = pickle.dumps(send_x, protocol=pickle.HIGHEST_PROTOCOL)
	print(f"data size: {len(data_input)}")
	s.sendall(data_input)
	print("data sent to server")

	while 1:
		output = s.recv(BUFFER_SIZE)
		if not output: break
		data.append(output)

	final_output=pickle.loads(b"".join(data))
	s.close()
	end=timer()
	print(f"output: {len(final_output[0])}")
	print('Output shown on mobile side')
	print("final time =")
	print (end - start)

send_workload(RX_IP, RX_PORT)