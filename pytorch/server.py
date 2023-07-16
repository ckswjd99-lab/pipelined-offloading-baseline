import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import socket
import pickle
import threading

from models.vgg16 import Vgg16Tail, VGG16_INTER_SIZE
from timeit import default_timer as timer


## GLOBALS ##

model_tail = Vgg16Tail()

RX_IP = '127.0.0.1'
RX_WORK_PORT = 3787
RX_SYNC_PORT = RX_WORK_PORT+1
BUFFER_SIZE = VGG16_INTER_SIZE

SYNC_REPEAT = 16

sem_workload_append = threading.Semaphore(1)
sem_workload_pop = threading.Semaphore(0)

workload_queue = []
client_sock_queue = []
ts_receive_queue = []

num_work = 2
remaining_work = 0


recved = 0
## FUNCTIONS ##

def synchronize_timestamp_server(sync_sock):
	conn, addr = sync_sock.accept()
	print("[CONT_THREAD] Accepted")
	for _ in range(SYNC_REPEAT):
		conn.recv(21)
		timestamp = pickle.dumps(timer(), protocol=pickle.HIGHEST_PROTOCOL)
		conn.sendall(timestamp)
	print("[CONT_THREAD] Done")

def receive_workload(server_sock, workload_queue, client_sock_queue):
	conn, addr = server_sock.accept()
	print(f"[NET_THREAD] receive_workload: accepted")
	data = b''
	while 1:
		tensor = conn.recv(BUFFER_SIZE)
		data += tensor
		if len(data) >= BUFFER_SIZE: break
	print(f"[NET_THREAD] receive_workload: data received ({len(data)})")

	ts_receive = timer()

	input_ten = pickle.loads(data)
	input = torch.from_numpy(input_ten)
	print(f"[NET_THREAD] receive_workload: data converted to {input.size()}")
	
	sem_workload_append.acquire()
	workload_queue.append(input)
	client_sock_queue.append(conn)
	ts_receive_queue.append(ts_receive)
	sem_workload_pop.release()
	print(f"[NET_THREAD] receive_workload: workload pushed")

def process_workload(workload_queue, client_sock_queue):
	sem_workload_pop.acquire()
	input = workload_queue.pop(0)
	client_sock = client_sock_queue.pop(0)
	ts_receive = ts_receive_queue.pop(0)
	sem_workload_append.release()
	print("[WORK_THREAD] Popped workload")

	output = model_tail(input).detach().numpy()
	ts_tail_done = timer()
	print("[WORK_THREAD] Inference done")

	data_final = pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
	client_sock.sendall(data_final)
	print(f"[WORK_THREAD] Returned result ({len(data_final)})")
	client_sock.sendall(pickle.dumps(ts_receive, protocol=pickle.HIGHEST_PROTOCOL))
	client_sock.sendall(pickle.dumps(ts_tail_done, protocol=pickle.HIGHEST_PROTOCOL))
	print(f"[WORK_THREAD] Returned timestamp ({len(pickle.dumps(ts_receive, protocol=pickle.HIGHEST_PROTOCOL))})")
	client_sock.close()

def control_thread(num_work):
	print("[CONT_THREAD] Start...")
	sync_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sync_sock.bind((RX_IP, RX_SYNC_PORT))
	sync_sock.listen(1)
	print(f"[CONT_THREAD] Controller running on {RX_IP}:{RX_SYNC_PORT}")

	remaining_work = num_work
	while (num_work == -1 or remaining_work > 0):
		synchronize_timestamp_server(sync_sock)
		remaining_work -= 1

	sync_sock.close()

def network_thread(num_work):
	print("[NET_THREAD] Start...")
	server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_sock.bind((RX_IP, RX_WORK_PORT))
	server_sock.listen(1)

	print(f"[NET_THREAD] Server listening at {RX_IP}:{RX_WORK_PORT}")

	remaining_work = num_work
	while (num_work == -1 or remaining_work > 0):
		print(f"[NET_THREAD] Remaining work: {remaining_work}")
		receive_workload(server_sock, workload_queue, client_sock_queue)
		remaining_work -= 1
	
	server_sock.close()

def worker_thread(num_work):
	print("[WORK_THREAD] Start...")

	remaining_work = num_work
	while (num_work == -1 or remaining_work > 0):
		print(f"[WORK_THREAD] Remaining work: {remaining_work}")
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