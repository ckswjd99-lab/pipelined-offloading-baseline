def send_python_object(sock, object):
    pickle = pickle.dumps(object, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(pickle)