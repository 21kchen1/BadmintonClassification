import socket

def receive_broadcast():
    # 创建一个 UDP 套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定到一个端口
    sock.bind(('', 8005))

    try:
        while True:
            # 接收数据
            data, addr = sock.recvfrom(1024)
            print(f"Received broadcast from {addr}: {data.decode()}")
    finally:
        sock.close()

# 在一个单独的线程中运行接收函数
import threading
thread = threading.Thread(target=receive_broadcast)
thread.start()