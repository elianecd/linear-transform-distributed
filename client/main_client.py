import socket
import pickle
import numpy as np
import time
from client.gpu_processor import process_points  # ou gpu_processor_opencl se estiver usando OpenCL
from server.config import HOST, PORT

# Função para receber todos os dados do servidor
def recv_all(sock):
    data = b""
    try:
        while True:
            part = sock.recv(4096)
            if not part:
                break
            data += part
            if len(part) < 4096:
                break
    except ConnectionResetError:
        print("[X] Conexão encerrada pelo servidor.")
        return None
    return data

# Cria socket TCP e conecta ao servidor
s = socket.socket()
print(f"\nConectando ao servidor em {HOST}:{PORT}...\n")
s.connect((HOST, PORT))
print("Conectado com sucesso.\n")

# Começa a contagem total de tempo do cliente
global_start = time.time()

# Recebe a matriz de transformação (T)
msg = recv_all(s)
if msg is None:
    print("Falha ao receber matriz de transformação.")
    exit(1)

T = pickle.loads(msg)["T"]
print("Matriz de transformação recebida:")
print(T, "\n")

# Loop principal de solicitação e processamento de blocos
while True:
    print("Solicitando novo bloco ao servidor...\n")
    s.sendall(pickle.dumps({"ask_block": True}))

    raw = recv_all(s)
    if raw is None:
        print("Não foi possível receber o bloco. Encerrando cliente.\n")
        break

    resp = pickle.loads(raw)

    if resp["status"] == 0:
        print("Processamento finalizado! Nenhum bloco restante.\n")
        break

    # Extrai dados do pacote recebido
    points = resp["points"]
    start = resp["start"]
    block_id = resp["block_id"]

    print(f"Bloco {block_id} recebido (pontos {start} a {start + len(points) - 1})\n")

    # Processamento com GPU
    print(f"Processando bloco {block_id} com GPU...\n")
    start_time = time.time()
    result = process_points(np.array(points), T)
    end_time = time.time()
    print(f"Bloco {block_id} processado em {end_time - start_time:.3f} segundos.\n")

    # Envia os dados processados de volta ao servidor
    s.sendall(pickle.dumps({
        "block_done": block_id,
        "start": start,
        "data": result
    }))

    print(f"Bloco {block_id} enviado ao servidor.\n" + "-" * 60 + "\n")

# Tempo total
print("Todos os blocos foram processados.\n")
global_end = time.time()
print(f"Tempo total do cliente: {global_end - global_start:.2f} segundos\n")

s.close()
