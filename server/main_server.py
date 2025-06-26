import socket
import threading
import pickle
import numpy as np
from server.transform_utils import generate_random_points, create_transformation_matrix
from server.config import HOST, PORT, TOTAL_POINTS

# Input do operador
print("Tamanhos de bloco permitidos: 10000, 20000, 50000, 100000")
BLOCK_SIZE = int(input("Digite o tamanho do bloco desejado: "))
while BLOCK_SIZE not in [10000, 20000, 50000, 100000]:
    BLOCK_SIZE = int(input("Valor inválido. Digite novamente: "))

NUM_BLOCKS = TOTAL_POINTS // BLOCK_SIZE  # Calcula o número total de blocos baseado nos pontos e tamanho

data = generate_random_points(TOTAL_POINTS)  # Gera 5 milhões de pontos 3D aleatórios
result = np.zeros_like(data)  # Cria uma matriz zerada para armazenar os resultados processados
processed = [False] * NUM_BLOCKS  # Lista booleana para marcar quais blocos já foram processados

position = 0  # Posição inicial do próximo bloco
position_lock = threading.Lock()  # Trava para proteger a posição entre threads
processed_lock = threading.Lock()  # Trava para proteger a lista de blocos processados

# Parâmetros da transformação
print("Informe os parâmetros da transformação linear:")
sx = float(input("Escala no eixo X: "))
sy = float(input("Escala no eixo Y: "))
sz = float(input("Escala no eixo Z: "))
rx = float(input("Rotação em graus no eixo X: "))
ry = float(input("Rotação em graus no eixo Y: "))
rz = float(input("Rotação em graus no eixo Z: "))
tx = float(input("Translação no eixo X: "))
ty = float(input("Translação no eixo Y: "))
tz = float(input("Translação no eixo Z: "))

scale = (sx, sy, sz)  # Agrupa escala em uma tupla
rotation = (rx, ry, rz)  # Agrupa rotação em uma tupla
translation = (tx, ty, tz)  # Agrupa translação em uma tupla
T = create_transformation_matrix(scale, rotation, translation) # Cria matriz 4x4 com os parâmetros

print("\nMatriz de transformação criada com sucesso.\n")  # Confirma a criação da matriz

# Função auxiliar para receber dados
def recv_all(sock):  # Função para receber dados completos de um socket
    data = b""  # Inicializa buffer
    while True:  # Loop até receber todos os dados
        part = sock.recv(4096)  # Recebe parte dos dados
        if not part:  # Se conexão fechada, encerra
            break
        data += part  # Adiciona os dados ao buffer
        if len(part) < 4096:  # Se parte menor que buffer, fim da mensagem
            break
    return data  # Retorna os dados completos

def all_blocks_processed():  # Verifica se todos os blocos foram processados
    with processed_lock:  # Trava o acesso à lista
        return all(processed)  # Retorna True se todos forem True

def handle_client(conn, addr):  # Função para tratar conexão de cliente
    global position  # Acessa a variável global position
    print(f"\n[+] Cliente conectado: {addr}\n")  # Log de conexão
    conn.sendall(pickle.dumps({"T": T}))  # Envia matriz de transformação

    try:  # Tenta comunicação com cliente
        while True:  # Loop até encerrar cliente
            msg = recv_all(conn)  # Recebe requisição do cliente
            if not msg:  # Se vazia, encerra
                break
            req = pickle.loads(msg)  # Desserializa mensagem

            if req.get("ask_block"):  # Se cliente pediu novo bloco
                with position_lock:  # Trava acesso à posição
                    if position >= TOTAL_POINTS:  # Se acabou os pontos
                        conn.sendall(pickle.dumps({"status": 0}))  # Informa fim
                        print(f"Todos os blocos já foram distribuídos. Cliente {addr} será encerrado.\n")  # Log fim
                        break

                    block_id = position // BLOCK_SIZE  # Calcula ID do bloco
                    points = data[position:position + BLOCK_SIZE]  # Seleciona os pontos

                    print(f"[->] Enviando bloco {block_id} para {addr} ({len(points)} pontos)\n")  # Log de envio

                    conn.sendall(pickle.dumps({  # Envia os pontos
                        "status": 1,
                        "block_id": block_id,
                        "start": position,
                        "points": points
                    }))

                    position += BLOCK_SIZE  # Avança a posição

            elif req.get("block_done") is not None:  # Se cliente devolveu bloco
                start = req["start"]  # Início do bloco
                block_data = req["data"]  # Dados processados
                block_id = req["block_done"]  # ID do bloco

                with processed_lock:  # Trava acesso ao resultado
                    result[start:start + len(block_data)] = block_data  # Salva dados processados
                    processed[block_id] = True  # Marca bloco como processado

                print(f"[✓] Bloco {block_id} processado por {addr}\n")  # Log de sucesso

    except Exception as e:  # Captura erros
        print(f"[!] Erro com o cliente {addr}: {e}\n")  # Log de erro

    finally:  # Sempre executa ao final
        try:
            conn.shutdown(socket.SHUT_RDWR)  # Encerra conexão
        except:
            pass
        conn.close()  # Fecha socket
        print(f"[-] Cliente {addr} desconectado\n")  # Log de desconexão

print("Servidor escutando conexões em paralelo...\n")  # Log inicial

s = socket.socket()  # Cria socket do servidor
s.bind((HOST, PORT))  # Associa IP e porta
s.listen()  # Inicia escuta

threads = []  # Lista para guardar threads

try:
    while not all_blocks_processed():  # Enquanto ainda há blocos
        conn, addr = s.accept()  # Aceita conexão
        thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)  # Cria thread para cliente
        thread.start()  # Inicia thread
        threads.append(thread)  # Guarda thread

    print("\nAguardando finalização das threads...\n")  # Log de espera

    for t in threads:  # Aguarda todas as threads
        t.join(timeout=2)

    print("Todos os blocos foram processados com sucesso.")  # Log final
    print("Salvando resultado final em 'resultado_final.npy'...\n")  # Log de salvamento

    np.save("resultado_final.npy", result)  # Salva resultado

finally:
    s.close()  # Fecha servidor
    print("Servidor encerrado.\n")  # Log de encerramento
