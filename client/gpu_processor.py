import numpy as np
import pyopencl as cl

# Código do kernel OpenCL — roda na GPU
# Aplica uma transformação linear (matriz 4x4) a pontos 3D
kernel_code = """
__kernel void transform(
    __global const float *points,   // Vetor de entrada com coordenadas (x, y, z)
    __global const float *T,        // Matriz 4x4 (apenas 3 linhas) achatada
    __global float *result,         // Vetor de saída com pontos transformados
    const int num_points)           // Número total de pontos
{
    int i = get_global_id(0);       // Índice global do ponto a ser processado

    if (i < num_points) {           // Garante que o índice não ultrapasse o limite
        // Extrai as coordenadas x, y, z do ponto i
        float x = points[3*i + 0];
        float y = points[3*i + 1];
        float z = points[3*i + 2];

        // Aplica a transformação linear (T * ponto + translação)
        result[3*i + 0] = T[0]*x + T[1]*y + T[2]*z + T[3];   // X'
        result[3*i + 1] = T[4]*x + T[5]*y + T[6]*z + T[7];   // Y'
        result[3*i + 2] = T[8]*x + T[9]*y + T[10]*z + T[11]; // Z'
    }
}
"""

def process_points(points, T):
    # Garante que os pontos e a matriz estejam no tipo float32 (requerido pelo OpenCL)
    points = points.astype(np.float32)

    # Achata a matriz 4x4 de transformação (usamos só as 3 primeiras linhas)
    T_flat = T[:3, :].flatten().astype(np.float32)

    # Cria um array vazio com a mesma forma dos pontos para armazenar o resultado
    result = np.empty_like(points)

    # Cria o contexto OpenCL (seleciona automaticamente a GPU ou CPU compatível)
    ctx = cl.create_some_context()

    # Cria a fila de comandos para enviar e receber dados da GPU
    queue = cl.CommandQueue(ctx)

    # Atalhos para as flags de alocação de memória (host/device)
    mf = cl.mem_flags

    # Envia os dados para a memória da GPU
    d_points = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=points)
    d_T = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T_flat)
    d_result = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

    # Compila o código do kernel na GPU
    program = cl.Program(ctx, kernel_code).build()

    # Obtém o kernel e define os argumentos
    kernel = program.transform
    kernel.set_args(d_points, d_T, d_result, np.int32(len(points)))

    # Enfileira o kernel para execução com uma thread por ponto
    cl.enqueue_nd_range_kernel(queue, kernel, (len(points),), None)

    # Copia os dados processados da GPU de volta para o array result no host
    cl.enqueue_copy(queue, result, d_result)

    # Aguarda até que todas as tarefas terminem
    queue.finish()

    # Retorna o array com os pontos transformados
    return result
