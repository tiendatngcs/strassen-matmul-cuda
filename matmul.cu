/*
module load intel CUDA
compile
nvcc -ccbin=icc -o matmul.exe matmul.cu
*/
#include <stdio.h>      /* printf */
#include <assert.h>     /* assert */



typedef struct {
    int k; // current dim
    int org_k; // current dim
    int x_start;
    int y_start;
    float* elements;
} Matrix;

typedef struct {
    Matrix M1;
    Matrix M2;
    Matrix M3;
    Matrix M4;
    Matrix M5;
    Matrix M6;
    Matrix M7;
} Mset;

enum QUARTER {
    m11,
    m12,
    m21,
    m22
};

// __device__ static QUARTER get_zero() { return m22; }
__device__ static QUARTER get_quarter(int k) {
    printf("x %d, y %d, n %d, n/2 %d (threadIdxy % (1 << (k))) %d\n", threadIdx.x, threadIdx.y, 1 << k, 1 << (k-1), (threadIdx.y % (1 << (k))));

    int kx = threadIdx.x / (1 << (k-1));
    int ky = threadIdx.y / (1 << (k-1));
    printf("kx %d, ky %d\n", kx, ky);
    return (QUARTER)(kx*2 + ky);
}

__device__ static float* first_pointer(Matrix M) {
    int n = 1 << M.k;
    int x = M.x_start;
    int y = M.y_start; 
    return &(M.elements[x * n + y]);
}

__device__ static Matrix new_device_matrix(int k) {
    Matrix ret;
    int n = 1 << k;
    ret.k = k;
    ret.org_k = k;
    ret.x_start = 0;
    ret.y_start = 0;
    // ret.elements = (float*)malloc(n*sizeof(float));
    // cudaMalloc(&ret.elements, n*sizeof(float));
    ret.elements = (float*)malloc(n*sizeof(float));
    return ret;
}

// static Matrix new_device_matrix(int k) {
//     Matrix ret;
//     int n = 1 << k;
//     ret.k = k;
//     ret.org_k = k;
//     ret.x_start = 0;
//     ret.y_start = 0;
//     // ret.elements = (float*)malloc(n*sizeof(float));
//     cudaMalloc(&ret.elements, n*sizeof(float));
//     // ret.elements = (float*)malloc(n*sizeof(float));
//     return ret;
// }

__device__ static void free_device_matrix(Matrix M) {
    free(M.elements);
}

// static void free_device_matrix(Matrix M) {
//     cudaFree(M.elements);
// }

// // Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    printf("m11, 1, 2, 2:\n");
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    C[i] = A[i] + B[i];
}

// Matrix get_sub(Matrix A, QUARTER q) {
//     A.k = A.k -1;
//     if (q == m21 || q == m22) A.x_start=A.x_start + A.k;
//     if (q == m12 || q == m22) A.y_start=A.y_start + A.k;
//     return A;
// }

// __global__ void get_M1(Matrix A, Matrix B, Matrix M1) {
//     // test this 
    // assert(M1.k == A.k -1);
    // assert(B.k == A.k);
//     int n = 1 << A.k;
//     // int x = (x_start + threadIdx.x);
//     // int y = (y_start + threadIdx.y)
//     Matrix A11 = get_sub(A, m11);
//     Matrix A22 = get_sub(A, m22);
//     Matrix B11 = get_sub(B, m11);
//     Matrix B22 = get_sub(B, m22);
//     if (M1.k == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
//        first_pointer(M1)[0] = (first_pointer(A11)[0] + first_pointer(A22)[0]) * (first_pointer(B11)[0] + first_pointer(B22)[0]);
//         return;
//     }
//     // allocate M1a and M1b to store itermediate;
//     Matrix M1a = new_device_matrix(M1.k);
//     Matrix M1b = new_device_matrix(M1.k);
//     MatAdd(A11, A22, M1a);
//     MatAdd(B11, B22, M1b);
//     MatMull(M1a, M1b, M1);
//     free_device_matrix(M1a);
//     free_device_matrix(M1b);
// }

__global__ void MatMul(Matrix A, Matrix B, Matrix C) 
{   
    assert(A.k == B.k);
    assert(A.k == C.k);
    int n = 1 << A.k;
    if (A.k == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        int idx = threadIdx.x * n + threadIdx.y;
        first_pointer(C)[0] = first_pointer(A)[0] * first_pointer(B)[0];
        // C.elements[idx] = A.ele
        return;
    }
    // allow all threads to go through allocate and calculate M1 to M7
    
}


__global__ void testfunc(int k)
{
    bool* ret;
    // cudaMalloc(&ret, 4*sizeof(bool));
    ret = (bool*)malloc(4*sizeof(bool));
    int n = 1 << k;
    // is_of_quarter(m11, threadIdx.x, threadIdx.y, k, ret);
    // printf("%d, %d, %d: %d\n", threadIdx.x, threadIdx.y, k, ret[threadIdx.x * (1<<k) + threadIdx.y]);
    // printf("m11, 1, 2, 2:\n");
    // is_of_quarter(m11, )
    // cudaFree(ret);
    free(ret);
    // printf("zero %d\n", get_quarter(k));

    // Matrix A = new_device_matrix(k);
    // float* A;
    // // A = (float*)malloc(n * n * sizeof(float));
    // cudaMalloc(&A, n * n * sizeof(float));
    // if (threadIdx.x == 0) {
    //     A[0] = 1.0;
    // // printf("Thread x %d, A[0] %d\n", threadIdx.x, A.elements[0]);
    // }
    // printf("Thread x %d, A[0] %f\n", threadIdx.x, A[0]);
    // // free_device_matrix(A);
    // // free(A);
    // cudaFree(A);


}

static Matrix new_device_matrix_from_host(int k, float* from_M) {
    Matrix ret;
    int n = 1 << k;
    int bytes = n*n*sizeof(float);
    ret.k = k;
    ret.org_k = k;
    ret.x_start = 0;
    ret.y_start = 0;
    // ret.elements = (float*)malloc(n*sizeof(float));
    cudaMalloc(&ret.elements, bytes);
    cudaMemcpy(ret.elements, from_M, bytes, cudaMemcpyHostToDevice);
    // ret.elements = (float*)malloc(n*sizeof(float));
    return ret;
}

static void free_device_matrix_from_host(Matrix M) {
    cudaFree(M.elements);
}

void* strassenMatMul(int**)


// Host code
int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: %s <k>\n", argv[0]);
    }

    int k = atoi(argv[1]);
    int n = 1 << k;
    printf("Matmul of size %d x %d\n", n, n);
    size_t size = n*n;
    size_t bytes = n*n*sizeof(float);
    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    h_A[0] = 23;
    h_B[0] = 34;

    // // Allocate vectors in device memory
    // float* d_A;
    // float* d_B;
    // float* d_C;
    // cudaMalloc(&d_A, bytes);
    // cudaMalloc(&d_B, bytes);
    // cudaMalloc(&d_C, bytes);
    // // // Copy vectors from host memory to device memory
    // cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    Matrix d_A = new_device_matrix_from_host(k, h_A);
    Matrix d_B = new_device_matrix_from_host(k, h_B);
    Matrix d_C = new_device_matrix_from_host(k, h_C);

    // // Invoke kernel
    // int threadsPerBlock = 256;
    // int blocksPerGrid =
    // (size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 dimGrid(n, n);
    // VecAdd<<<1, dimGrid>>>(d_A, d_B, d_C, size);
    MatMul<<<1, dimGrid>>>(d_A, d_B, d_C);
    // printf("m11, 1, 2, 2:\n");
    // Matrix A = new_device_matrix(k);
    // testfunc<<<1, dimGrid>>>(k);
    // // Copy result from device memory to host memory
    // // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C.elements, bytes, cudaMemcpyDeviceToHost);

    printf("C: %f\n", h_C[0]);

    // Free device memory
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);
    free_device_matrix_from_host(d_A);
    free_device_matrix_from_host(d_B);
    free_device_matrix_from_host(d_C);
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
}
