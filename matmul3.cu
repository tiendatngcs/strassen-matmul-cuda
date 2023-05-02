#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <assert.h>

#define MAXINT 2048
enum QUARTER {
    m11,
    m12,
    m21,
    m22
};

int **dM1, **dM2, **dM3, **dM4, **dM5, **dM6, **dM7;
int *tmp;

__global__
void saxpy(int n, float a, float *x, float *y)
{
    printf("sadfas\n");
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

__device__
static void get_sub_offset(int curr_dim, QUARTER q, int& x_offset, int& y_offset) {
    int new_dim = curr_dim/2;
    switch (q) {
        case m11:
            // do nothing
            break;
        case m12:
            y_offset = new_dim;
            break;
        case m21:
            x_offset = new_dim;
            break;
        case m22:
            x_offset = new_dim;
            y_offset = new_dim;
            break;
        default:
            assert(false);
    }
}

__global__
void matAdd(int* A, int* B, int* C, int org_dim, QUARTER qA, QUARTER qB, QUARTER qC, int k, bool is_subtract=false)
{   
    int curr_dim = 1 << k;
    int new_dim = curr_dim /2;
    int Ax_offset = 0;
    int Ay_offset = 0;
    int Bx_offset = 0;
    int By_offset = 0;
    int Cx_offset = 0;
    int Cy_offset = 0;

    get_sub_offset(curr_dim, qA, Ax_offset, Ay_offset);
    get_sub_offset(curr_dim, qB, Bx_offset, By_offset);
    get_sub_offset(curr_dim, qC, Cx_offset, Cy_offset);

    if (is_subtract) {
        C[(Cx_offset+threadIdx.x)*org_dim + Cy_offset + threadIdx.y] = A[(Ax_offset+threadIdx.x)*org_dim + Ay_offset + threadIdx.y] - B[(Bx_offset+threadIdx.x)*org_dim + By_offset + threadIdx.y];
        return;
    }
    C[(Cx_offset+threadIdx.x)*org_dim + Cy_offset + threadIdx.y] = A[(Ax_offset+threadIdx.x)*org_dim + Ay_offset + threadIdx.y] + B[(Bx_offset+threadIdx.x)*org_dim + By_offset + threadIdx.y];
}

__global__
void matCopy(int* fromM, int* toM, int org_dim, QUARTER fromQ, QUARTER toQ, int k) {
    int curr_dim = 1 << k;
    int new_dim = curr_dim /2;
    int fromx_offset = 0;
    int fromy_offset = 0;
    int tox_offset = 0;
    int toy_offset = 0;

    get_sub_offset(curr_dim, fromQ, fromx_offset, fromy_offset);
    get_sub_offset(curr_dim, toQ, tox_offset, toy_offset);
    toM[(tox_offset+threadIdx.x)*org_dim + toy_offset + threadIdx.y] = fromM[(fromx_offset+threadIdx.x)*org_dim + fromy_offset + threadIdx.y];
}

static void normalMatMul(int*A, int*B, int*C, int dim) {
    for (int i = 0; i < dim; i ++) {
        for (int j = 0; j < dim; j++) {
            // for the current row of A and col of B
            C[i*dim + j] = 0;
            for (int k = 0; k < dim; k ++) {
                C[i*dim + j] += A[i*dim+k] * B[k*dim + j];
            }
        }
    }
}

__global__
void matMul(int* A, int* B, int* C, int curr_dim) {
    assert(curr_dim == 1);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        C[0] = A[0] * B[0];
    }
}

static void printMat(int* mat, int xDim, int yDim, char* name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < xDim; i++) {

        for(int j = 0; j < yDim; j++) {
            printf("\t%d", mat[i*xDim+j]);
        }
        printf("\n");
    }
}

static void printDevMat(int* mat, int xDim, int yDim, char* name) {
    cudaMemcpy(tmp, mat, xDim*yDim*sizeof(int), cudaMemcpyDeviceToHost);
    printMat(tmp, xDim, yDim, name);
}

void Strassen(int** dA, int** dB, int** dC, int org_dim, int k) {
    int curr_dim = 1 << k;
    int new_dim = org_dim /2;
    dim3 grid(new_dim, new_dim);
    // matAdd<<<1, grid>>>(dA[k], dB[k], dC[k], org_dim, m11, m11);

    if (k == 0) {
        assert(curr_dim == 1);
        matMul<<<1, 1>>>(dA[k], dB[k], dC[k], curr_dim);
        return;
    }

    // M1v
    matAdd<<<1, grid>>>(dA[k], dA[k], dA[k-1], org_dim, m11, m22, m11, k);
    matAdd<<<1, grid>>>(dB[k], dB[k], dB[k-1], org_dim, m11, m22, m11, k);
    Strassen(dA, dB, dM1, org_dim, k-1);

    // M2v
    matAdd<<<1, grid>>>(dA[k], dA[k], dA[k-1], org_dim, m21, m22, m11, k);
    matCopy<<<1, grid>>>(dB[k], dB[k-1], org_dim, m11, m11, k);
    Strassen(dA, dB, dM2, org_dim, k-1);

    // M3v
    matCopy<<<1, grid>>>(dA[k], dA[k-1], org_dim, m11, m11, k);
    matAdd<<<1, grid>>>(dB[k], dB[k], dB[k-1], org_dim, m12, m22, m11, k, true);
    Strassen(dA, dB, dM3, org_dim, k-1);

    // M4v
    matCopy<<<1, grid>>>(dA[k], dA[k-1], org_dim, m22, m11, k);
    matAdd<<<1, grid>>>(dB[k], dB[k], dB[k-1], org_dim, m21, m11, m11, k, true);
    Strassen(dA, dB, dM4, org_dim, k-1);

    // M5v
    matAdd<<<1, grid>>>(dA[k], dA[k], dA[k-1], org_dim, m11, m12, m11, k);
    matCopy<<<1, grid>>>(dB[k], dB[k-1], org_dim, m22, m11, k);
    Strassen(dA, dB, dM5, org_dim, k-1);

    // M6v
    matAdd<<<1, grid>>>(dA[k], dA[k], dA[k-1], org_dim, m21, m11, m11, k, true);
    matAdd<<<1, grid>>>(dB[k], dB[k], dB[k-1], org_dim, m11, m12, m11, k);
    Strassen(dA, dB, dM6, org_dim, k-1);

    // M7
    matAdd<<<1, grid>>>(dA[k], dA[k], dA[k-1], org_dim, m12, m22, m11, k, true);
    matAdd<<<1, grid>>>(dB[k], dB[k], dB[k-1], org_dim, m21, m22, m11, k);
    Strassen(dA, dB, dM7, org_dim, k-1);


    // C11
    matAdd<<<1, grid>>>(dM1[k-1], dM4[k-1], dC[k], org_dim, m11, m11, m11, k);
    matAdd<<<1, grid>>>(dC[k], dM5[k-1], dC[k], org_dim, m11, m11, m11, k, true);
    matAdd<<<1, grid>>>(dC[k], dM7[k-1], dC[k], org_dim, m11, m11, m11, k);

    // C12
    matAdd<<<1, grid>>>(dM3[k-1], dM5[k-1], dC[k], org_dim, m11, m11, m12, k);

    // C21
    matAdd<<<1, grid>>>(dM2[k-1], dM4[k-1], dC[k], org_dim, m11, m11, m21, k);

    // C22
    matAdd<<<1, grid>>>(dM1[k-1], dM2[k-1], dC[k], org_dim, m11, m11, m22, k, true);
    matAdd<<<1, grid>>>(dC[k], dM3[k-1], dC[k], org_dim, m22, m11, m22, k);
    matAdd<<<1, grid>>>(dC[k], dM6[k-1], dC[k], org_dim, m22, m11, m22, k);

}


int main(int argc, char** argv)
{
    srand (time(NULL));
    if (argc != 2) {
        printf("Usage: %s <k>\n", argv[0]);
        exit(0);
    }

    int k = atoi(argv[1]);
    int n = 1 << k;
    int size = n*n;
    int bytes = size*sizeof(int);
    int *hA, *hB, *hC, *testC;
    int **dA, **dB, **dC;
    hA = (int*)malloc(bytes);
    hB = (int*)malloc(bytes);
    hC = (int*)malloc(bytes);
    testC = (int*)malloc(bytes);
    tmp = (int*)malloc(bytes);

    // init host matrices

    for (int i = 0; i < size; i++) {
        hA[i] = rand() % MAXINT;
        hB[i] = rand() % MAXINT;
        hC[i] = 0;
        testC[i] = 0;
        tmp[i] = 0;
    }

    // alloc device matrices

    dA = (int**)malloc((k+1)*sizeof(int*));
    dB = (int**)malloc((k+1)*sizeof(int*));
    dC = (int**)malloc((k+1)*sizeof(int*));
    for (int i = 0; i < k+1; i++) {
        cudaMalloc(&dA[i], bytes);
        cudaMalloc(&dB[i], bytes);
        cudaMalloc(&dC[i], bytes);
    }

    cudaMemcpy(dA[k], hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB[k], hB, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC[k], hC, bytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < k; i++) {
        cudaMemcpy(dA[i], hC, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dB[i], hC, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dC[i], hC, bytes, cudaMemcpyHostToDevice);
    }

    // alloc temp matrices
    dM1 = (int**)malloc((k+1)*sizeof(int*));
    dM2 = (int**)malloc((k+1)*sizeof(int*));
    dM3 = (int**)malloc((k+1)*sizeof(int*));
    dM4 = (int**)malloc((k+1)*sizeof(int*));
    dM5 = (int**)malloc((k+1)*sizeof(int*));
    dM6 = (int**)malloc((k+1)*sizeof(int*));
    dM7 = (int**)malloc((k+1)*sizeof(int*));
    for (int i = 0; i < k+1; i++) {
        cudaMalloc(&dM1[i], bytes);
        cudaMalloc(&dM2[i], bytes);
        cudaMalloc(&dM3[i], bytes);
        cudaMalloc(&dM4[i], bytes);
        cudaMalloc(&dM5[i], bytes);
        cudaMalloc(&dM6[i], bytes);
        cudaMalloc(&dM7[i], bytes);

        cudaMemcpy(dM1[i], hC, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dM2[i], hC, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dM3[i], hC, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dM4[i], hC, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dM5[i], hC, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dM6[i], hC, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dM7[i], hC, bytes, cudaMemcpyHostToDevice);
    }

    printMat(hA, n, n, "A");
    printMat(hB, n, n, "B");
    printMat(hC, n, n, "init C");
    
    Strassen(dA, dB, dC, n, k);
    // int new_dim = n /2;
    // dim3 grid(new_dim, new_dim);
    // matAdd<<<1, grid>>>(dA[k], dB[k], dC[k], n, m11, m12, m12, k, true);
    // matCopy<<<1, grid>>>(dC[k], dA[k], n, m12, m21, k);

    cudaMemcpy(hC, dC[k], bytes, cudaMemcpyDeviceToHost);
    printMat(hC, n, n, "result C");

    normalMatMul(hA, hB, testC, n);
    printMat(testC, n, n, "test C");

    int err_count = 0;
    for (int i = 0; i < size; i++) {
        if (hC[i] != testC[i]) err_count ++;
    }
    if (err_count == 0) {
        printf("Congrats, matmul calculated withut errors!\n");
    } else {
        printf("Houston, we have a problem! Err_count = %d\n", err_count);
    }

    for (int i = 0; i < k+1; i++) {
        cudaFree(dA[i]);
        cudaFree(dB[i]);
        cudaFree(dC[i]);

        cudaFree(dM1[i]);
        cudaFree(dM2[i]);
        cudaFree(dM3[i]);
        cudaFree(dM4[i]);
        cudaFree(dM5[i]);
        cudaFree(dM6[i]);
        cudaFree(dM7[i]);
    }

    free(dM1);
    free(dM2);
    free(dM3);
    free(dM4);
    free(dM5);
    free(dM6);
    free(dM7);
    free(dA);
    free(dB);
    free(dC);
    free(hA);
    free(hB);
    free(hC);
    free(tmp);
    free(testC);
}