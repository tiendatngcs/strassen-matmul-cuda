/*
module load intel CUDA
compile
nvcc -ccbin=icc -o matmul.exe matmul.cu
*/
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* srand, rand */
#include <assert.h>     /* assert */
#include <time.h>       /* time */
#include <string.h>
#include <iostream>

#define MAXINT 2048
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

enum QUARTER {
    m11,
    m12,
    m21,
    m22
};

class HMatrix {
public:
    int xDim, yDim;
    int* h_mat;
    std::string name;

    HMatrix(int xDim, int yDim, std::string name, bool zero=true) :
            xDim(xDim), 
            yDim(yDim), 
            name(name),
            h_mat((int*)malloc(xDim*yDim*sizeof(int))) {
        if (zero) {
            // memset(h_mat, 0, xDim*yDim*sizeof(int));
        } else {
            for (int i = 0; i < xDim; i++) {
                for (int j = 0; j < yDim; j++) {
                    h_mat[i*xDim+j] = rand() % MAXINT;
                }
            }
        }
    }

    ~HMatrix() {
        assert(h_mat != NULL);
        free(h_mat);
    }

    void print() {
        printf("HMatrix %s:\n", name.c_str());
        for (int i = 0; i < xDim; i++) {
            for (int j = 0; j < yDim; j++) {
                printf("\t%d", h_mat[i*xDim+j]);
            }
            printf("\n");
        }
    }
    
    int* elem(int x, int y) {
        return &h_mat[x*xDim+y]; 
    }

    int size() { return xDim*yDim*sizeof(int); }
};

class DMatrix {
public:
    int xDim, yDim;
    int* d_mat;
    std::string name;

    DMatrix(HMatrix* h_mat) :
            xDim(h_mat->xDim), 
            yDim(h_mat->yDim), 
            name(h_mat->name) {
        cudaMalloc(&d_mat, h_mat->size());
        cudaMemcpy(d_mat, h_mat->h_mat, h_mat->size(), cudaMemcpyHostToDevice);
    }

    void free() {
        assert(d_mat != NULL);
        cudaFree(d_mat);
    }

    CUDA_CALLABLE_MEMBER ~DMatrix() {
        assert(d_mat != NULL);
        cudaFree(d_mat);
    }
    
    CUDA_CALLABLE_MEMBER void print() {
        printf("DMatrix %s:\n", name.c_str());
        for (int i = 0; i < xDim; i++) {
            for (int j = 0; j < yDim; j++) {
                printf("\t%d", d_mat[i*xDim+j]);
            }
            printf("\n");
        }
    }
    
    CUDA_CALLABLE_MEMBER int* elem(int x, int y) {
        return &d_mat[x*xDim+y]; 
    }
};

__global__ void matAdd(int* A, int* B, int* C, int dim, QUARTER q) {
    int x_offset = 0;
    int y_offset = 0;
    switch (q) {
        case m11:
            // do nothing
            break;
        case m12:
            y_offset = dim/2;
            break;
        case m21:
            x_offset = dim/2;
            break;
        case m22:
            x_offset = dim/2;
            y_offset = dim/2;
            break;
        default:
            assert(false);
    };
    printf("x %d, y%d \n", threadIdx.x, threadIdx.y);
    C[threadIdx.x*dim+threadIdx.y] = A[(threadIdx.x + x_offset)*dim+(threadIdx.y + y_offset)] + B[(threadIdx.x + x_offset)*dim+(threadIdx.y + y_offset)];
    return;
}

__global__ void ADD(int* A, int* B, int* C, int dim) {
    printf("Print %d\n", threadIdx.x);
    C[threadIdx.x*dim+threadIdx.y] = A[threadIdx.x*dim+threadIdx.y] + B[threadIdx.x*dim+threadIdx.y];
    return;
}

__global__ void test() {
    printf("Print %d\n", threadIdx.x);
    printf("Cp 1\n");
    return;
}

void Strassen(DMatrix** A_arr, DMatrix** B_arr, DMatrix** C_arr, int k) {
    DMatrix* A = A_arr[k-1];
    DMatrix* B = B_arr[k-1];
    DMatrix* C = C_arr[k-1];
    assert(A->d_mat != NULL);

    dim3 grid(A->xDim/2, B->yDim/2);
    matAdd<<<1, grid>>>(A->d_mat, B->d_mat, C->d_mat, A->xDim, m11);
    // test<<<2, 2>>>();

    // M1

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
    printf("Matmul of size %d x %d\n", n, n);

    // HMatrix** A = (HMatrix**)malloc((k + 1) * sizeof(HMatrix*));
    // HMatrix** B = (HMatrix**)malloc((k + 1) * sizeof(HMatrix*));
    // HMatrix** C = (HMatrix**)malloc((k + 1) * sizeof(HMatrix*));
    // for (int i = 1; i <= k; i ++) {
    //     int n = 1 << (k-i);
    //     bool zero = (i == k) ? false : true;
    //     printf("n %d\n", n);
    //     A[i] = new Matrix(n, n, "A"+std::to_string(i), zero);
    //     B[i] = new Matrix(n, n, "B"+std::to_string(i), zero);
    //     C[i] = new Matrix(n, n, "C"+std::to_string(i), zero);
    // }


    // HMatrix* hA = new HMatrix(n, n, "A", false);
    // HMatrix* hB = new HMatrix(n, n, "B", false);
    // HMatrix* hC = new HMatrix(n, n, "C", true);

    // hA->print();
    // hB->print();


    // DMatrix **dA = (DMatrix**)malloc(k * sizeof(DMatrix*));
    // DMatrix **dB = (DMatrix**)malloc(k * sizeof(DMatrix*));
    // DMatrix **dC = (DMatrix**)malloc(k * sizeof(DMatrix*));

    // for (int i = 0; i < k; i ++) {
    //     dA[i] = new DMatrix(hA);
    //     dB[i] = new DMatrix(hB);
    //     dC[i] = new DMatrix(hC);
    // }
    // dim3 grid(hA->xDim/2, hB->yDim/2);
    // ADD<<<1,1>>>(dA[0]->d_mat, dB[0]->d_mat, dC[0]->d_mat, hA->xDim);
    test<<<3, 3>>>();


    // Strassen(dA, dB, dC, k);
    // cudaMemcpy(hC, dC[0], hC->size(), cudaMemcpyDeviceToHost);
    // cudaMemcpy(hA, dA[0], hA->size(), cudaMemcpyDeviceToHost);
    // cudaMemcpy(hB, dB[0], hB->size(), cudaMemcpyDeviceToHost);
    // hC->print();
    // hB->print();
    // hA->print();

    // for (int i = 0; i < k; i ++) {
    //     dA[i]->free();
    //     dB[i]->free();
    //     dC[i]->free();
    // }
    // test<<<2, 2>>>();


    // Matrix* M = new Matrix(n, n, "M2", true);

    // subAdd<<<1, 2>>>(A[0], B[0]);

    
}