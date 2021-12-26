#include <stdio.h>

#define INPUT_FILE_PATH "C:\\Users\\User\\source\\repos\\cudaSLAERelaxationMethod\\test_data.txt"
#define OUTPUT_FILE_PATH "output.txt"

void printMassive(double* mas, int size);
void printMatrix(double** matrix, int size_x, int size_y);

/*
Функция ядра для приведения исходных матриц коэффициентов и свободных членов
    задачи СЛАУ с ленточной структурой данных к требуемому в методе реалксации виду
Работает с одномерной сеткой из одного двумерного блока с рекомндумыми размерами:
    Высота: количество уравнений n; Ширина: ширина ленты
Принимает матрицу коэффициентов A,
          матрица-столбец свободных членов B,
          порядок матрицы коэффициентов n,
          матрицу преобразованных коэффициентов P,
          матрицу преобразованных свободных членов C
*/
__global__ void 
relaxationMatrixReductionKernel(double** A, double* B, int n, double** P, double* C) {
    // идентификтаоры блока и потока
    /*int bx = blockIdx.x;
    int by = blockIdx.y;*/
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    //размеры блока по x и по y
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    //число потоков в блоке
    int tnum = bdx * bdy;

    //вычисление приведённой матрицы коэффициентов
    for (int ptrx = tx; ptrx < n; ptrx += bdx) {
        for (int ptry = ty; ptry < n; ptry += bdy) {
            P[ptrx][ptry] = -A[ptrx][ptry] / A[ptrx][ptrx];
        }
    }

    //вычисление приведённой матрицы-столбца
    for (int tind = tx + ty * bdx; tind < n; tind += tnum) {
        C[tind] = B[tind] / A[tind][tind];
    }
}

/*
Функция ядра для решения СЛАУ с ленточной сруктурой матрицы
Работает с одномерной сеткой из одного двумерного блока с рекомндумыми размерами:
    Высота: количество уравнений n; Ширина: ширина ленты
Принимает матрицу коэффициентов A,
          массив свободных членов B,
          массив ответов X, куда записываются найденные значения неизвестных,
          приближения eps.
*/
//__global__ void
//relaxationIterationKernel(double** P, double* C, double* X, double eps)
//{
//    // идентификтаоры блока и потока
//    /*int bx = blockIdx.x;
//    int by = blockIdx.y;*/
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//
//
//
//}

/*

*/
double* relaxationMethod(double** A, double* B, int n ) {
    double** ADev;
    double* BDev;
    //float* nDev;
    double** PDev;
    double* CDev;
    cudaMalloc(&ADev, n * n * sizeof(double));
    cudaMalloc(&BDev, n * sizeof(double));
    //cudaMalloc(&nDev, sizeof(int));
    cudaMalloc(&PDev, n * n * sizeof(double));
    cudaMalloc(&CDev, n * sizeof(double));

    cudaMemcpy(ADev, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(BDev, B, n * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(nDev, n, sizeof(int), cudaMemcpyHostToDevice);
    relaxationMatrixReductionKernel <<<1, dim3(n, n)>>>(ADev, BDev, n, PDev CDev);

    double** P;
    double* C;

    cudaMemcpy(P, PDev, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C, CDev, n * sizeof(double), cudaMemcpyDeviceToHost);

    printMatrix(P, n, n);
    printf("\n");
    printMassive(C, n);

    cudaFree(ADev);
    cudaFree(BDev);
    //cudaFree(nDev);

    
}

double** readMatrix(FILE *input, int size_x, int size_y) {
    double** matrix = new double*[size_x];
    for (int i = 0; i < size_x; i++) {
        matrix[i] = new double[size_y];
        for (int j = 0; j < size_y; j++) {
            fscanf(input, "%lf", &matrix[i][j]);
        }
    }
    return matrix;
}

double* readMassive(FILE* input, int size) {
    return readMatrix(input,  1, size)[0];
}

void printMassive(double* mas, int size) {
    for (int i = 0; i < size; i++) {
        printf("%lf ", mas[i]);
    }
    printf("\n");
}
    
void printMatrix(double** matrix, int size_x, int size_y) {
    for (int i = 0; i < size_x; i++) {
        printMassive(matrix[i], size_y);
    }
}

int main(void)
{

    FILE* input_data;
    if ((input_data = fopen(INPUT_FILE_PATH, "r")) == NULL)
    {
        printf("Input file open error");
        return 0;
    }

    FILE* output_data;
    if ((output_data = fopen(OUTPUT_FILE_PATH, "w")) == NULL)
    {
        printf("Output file open error");
        return 0;
    }

    int n = 0;
    fscanf(input_data, "%d", &n);
    double** A = readMatrix(input_data, n, n);
    double* B = readMassive(input_data, n);

    printMatrix(A, n, n); 
    printf("\n");
    printMassive(B, n);

    return 0;
}