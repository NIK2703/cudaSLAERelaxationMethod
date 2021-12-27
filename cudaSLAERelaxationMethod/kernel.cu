#include <stdio.h>
#include <math_functions.h>

//#define N 4
#define INPUT_FILE_PATH "C:\\Users\\User\\source\\repos\\cudaSLAERelaxationMethod\\test_data.txt"
#define OUTPUT_FILE_PATH "output.txt"

void printMassive(double* mas, int size);
void printMatrix(double** matrix, int size_x, int size_y);
double** squeezeMatrix(double* matrix, int size_x, int size_y);

//const int N = 4;

// Выделение константной памяти на GPU под коэффициенты и свободные члены для ускорения доступа
//__constant__ double** constP[N][N];
//__constant__ double* constC[N];

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
//__global__ void
//relaxationMatrixReductionKernel(double* A, double* B, int n, double* P, double* C) {
//    // идентификтаоры блока и потока
//    /*int bx = blockIdx.x;
//    int by = blockIdx.y;*/
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//    //размеры блока по x и по y
//    int bdx = blockDim.x;
//    int bdy = blockDim.y;
//    //число потоков в блоке
//    int tnum = bdx * bdy;
//
//    //вычисление приведённой матрицы коэффициентов
//   for (int ptrx = tx; ptrx < n; ptrx += bdx) {
//        for (int ptry = ty; ptry < n; ptry += bdy) {
//            P[ptrx + ptry * n] = -A[ptrx + ptry * n] / A[ptrx + ptrx * n];
//        }
//    }
//
//    //вычисление приведённой матрицы-столбца
//    for (int tind = tx + ty * bdx; tind < n; tind += tnum) {
//        C[tind] = B[tind] / A[tind + tind * n];
//    }
//}

/*
Функция ядра для решения СЛАУ с ленточной сруктурой матрицы
Работает с одномерной сеткой из одного двумерного блока с рекомндумыми размерами:
    Высота: количество уравнений n; Ширина: ширина ленты
Принимает матрицу коэффициентов A,
          массив свободных членов B,
          начальное приближение initX,
          порядок матрицы коэффциентов n,
          точность eps,
          массив ответов X, куда записываются найденные значения неизвестных.
*/
__global__ void
relaxationKernel(double* A, double* B, int n, double eps,
    double* X, double* P, double* C)
{
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
            P[ptrx + ptry * n] = -A[ptrx + ptry * n] / A[ptrx * n + ptrx];
        }
    }

    //глобальный индекс потока
    int tid = tx + ty * bdx;
    //вычисление приведённого матрицы-столбца, инициализация текущего ответа начальным значением
    for (int i = tid; i < n; i += tnum) {
        C[i] = B[i] / A[i + i * n];
    }

    //исходный массив коэффициентов
   if (tid == 0) {
        printf("\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%lf ", A[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    //приведённый массив коэффициентов
    if (tid == 0) {
        printf("\n");
        for (int i = 0; i < n; i ++) {
            for (int j = 0; j < n; j++) {
                printf("%lf ", P[i * n + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    //приведение матрицы должно быть полностью завершено, прежде потоки перейдут 
    //  к вычислению невязок
    __syncthreads();

    //условие перехода к следующей итерации
    __shared__ bool nextIter;
    nextIter = true;

    int discrepTermNum = n + 1; //количесвто слагаемых при вычислении одной невязки
    int totalDiscrepTermNum = discrepTermNum * n; //общее количество слагаемых при вычислении невязок

    while (nextIter) {
        if (tid == 0) {
            nextIter = false;
        }

        //массив для частичных сумм невязок 
        extern __shared__ double sumArray[]; //(n+1) * n

        for (int i = tid; i < totalDiscrepTermNum; i += tnum) {
            int discrepIndex = i / discrepTermNum; //номер невязки, с которой работает поток
            int termIndex = i % discrepTermNum; //номер слагаемого в невязке, с которым работает поток

            // Запись слагаемых невязок в массив частичных сумм
            sumArray[i] = termIndex == 0 ? C[discrepIndex] :
                termIndex == 1 ? -X[discrepIndex] :
                termIndex - 2 < discrepIndex ? P[discrepIndex * n + (termIndex - 2)] * X[termIndex - 2] :
                P[discrepIndex * n + (termIndex - 1)] * X[termIndex - 2];

        }

        //перед тем, как прейти к суммированию, необходимо, чтобы все слагаемые были записаны в массив
        __syncthreads();

        //массив слагаемых невязок
        /*if (tid == 0) {
            for (int i = 0; i < (n + 1) * n; i++) {
                printf("%lf ", sumArray[i]);
                if ((i + 1) % (n + 1) == 0) {
                    printf("\n");
                }
            }
        }*/

        //суммирование слагаемых невязок методом редукции
		for (int sumRange = discrepTermNum; sumRange > 1;  sumRange = (sumRange + 1) / 2) {
            /*if (tid == 0) {
                printf("%d \n", sumRange);
            }*/
			for (int i = tid, int sumElLimit = sumRange / 2; i < sumElLimit * n; i += tnum) {

				int discrepIndex = i / sumElLimit; //номер невязки, с которой работает поток
				int termIndex = i % sumElLimit; //номер слагаемого в невязке, с которым работает поток
                int discrepFirstIndex = discrepIndex * discrepTermNum;

				sumArray[discrepFirstIndex + termIndex] +=
					sumArray[discrepFirstIndex + (sumRange - termIndex - 1)];

				/*if (tid == 0) {
					printf("\n");
					for (int i = 0; i < (n + 1) * n; i++) {
						printf("%lf ", sumArray[i]);
						if ((i + 1) % (n + 1) == 0) {
							printf("\n");
						}
					}
					printf("\n");
				}*/
			}
		}

        //перед тем, как прибавить полученные невязки к ответам, необходимо дождаться
        //  их вычисления методом редукции
        __syncthreads();

        for (int i = tid; i < n; i += tnum) {
            if (abs(sumArray[discrepTermNum * i]) > eps) {
                nextIter = true;
            }
            X[i] += sumArray[discrepTermNum * i];
            printf("%lf ", X[i]);
        }
        if (tid == 0) {
            printf("\n");
        }

        /*if (tid == 0) {
            printf("\n");
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    printf("%lf ", A[i * n + j]);
                }
                printf("\n");
            }
            printf("\n");
        }*/

        //массив слагаемых невязок
       /* if (tid == 0) {
            printf("\n");
            for (int i = 0; i < (n + 1) * n; i++) {
                printf("%lf ", sumArray[i]);
                if ((i + 1) % (n + 1) == 0) {
                    printf("\n");
                }
            }
            printf("\n");
        }*/



    }
}
//double* stretchMatrix(double** matrix, int size_x, int size_y) {
//    double* stretchedMatrix = new double[size_x * size_y];
//    for (int i = 0; i < size_x; i++) {
//        for (int j = 0; j < size_y; j++) {
//            stretchedMatrix[i + j * size_x] = matrix[i][j];
//        }
//    }
//    return stretchedMatrix;
//}
//
//double** squeezeMatrix(double* matrix, int size_x, int size_y) {
//    double** squeezedMatrix = new double*[size_x];
//    for (int i = 0; i < size_x; i++) {
//        squeezedMatrix[i] = new double[size_y];
//        for (int j = 0; j < size_y; j++) {
//            squeezedMatrix[i][j] = matrix[i + j * size_x] ;
//        }
//    }
//    return squeezedMatrix;
//}

double* stretchMatrix(double** matrix, int size_x, int size_y) {
    double* stretchedMatrix = new double[size_x * size_y];
    for (int i = 0; i < size_x; i++) {
        for (int j = 0; j < size_y; j++) {
            stretchedMatrix[i * size_y + j] = matrix[i][j];
        }
    }
    return stretchedMatrix;
}

double** squeezeMatrix(double* matrix, int size_x, int size_y) {
    double** squeezedMatrix = new double* [size_x];
    for (int i = 0; i < size_x; i++) {
        squeezedMatrix[i] = new double[size_y];
        for (int j = 0; j < size_y; j++) {
            squeezedMatrix[i][j] = matrix[i * size_y + j];
        }
    }
    return squeezedMatrix;
}

/*

*/
double* relaxationMethod(double** A, double* B, int n, double* initX, double eps) {
    double* ADev;
    double* BDev;
    //int* nDev;
    //double* epsDev;
    double* XDev;
    double* PDev;
    double* CDev;

    double* stretchedA = stretchMatrix(A, n, n);

    cudaMalloc(&ADev, n * n * sizeof(double));
    cudaMalloc(&BDev, n * sizeof(double));
    //cudaMalloc(&nDev, sizeof(int));
    //cudaMalloc(&epsDev, sizeof(double));
    cudaMalloc(&XDev, n * sizeof(double));
    cudaMalloc(&PDev, n * n * sizeof(double));
    cudaMalloc(&CDev, n * sizeof(double));

    cudaMemcpy(ADev, stretchedA, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(BDev, B, n * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(nDev, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(XDev, initX, n * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(epsDev, &eps, sizeof(double), cudaMemcpyHostToDevice);

    relaxationKernel <<<1, dim3(n, n)>>>(ADev, BDev, n, eps, /*nDev, epsDev,*/ XDev, PDev, CDev);

    double* X = new double[n];

    cudaMemcpy(X, XDev, n * sizeof(double), cudaMemcpyDeviceToHost);

    printMassive(X, n);

    cudaFree(ADev);
    cudaFree(BDev);
    //cudaFree(nDev);
    //cudaFree(epsDev);
    cudaFree(XDev);

    return X;
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

    double* initX = new double[n];
    for (int i = 0; i < n; i++) {
        initX[i] = 0;
    }
    double eps = 0.01;

    relaxationMethod(A, B, n, initX, eps);

    /*printMatrix(A, n, n);
    printf("\n");

    double* stretchedA = stretchMatrix(A, n, n);
    printMassive(stretchedA, n * n);
    printf("\n");

    double** squeezedA = squeezeMatrix(stretchedA, n, n);
    printMatrix(squeezedA, n, n);*/

    

   /* printMatrix(A, n, n); 
    printf("\n");
    printMassive(B, n);*/

    return 0;
}