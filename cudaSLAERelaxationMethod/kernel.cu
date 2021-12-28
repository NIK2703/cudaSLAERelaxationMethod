#include <stdio.h>
#include <math_functions.h>

#define INPUT_FILE_PATH "C:\\Users\\User\\source\\repos\\cudaSLAERelaxationMethod\\test_data.txt"
#define OUTPUT_FILE_PATH "output.txt"

void printMassive(double* mas, int size);
void printMatrix(double** matrix, int size_x, int size_y);
double** squeezeMatrix(double* matrix, int size_x, int size_y);


/*
Функция ядра для решения СЛАУ общем виде
Работает с одномерной сеткой из одного двумерного блока с рекомндумыми размерами:
    Высота: количество уравнений n; Ширина: количество неизвестных n
Принимает матрицу коэффициентов A,
          массив свободных членов B,
          порядок матрицы коэффциентов n,
          точность eps,
          массив ответов X (изначально хранит начальное приближение, после выполнения ядра - решение задачи)
          ссылку на область памяти для хранения приведённой матрицы коэффициентов P,
          ссылку на область памяти для хранения приведённого матрицы-столбца свободных членов C,

*/
__global__ void
yakobiKernel(double* A, double* B, int n, double eps,
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
    int tid = tx + ty * bdx;

    //вычисление приведённой матрицы коэффициентов
    for (int ptrx = tx; ptrx < n; ptrx += bdx) {
        for (int ptry = ty; ptry < n; ptry += bdy) {
            P[ptrx + ptry * n] = -A[ptrx + ptry * n] / A[ptry + ptry * n];
        }
    }

    //глобальный индекс потока
    
    //вычисление приведённого матрицы-столбца, инициализация текущего ответа начальным значением
    for (int i = tid; i < n; i += tnum) {
        C[i] = B[i] / A[i + i * n];
    }

    //приведение матрицы должно быть полностью завершено, прежде потоки перейдут 
    //  к вычислению невязок
    __syncthreads();

    //условие перехода к следующей итерации
    __shared__ bool nextIter;
    nextIter = true;

    int xTermNum = n; //количество слагаемых при вычислении одного приближения
    int totalXTermNum = xTermNum * n; //общее количество слагаемых при вычислении приближений

    while (nextIter) {
        if (tid == 0) {
            nextIter = false;
        }

        //массив для частичных сумм приближений
        extern __shared__ double sumArray[]; //(n+1) * n

        for (int i = tid; i < totalXTermNum; i += tnum) {
            int xIndex = i / xTermNum; //номер приближения, с которой работает поток
            int termIndex = i % xTermNum; //номер слагаемого в приближении, с которым работает поток

            // Запись слагаемых приближений в массив частичных сумм
            sumArray[i] = termIndex == 0 ? C[xIndex] :
                termIndex - 1 < xIndex ? P[xIndex * n + (termIndex - 1)] * X[termIndex - 1] :
                P[xIndex * n + termIndex] * X[termIndex];

        }

        //перед тем, как прейти к суммированию, необходимо, чтобы все слагаемые были записаны в массив
        __syncthreads();

        //суммирование слагаемых приближений методом редукции
		for (int sumRange = xTermNum; sumRange > 1;  sumRange = (sumRange + 1) / 2) {
            /*if (tid == 0) {
                printf("%d \n", sumRange);
            }*/
			for (int i = tid, int sumElLimit = sumRange / 2; i < sumElLimit * n; i += tnum) {

				int xIndex = i / sumElLimit; //номер невязки, с которой работает поток
				int termIndex = i % sumElLimit; //номер слагаемого в невязке, с которым работает поток
                int xFirstTermIndex = xIndex * xTermNum;

				sumArray[xFirstTermIndex + termIndex] +=
					sumArray[xFirstTermIndex + (sumRange - termIndex - 1)];

				
			}
		}

        //перед тем, как записать новые приближения, необходимо дождаться
        //  их вычисления методом редукции
        __syncthreads();

        for (int i = tid; i < n; i += tnum) {
            //если разница между предыдущим и новым x не удовлетворяет заданной точночти,
            //  выполнить ещё итерaцию
            if (abs(sumArray[xTermNum * i] - X[i]) > eps) {
                nextIter = true;
            }
            //запись новых приближений
            X[i] = sumArray[xTermNum * i];
        }

    }
}

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
Решение СЛАУ в общем виде методом Якоби на GPU
*/
double* yakobiMethod(double** A, double* B, int n, double* initX, double eps) {
    double* ADev;
    double* BDev;
    double* XDev;
    double* PDev;
    double* CDev;

    double* stretchedA = stretchMatrix(A, n, n);

    cudaMalloc(&ADev, n * n * sizeof(double));
    cudaMalloc(&BDev, n * sizeof(double));
    cudaMalloc(&XDev, n * sizeof(double));
    cudaMalloc(&PDev, n * n * sizeof(double));
    cudaMalloc(&CDev, n * sizeof(double));

    cudaMemcpy(ADev, stretchedA, n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(BDev, B, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(XDev, initX, n * sizeof(double), cudaMemcpyHostToDevice);

    yakobiKernel <<<1, dim3(n, n)>>>(ADev, BDev, n, eps, XDev, PDev, CDev);

    double* X = new double[n];

    cudaMemcpy(X, XDev, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(ADev);
    cudaFree(BDev);
    cudaFree(XDev);
    cudaFree(PDev);
    cudaFree(CDev);

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

    double* X = new double[n];
    for (int i = 0; i < n; i++) {
        X[i] = 0;
    }
    double eps = 0.01;

    X = yakobiMethod(A, B, n, X, eps);

    printMassive(X, n);

    return 0;
}