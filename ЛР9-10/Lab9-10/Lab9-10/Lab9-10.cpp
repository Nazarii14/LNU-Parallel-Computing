#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <chrono>
#include <algorithm>
#include "matrix.h"
#include "matrix.cpp"

using namespace std;

void fill(Matrix <int>& mtrx);

template <typename T>
void print(Matrix <T>& mtrx);

template <typename T>
void multiplySeq(Matrix <T>& mtrxA, Matrix <T>& mtrxB, Matrix <T>& mtrxRes);

template <typename T>
int equalMatrices(Matrix <T>& mtrxA, Matrix <T>& mtrxB);

#define ROW_START_TAG 0
#define ROW_END_TAG 1
#define A_ROWS_TAG 2
#define C_ROWS_TAG 3
#define LOCAL_TIME_TAG 4


int nProcesses;
MPI_Status status;
MPI_Request request;
size_t rowStart, rowEnd;
size_t granularity;


double start_time, end_time;
double localTimeSaver;

int main(int argc, char* argv[]) {
    srand(time(NULL));

    int rank;

    if (argv[1] == NULL)
        return 1;

    int N = atoi(argv[1]);

    int rowsNumA = N;
    int colsNumA = N;
    int rowsNumB = N;
    int colsNumB = N;

    Matrix <int> A = Matrix <int>(rowsNumA, colsNumA);
    Matrix <int> B = Matrix <int>(rowsNumB, colsNumB);
    Matrix <int> seqC = Matrix <int>(rowsNumA, colsNumB);
    Matrix <int> mpiC = Matrix <int>(rowsNumA, colsNumB);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

    if (rank == 0) {
        cout << "Dimensions: " << N << "x" << N << '\n';
        fill(A);
        fill(B);

        start_time = MPI_Wtime();
        for (int i = 1; i < nProcesses; i++) {
            granularity = (rowsNumA / (nProcesses - 1));
            rowStart = (i - 1) * granularity;

            if (((i + 1) == nProcesses) && ((rowsNumA % (nProcesses - 1)) != 0)) {
                rowEnd = rowsNumA;
            }
            else {
                rowEnd = rowStart + granularity;
            }

            MPI_Isend(&rowStart, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &request);
            MPI_Isend(&rowEnd, 1, MPI_INT, i, ROW_START_TAG, MPI_COMM_WORLD, &request);
            MPI_Isend(&A(rowStart, 0), (rowEnd - rowStart) * colsNumA, MPI_INT, i, A_ROWS_TAG, MPI_COMM_WORLD, &request);
        }
    }
    MPI_Bcast(&B(0, 0), rowsNumB * colsNumB, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank > 0) {
        MPI_Recv(&rowStart, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&rowEnd, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&A(rowStart, 0), (rowEnd - rowStart) * colsNumA, MPI_INT, 0, A_ROWS_TAG, MPI_COMM_WORLD, &status);

        localTimeSaver = MPI_Wtime();

        for (int i = rowStart; i < rowEnd; i++) {
            for (int j = 0; j < B.getCols(); j++) {
                for (int k = 0; k < B.getRows(); k++)
                    mpiC(i, j) += (A(i, k) * B(k, j));
            }
        }
        localTimeSaver = MPI_Wtime() - localTimeSaver;

        MPI_Isend(&rowStart, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&rowEnd, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&mpiC(rowStart, 0), (rowEnd - rowStart) * colsNumB, MPI_INT, 0, C_ROWS_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&localTimeSaver, 1, MPI_INT, 0, LOCAL_TIME_TAG, MPI_COMM_WORLD, &request);
    }

    if (rank == 0) {
        for (int i = 1; i < nProcesses; i++) {
            MPI_Recv(&rowStart, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&rowEnd, 1, MPI_INT, i, ROW_START_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&mpiC(rowStart, 0), (rowEnd - rowStart) * colsNumB, MPI_INT, i, C_ROWS_TAG, MPI_COMM_WORLD, &status);
        }
        end_time = MPI_Wtime();
        double totalMultiplicationTime = end_time - start_time;

        vector<double> LocalMultiplicationTimes = vector<double>(nProcesses);

        for (int i = 1; i < nProcesses; i++) {
            MPI_Recv(&LocalMultiplicationTimes[i], 1, MPI_INT, i, LOCAL_TIME_TAG, MPI_COMM_WORLD, &status);
        }
        double maxLocalMultiplicationTime = *std::max_element(LocalMultiplicationTimes.begin(), LocalMultiplicationTimes.end());

        cout << "MPI time =  " << totalMultiplicationTime << endl;
        cout << "Longest thread time =  " << maxLocalMultiplicationTime << endl;

        auto seqStart = chrono::high_resolution_clock::now();
        multiplySeq(A, B, seqC);
        auto seqEnd = chrono::high_resolution_clock::now();
        chrono::duration<double> seqDuration = seqEnd - seqStart;
        cout << "Seq: " << seqDuration.count() << " seconds\n";

        if (N <= 10)
        {
            cout << "Matrix A:\n";
            print(A);
            cout << "Matrix B:\n";
            print(B);
            cout << "Matrix C(mpi):\n";
            print(mpiC);
            cout << "Matrix C(seq):\n";
            print(seqC);
        }

        bool matricesEqual = equalMatrices(mpiC, seqC) == 0;
        cout << "Equality: " << (matricesEqual ? "True" : "False") << '\n';
    }
    MPI_Finalize();

    return 0;
}

void fill(Matrix <int>& mtrx)
{
    for (size_t i = 0; i < mtrx.getRows(); i++)
    {
        for (size_t j = 0; j < mtrx.getCols(); j++)
            mtrx(i, j) = rand() % 100;
    }
}

template <typename T>
void print(Matrix <T>& mtrx)
{
    for (int i = 0; i < mtrx.getRows(); i++) {
        for (int j = 0; j < mtrx.getCols(); j++)
            cout << mtrx(i, j) << " ";
        cout << endl;
    }
}

template <typename T>
void multiplySeq(Matrix <T>& first, Matrix <T>& second, Matrix <T>& result)
{
    int rowsA = first.getRows();
    int colsA = first.getCols();
    int colsB = second.getCols();

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                result(i, j) += first(i, k) * second(k, j);
            }
        }
    }
}

template <typename T>
int equalMatrices(Matrix <T>& first, Matrix <T>& second)
{
    if (first.getCols() != second.getCols() || first.getRows() != second.getRows())
        return INT_MAX;
    
    int res = 0;
    for (size_t i = 0; i < first.getRows(); i++) {
        for (size_t j = 0; j < first.getCols(); j++)
            res += abs(first(i, j) - second(i, j));
    }
    return res;
}
