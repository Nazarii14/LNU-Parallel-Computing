#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>

using namespace std;
using namespace chrono;


class Matrix 
{
public:
    std::vector<std::vector<int>> matrix;
    int width;
    int height;

    Matrix(int width, int height)
    {
        this->width = width;
        this->height = height;
        this->matrix = std::vector<std::vector<int>>(this->width, std::vector<int>(this->height, 0));
    }
    void generate()
    {
        for (int i = 0; i < this->width; i++)
        {
            for (int j = 0; j < this->height; j++)
                this->matrix[i][j] = rand() % 100;
        }
    }
    void print()
    {
        for (int i = 0; i < this->width; i++)
        {
            for (int j = 0; j < this->height; j++)
                std::cout << this->matrix[i][j] << "\t";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    Matrix operator+(const Matrix& other)
    {
        if (this->width != other.width || this->height != other.height)
        {
            throw std::invalid_argument("Matrices must have the same dimensions for addition.");
        }

        Matrix result(this->width, this->height);
        for (int i = 0; i < this->width; i++)
        {
            for (int j = 0; j < this->height; j++)
                result.matrix[i][j] = this->matrix[i][j] + other.matrix[i][j];
        }
        return result;
    }

    void addMatricesInArea(
        std::vector<std::vector<int>>& other,
        std::vector<std::vector<int>>& result, 
        int startRow, 
        int endRow, 
        int startCol, 
        int endCol) 
    {
        for (int i = startRow; i < endRow; i++) {
            for (int j = startCol; j < endCol; j++) {
                result[i][j] = this->matrix[i][j] + other[i][j];
            }
        }
    }

    std::vector<std::vector<int>> addMatricesParallel(std::vector<std::vector<int>>& other, int threadNum) {
        int numRows = this->matrix.size();
        int numCols = this->matrix[0].size();

        int rowsPerThread = numRows / threadNum;
        int remainingRows = numRows % threadNum;

        std::vector<std::thread> threads;
        std::vector<std::vector<int>> result(numRows, std::vector<int>(numCols, 0));
    
        for (int i = 0; i < threadNum; i++) {
            int startRow = i * rowsPerThread;
            int endRow = (i + 1) * rowsPerThread;

            if (i == threadNum - 1) {
                endRow += remainingRows;
            }

            auto threadFunction = [this, &other, &result, startRow, endRow, numCols]() {
                this->addMatricesInArea(other, result, startRow, endRow, 0, numCols);
            };
            
            std::thread thread1(threadFunction);
            threads.push_back(std::move(thread1));
        }

        for (auto& thread : threads) {
            thread.join();
        }
        return result;
    }
};

int main()
{
    srand(time(NULL));
    int dimension = 10000;
    Matrix m1(dimension, dimension);
    Matrix m2(dimension, dimension);

    m1.generate();
    m2.generate();

    //std::cout << "First matrix: " << "\n";
    //m1.print();
    //std::cout << "Second matrix: " << "\n";
    //m2.print();

    Matrix m3(dimension, dimension);

    //Single process time measurements
    auto start1 = chrono::high_resolution_clock().now();
    m3 = m1 + m2;
    auto end1 = chrono::high_resolution_clock().now();

    std::cout << "Dimension: " << dimension << "\n";

    //Output
    std::cout << "Time of NON-paralel addition: " << duration_cast<nanoseconds>(end1 - start1).count() << " nanoseconds\n";    
    //m3.print();

    //Multi-processing time measurements
    //vector<int> threads = { 2, 4, 8, 16 };

    std::vector<std::vector<int>> result;
    int threadNum = 2;

    auto start2 = chrono::high_resolution_clock().now();
    result = m1.addMatricesParallel(m2.matrix, threadNum);
    auto end2 = chrono::high_resolution_clock().now();

    std::cout << "Time of parallel addition: " << duration_cast<nanoseconds>(end2 - start2).count() << " nanoseconds\n";
    Matrix resultMatrix(dimension, dimension);
    resultMatrix.matrix = result;
}
