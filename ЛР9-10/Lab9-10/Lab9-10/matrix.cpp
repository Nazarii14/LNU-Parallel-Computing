#include "matrix.h"

template <typename T>
Matrix<T>::Matrix(int numOfRows, int numOfCols)
	: rows(numOfRows), cols(numOfCols), elements(numOfRows* numOfCols) {}

template <typename T>
Matrix<T>::Matrix(int numOfRows, int numOfCols, T* data)
	: rows(numOfRows), cols(numOfCols), elements(data, data + numOfRows * numOfCols) {}

template <typename T>
int Matrix<T>::getRows()
{
	return rows;
}

template <typename T>
int Matrix<T>::getCols()
{
	return cols;
}

template <typename T>
T Matrix<T>::operator()(int row, int col) const
{
	return elements[cols * row + col];
}

template <typename T>
T& Matrix<T>::operator()(int row, int col)
{
	return elements[cols * row + col];
}

template <typename T>
T* Matrix<T>::data()
{
	return elements.data();
}

template <typename T>
const vector<T>& Matrix<T>::getElements()
{
	return elements;
}