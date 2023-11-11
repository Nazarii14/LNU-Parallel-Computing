#pragma once
#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <complex>

using std::vector;
using std::complex;

template <typename T>
class Matrix
{
private:
	int rows, cols;
	vector<T> elements;
public:
	Matrix(int numOfRows, int numOfCols);
	Matrix(int numOfRows, int numOfCols, T* data);
	int getRows();
	int getCols();
	T operator()(int row, int col) const;
	T& operator()(int row, int col);
	T* data();
	const vector<T>& getElements();
};

#endif