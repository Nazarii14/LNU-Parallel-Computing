using System;
using System.Data.SqlClient;
using System.Drawing;
using System.Security.Cryptography;
using System.Threading;

namespace MatrixSum
{
    public class Matrix
    {
        public uint Height { get; set; }
        public uint Width { get; set; }
        public int[,] Mtrx { get; set; }

        public Matrix(uint height, uint width)
        {
            Height = height;
            Width = width;
            Mtrx = new int[Height, Width];
        }
        public Matrix(Matrix other)
        {
            Height = other.Height;
            Width = other.Width;
            Mtrx = new int[Height, Width];
            Array.Copy(other.Mtrx, Mtrx, Height * Width);
        }
        public Matrix(uint height) 
        {
            Height = height;
            Width = height;
            Mtrx = new int[Height, Width];
        }
        public void Generate()
        {
            Random rand = new Random();
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    Mtrx[i, j] = rand.Next(1000);
            }
        }
        public Matrix SimpleAddition(Matrix other)
        {
            if (Height != other.Height || Width != other.Width)
                throw new ArgumentException("Matrix must have same dimensions");

            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    Mtrx[i, j] += other.Mtrx[i, j];
            }
            return this;
        }
        public Matrix ThreadsAddition(Matrix other, uint amount)
        {
            uint start = 0;
            var step = (Width * Height) / (amount);
            var end = step + 1;

            Thread[] threads = new Thread[amount];

            for (int i = 0; i < amount; i++)
            {
                var start_ = start;
                var end_ = end;

                threads[i] = new Thread(() =>
                {
                    for (var j = start_; j <= end_; j++)
                        Mtrx[j / Width, j % Width] += other.Mtrx[j / Width, j % Width];
                });

                threads[i].Start();
                start = end + 1;
                end = i == amount - 2 ? Height * Width - 1 : end + step;
            }

            foreach (var thread in threads)
                thread.Join();
            
            return this;
        }
        public Matrix SimpleMultiplication(Matrix other)
        {
            if (Width != other.Height)
                throw new InvalidOperationException("Dimensions does not match");

            Matrix resultMatrix = new Matrix(Height, other.Width);

            for (uint i = 0; i < Height; i++)
            {
                for (uint j = 0; j < other.Width; j++)
                {
                    var currentResult = 0;
                    for (uint k = 0; k < Width; k++)
                        currentResult += Mtrx[i, k] * other.Mtrx[k, j];
                    
                    resultMatrix.Mtrx[i, j] = currentResult;
                }
            }

            return resultMatrix;
        }
        public Matrix ThreadsMultiplication(Matrix other, uint amount)
        {
            if (Width != other.Height)
                throw new InvalidOperationException("Dimensions does not match");

            uint start = 0;
            var step = (Width * Height) / amount;
            var end = step + 1;

            Matrix resultMatrix = new Matrix(Height, other.Width);

            Thread[] threads = new Thread[amount];

            for (int i = 0; i < amount; i++)
            {
                var start_ = start;
                var end_ = end;

                threads[i] = new Thread(() =>
                {
                    for (uint j = start_; j < end_; j++)
                    {
                        uint row = j / Width;
                        uint col = j % Width;
                        int currentResult = 0;

                        for (var k = 0; k < Width; k++)
                            currentResult += Mtrx[row, k] * other.Mtrx[k, col];

                        resultMatrix.Mtrx[row, col] = currentResult;
                    }
                });

                threads[i].Start();
                start = end;
                end = (i == amount - 2) ? (Width * Height) : (end + step);
            }

            foreach (var thread in threads)
                thread.Join();

            return resultMatrix;
        }
        public Matrix ThreadMultParallel(Matrix other, uint amount)
        {
            ParallelOptions parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = (int)amount
            };

            Matrix resultMatrix = new Matrix(Height, other.Width);

            Parallel.For(0, Height, parallelOptions, i =>
            {
                for (int j = 0; j < other.Width; j++)
                {
                    int sum = 0;
                    for (int k = 0; k < Width; k++)
                    {
                        sum += Mtrx[i, k] * other.Mtrx[k, j];
                    }
                    resultMatrix.Mtrx[i, j] = sum;
                }
            });

            return resultMatrix;
        }
        public override string ToString()
        {
            string result = "";
            for (int i = 0; i < Height; i++)
            {
                for (int j = 0; j < Width; j++)
                    result += Mtrx[i, j].ToString() + " ";
                result += "\n";
            }
            return result;
        }
    }
}