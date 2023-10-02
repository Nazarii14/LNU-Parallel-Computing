using MathNet.Numerics.LinearAlgebra;
using System;
using System.Diagnostics;

namespace MatrixSum
{
    internal class Program
    {
        public static void oldMatrix()
        {
            uint rows = 2, columns = 2, threads = 2;

            var matrix1 = new Matrix(rows, columns);
            matrix1.Generate();
            var matrix2 = new Matrix(rows, columns);
            matrix2.Generate();

            Console.WriteLine($"Matrix 1: \n{matrix1}");
            Console.WriteLine($"Matrix 2: \n{matrix2}");

            var resultMatrix = new Matrix(rows, columns);
            resultMatrix = matrix1.SimpleMultiplication(matrix2);
            Console.WriteLine($"Simple multiplication: \n{resultMatrix}");

            resultMatrix = matrix1.ThreadMultParallel(matrix2, threads);
            Console.WriteLine($"Threads multiplication: \n{resultMatrix}");


            rows = 1000;
            columns = 1000;
            Console.WriteLine($"Dimensions:\nrows: {rows}\ncolumns: {columns}");

            matrix1 = new Matrix(rows, columns);
            matrix2 = new Matrix(rows, columns);

            var clock = System.Diagnostics.Stopwatch.StartNew();
            resultMatrix = matrix1.SimpleMultiplication(matrix2);
            clock.Stop();

            var singleThreadTime = clock.Elapsed;
            Console.WriteLine($"Single thread: {singleThreadTime}");

            var threadsNum = new uint[] { 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };

            foreach (var i in threadsNum)
            {
                clock = System.Diagnostics.Stopwatch.StartNew();
                resultMatrix = matrix1.ThreadsMultiplication(matrix2, i);
                clock.Stop();

                var acceleration = Math.Round(singleThreadTime / clock.Elapsed, 4);
                var efficiency = Math.Round(acceleration / i, 5);
                Console.WriteLine($"Threads: {i}\tTime: {clock.Elapsed}\tAcceleration: {acceleration}\tEfficiency: {efficiency}");
            }
        }

        public static void Main(string[] args)
        {
            
        }
    }
}