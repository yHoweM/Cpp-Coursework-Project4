#ifndef MATRIX_IMPROVED_H
#define MATRIX_IMPROVED_H
#include<stddef.h>

typedef struct Matrix
{   size_t rows;
    size_t cols;
    float ** pData;
}Matrix;

void printMatrix(const Matrix * mat);

Matrix *create_random_matrix(const size_t rows, const size_t cols);

Matrix * createMatrix(const size_t row, const size_t col, const char * filename);

void deleteMatrix(Matrix ** p_mat);

Matrix * copyMatrix(const Matrix * mat);

Matrix * addMatrix(const  Matrix *mat1, const  Matrix *mat2);

Matrix * subtractMatrix(const  Matrix *mat1, const  Matrix *mat2);

Matrix * addScalar(const  Matrix * mat1, float num);    

Matrix * subtractScalar(const  Matrix * mat1, float num);

Matrix * mulScalar(const  Matrix * mat1, float num);

Matrix * mulMatrix(const  Matrix * mat1,const  Matrix * mat2);

Matrix * matmul_plain(const  Matrix * mat1,const  Matrix * mat2);

Matrix * matmul_improved_SIMD(const  Matrix * mat1,const  Matrix * mat2);

Matrix * matmul_improved_omp(const  Matrix * mat1,const  Matrix * mat2);

Matrix *  matmul_improved_strassen(const  Matrix * mat1,const  Matrix * mat2);

float matrixMax(const  Matrix * mat);     

float matrixMin(const  Matrix * mat);  

#endif