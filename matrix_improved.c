#include"../include/matrix_improved.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <immintrin.h>
#include <stdio.h>


void printMatrix(const Matrix * mat){
    if (mat==NULL)
    {
        printf("The matrix does not exist!\n");
        return;
    }
    else{
        for (size_t i = 0; i < mat->rows; i++)
        {
            for (size_t j = 0; j < mat->cols; j++)
            {
                printf("%f ",mat->pData[i][j]);
            }
            printf("\n");
        }
        return;
    }
}

// create a random matrix
Matrix *create_random_matrix(const size_t rows, const size_t cols) {
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    m->pData = (float **) malloc(sizeof(float *) * rows); // allocate memory for the first D of the 2D matrix
    for (size_t i = 0; i < rows; i++) {
        m->pData[i] = (float *) malloc(sizeof(float) * cols); // allocate memory for the second D of the 2D matrix
    }
    m->rows = rows;
    m->cols = cols;
    srand((unsigned)time(0));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            m->pData[i][j] = rand()/(float)(RAND_MAX/100);
        }
    }
    return m;
}


Matrix * createMatrix(const size_t row, const size_t col, const char * filename){
    if (row<0||col<0)
    {
        printf("The parameters should be positive integer!\n");
        return NULL;
    }
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    mat->rows = row;
    mat->cols = col;
    mat->pData =  (float **)malloc(row*sizeof(float*));
    //allocate memory for rows in **
    for (size_t i = 0; i < row; i++)
    {
        mat->pData[i] = (float *)malloc(col*sizeof(float));
        //allocate memory for cols in *
    }
    if (filename!=NULL)
    {
        FILE *file = fopen(filename,"r");
    if (file==NULL)
        {
            printf("Input file does not exist.\n");
            return NULL;
        }
    else
        {
            float temp;
            for (size_t i = 0; i < row; i++)
            {
                for (size_t j = 0; j < col; j++)
                {
                    fscanf(file, "%f", &temp);
                    mat->pData[i][j] = temp;
                }
            }
        }
    }
    return mat;
}

void deleteMatrix(Matrix ** p_mat){
    Matrix * mat = *p_mat;
    if (mat==NULL)
    {
        printf("The aimed matrix does not exist!\n");
        return;
    }
    else
    {
        for (size_t i = 0; i < mat->rows; i++)
        {
            free(mat->pData[i]);//free the cols for mat
        }
        free(mat->pData);//free the rows for mat
        // free(mat);//free the created structure 
        // printf("The matrix has been deleted.\n");
        memset(mat, 0, sizeof(Matrix)); //this step is not needed, but to ensure privacy
        *p_mat = NULL;
        return;
    }
}

Matrix * copyMatrix(const Matrix * mat){
    Matrix * mat_copy = createMatrix(mat->rows,mat->cols,NULL);
    if (mat==NULL)
    {
        printf("The aimed matrix does not exist!\n");
        return NULL;
    }
    for (size_t i = 0; i < mat->rows; i++)
    {
        for (size_t j = 0; j < mat->cols; j++)
        {
            mat_copy->pData[i][j] = mat->pData[i][j];
        }
    }
    // try to convert from 2D to 1D
    // float * data1 = (float *)aligned_alloc(32,mat1->rows*mat1->cols*sizeof(float));
    // float * data1_copy = (float *)aligned_alloc(32,mat1->rows*mat1->cols*sizeof(float));
    // for (size_t i = 0; i < mat->rows; i++)
    // {
    //     for (size_t j = 0; j < mat->cols; j++)
    //     {
    //         data1[i*mat1->rows+j] = mat->pData[i][j];
    //     }
    // }
    // memcpy(data1_copy,data1,sizeof(data1)/sizeof(data1_copy[0]));
    return mat_copy;
}

Matrix * addMatrix( const Matrix *mat1, const  Matrix *mat2){
    Matrix * mat_add = createMatrix(mat1->rows,mat1->cols,NULL);
    if (mat1==NULL||mat2==NULL)
    {
        printf("The input matrixes do not exist!\n");
        return NULL;
    }
    else if (mat1->cols!=mat2->cols||mat1->rows!=mat2->rows)
    {
        printf("The size of input matrixes is not the same!\n");
        return NULL;
    }
    else 
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat1->cols; j++)
        {
            mat_add->pData[i][j] = mat1->pData[i][j]+mat2->pData[i][j];
        }
    }
    return mat_add;
}

Matrix * subtractMatrix(const  Matrix *mat1, const  Matrix *mat2){
    Matrix * mat_sub = createMatrix(mat1->rows,mat1->cols,NULL);
    if (mat1==NULL||mat2==NULL)
    {
        printf("The input matrixes do not exist!\n");
        return NULL;
    }
    else if (mat1->cols!=mat2->cols||mat1->rows!=mat2->rows)
    {
        printf("The size of input matrixes is not the same!\n");
        return NULL;
    }
    else 
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat1->cols; j++)
        {
            mat_sub->pData[i][j] = mat1->pData[i][j]-mat2->pData[i][j];
        }
    }
    return mat_sub;
}

Matrix * addScalar(const  Matrix * mat1, float num){
    Matrix * mat_add_sca = createMatrix(mat1->rows,mat1->cols,NULL);
    if (mat1==NULL)
    {
        printf("The input matrix does not exist!\n");
        return NULL;
    }
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat1->cols; j++)
        {
            mat_add_sca->pData[i][j] = mat1->pData[i][j]+num;
        }
    }
    return mat_add_sca;
}

Matrix * subtractScalar( const  Matrix * mat1, float num){
    Matrix * mat_sub_sca = createMatrix(mat1->rows,mat1->cols,NULL);
    if (mat1==NULL)
    {
        printf("The input matrix does not exist!\n");
        return NULL;
    }
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat1->cols; j++)
        {
            mat_sub_sca->pData[i][j] = mat1->pData[i][j]-num;
        }
    }
    return mat_sub_sca;
}

Matrix * mulScalar(const  Matrix * mat1, float num){
    Matrix * mat_mul_sca = createMatrix(mat1->rows,mat1->cols,NULL);
    if (mat1==NULL)
    {
        printf("The input matrix does not exist!\n");
        return NULL;
    }
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat1->cols; j++)
        {
            mat_mul_sca->pData[i][j] = mat1->pData[i][j]*num;
        }
    }
    return mat_mul_sca;
}

Matrix * mulMatrix(const  Matrix * mat1,const  Matrix * mat2){
    if (mat1->cols!=mat2->rows)
    {
        printf("The column of the first matrix and the row of the second matrix is not the same!\n");
        return NULL;
    }
    Matrix * mat_mul = createMatrix(mat1->rows,mat2->cols,NULL);
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat2->cols; j++)
        {
            float sum = 0;
            for (size_t x = 0; x < mat1->cols; x++)
            {
                sum += mat1->pData[i][x]*mat2->pData[x][j];
            }
            mat_mul->pData[i][j] = sum;
        }
    }
    return mat_mul;
}

//a straightforward way
Matrix * matmul_plain(const  Matrix * mat1,const  Matrix * mat2){//many original loops
    if (mat1->cols!=mat2->rows)
    {
        printf("The column of the first matrix or the row of the second matrix is not the same!\n");
        return NULL;
    }
    Matrix * mat_mul = createMatrix(mat1->rows,mat2->cols,NULL);
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat2->cols; j++)
        {
            float sum = 0;
            for (size_t x = 0; x < mat1->cols; x++)
            {
                sum += mat1->pData[i][x]*mat2->pData[x][j];
            }
            mat_mul->pData[i][j] = sum;
        }
    }
    return mat_mul;
}

//SIMD
Matrix * matmul_improved_SIMD(const  Matrix * mat1,const  Matrix * mat2){//many original loops
    if (mat1->cols!=mat2->rows)
    {
        printf("The column of the first matrix or the row of the second matrix is not the same!\n");
        return NULL;
    }
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();
    // try to convert from 2D to 1D
    float * data1 = (float *)aligned_alloc(32,mat1->rows*mat1->cols*sizeof(float));
    float * data2 = (float *)aligned_alloc(32,mat2->rows*mat2->cols*sizeof(float));
    float * data3 = (float *)aligned_alloc(32,mat1->rows * mat2->cols*sizeof(float));
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat1->cols; j++)
        {
            data1[i*mat1->rows+j] = mat1->pData[i][j];
        }
    }
    for (size_t i = 0; i < mat2->rows; i++)
    {
        for (size_t j = 0; j < mat2->cols; j++)
        {
            data2[i*mat1->rows+j] = mat2->pData[i][j];
        }
    }
    float * data2_new = (float *)aligned_alloc(32, mat2->rows * mat2->cols * sizeof(float));
    for (size_t i = 0; i < mat2->rows * mat2->cols; i++)
    {
      *(data2_new + i) = *(data2 + (i / mat2->rows) + (i % mat2->rows) * mat2->cols);
    }
    float * data3_new = (float *)aligned_alloc(32, mat1->rows * mat2->cols * sizeof(float));
    //tag
    for (size_t i = 0; i < mat1->rows*mat2->cols; i++)
    {
        for (size_t j = 0; j < mat1->cols; j+=8)
        {   
        
            a = _mm256_load_ps(data1+(i / mat1->cols) * mat1->cols + j);
            b = _mm256_load_ps(data2_new+(i % mat2->cols) * mat2->rows+j);
            c =  _mm256_add_ps(c, _mm256_mul_ps(a, b));
            _mm256_store_ps(data3, c);
        }

        for (size_t x = 0; x < 8; x++)
        {
            *(data3_new + i) += *(data3 + x);
        }
        
    }
    // data 3 is the matmul in the end
    Matrix * mat_mul = createMatrix(mat1->rows,mat2->cols,NULL);
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat2->cols; j++)
        {
            mat_mul->pData[i][j] = data3_new[i*mat1->rows+j];
        }
        
    }
    free(data1);
    free(data2);
    free(data3);
    free(data2_new);
    free(data3_new);
    return mat_mul;
}

//OpenMP
Matrix *  matmul_improved_omp(const  Matrix * mat1,const  Matrix * mat2){
    if (mat1->cols!=mat2->rows)
    {
        printf("The column of the first matrix or the row of the second matrix is not the same!\n");
        return NULL;
    }
    // if (mat1->rows <= 8)
    // {
    //     return matmul_plain(mat1, mat2);
    // }
    Matrix * mat_mul = createMatrix(mat1->rows,mat2->cols,NULL);
    omp_set_num_threads(32);
    #pragma omp parallel for
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat2->cols; j++)
        {
            float sum = 0;
            for (size_t x = 0; x < mat1->cols; x++)
            {
                sum += mat1->pData[i][x]*mat2->pData[x][j];
            }
            mat_mul->pData[i][j] = sum;
        }
    }
    return mat_mul;
}

//Strassen
Matrix *  matmul_improved_strassen(const  Matrix * mat1, const  Matrix * mat2){
    if (mat1->cols!=mat2->rows)
    {
        printf("The column of the first matrix or the row of the second matrix is not the same!\n");
        return NULL;
    }
    if (mat1->rows<=128)
    {
        return matmul_plain(mat1,mat2);
    }
    Matrix *S[10];
    for (size_t i = 0; i < 10; i++) {
        S[i] = createMatrix(mat1->rows/2, mat1->cols/2,NULL);
    }
    Matrix *P[7];
    for (size_t i = 0; i < 7; i++) {
        P[i] = createMatrix(mat1->rows/2, mat1->cols/2,NULL);
    }
    
    Matrix *A11, *A12, *A21, *A22;
    Matrix *B11, *B12, *B21, *B22;
    Matrix *C11, *C12, *C21, *C22;
    A11 = createMatrix(mat1->rows / 2, mat1->cols / 2,NULL);
    A12 = createMatrix(mat1->rows / 2, mat1->cols / 2,NULL);
    A21 = createMatrix(mat1->rows / 2, mat1->cols / 2,NULL);
    A22 = createMatrix(mat1->rows / 2, mat1->cols / 2,NULL);
    B11 = createMatrix(mat2->rows / 2, mat2->cols / 2,NULL);
    B12 = createMatrix(mat2->rows / 2, mat2->cols / 2,NULL);
    B21 = createMatrix(mat2->rows / 2, mat2->cols / 2,NULL);
    B22 = createMatrix(mat2->rows / 2, mat2->cols / 2,NULL);
    C11 = createMatrix(mat1->rows / 2, mat2->cols / 2,NULL);
    C12 = createMatrix(mat1->rows / 2, mat2->cols / 2,NULL);
    C21 = createMatrix(mat1->rows / 2, mat2->cols / 2,NULL);
    C22 = createMatrix(mat1->rows / 2, mat2->cols / 2,NULL);
    //value assignment for block
    for (size_t i = 0; i <mat1->rows / 2 ; i++)
    {
        for (size_t j = 0; j <mat1->cols / 2 ; j++)
        {   
            A11->pData[i][j] = mat1->pData[i][j];
        }
    }
    for (size_t i = 0; i <mat1->rows / 2 ; i++)
    {
        for (size_t j = mat1->cols/2; j <mat1->cols ; j++)
        {
            A12->pData[i][j-mat1->cols/2] = mat1->pData[i][j];
        }
    }
    for (size_t i = mat1->rows / 2; i <mat1->rows; i++)
    {
        for (size_t j = 0; j <mat1->cols/2 ; j++)
        {
            A21->pData[i-mat1->rows/2][j] = mat1->pData[i][j];
        }
    }
    for (size_t i = mat1->rows / 2; i <mat1->rows; i++)
    {
        for (size_t j = mat1->cols/2; j <mat1->cols ; j++)
        {
            A21->pData[i-mat1->rows/2][j-mat1->cols/2] = mat1->pData[i][j];
        }
    }
    for (size_t i = 0; i <mat2->rows / 2 ; i++)
    {
        for (size_t j = 0; j <mat2->cols / 2 ; j++)
        {   
            B11->pData[i][j] = mat2->pData[i][j];
        }
    }
    for (size_t i = 0; i <mat2->rows / 2 ; i++)
    {
        for (size_t j = mat2->cols/2; j <mat2->cols ; j++)
        {
            B12->pData[i][j-mat2->cols/2] = mat2->pData[i][j];
        }
    }
    for (size_t i = mat2->rows / 2; i <mat2->rows; i++)
    {
        for (size_t j = 0; j <mat2->cols/2 ; j++)
        {
            B21->pData[i-mat2->rows/2][j] = mat2->pData[i][j];
        }
    }
    for (size_t i = mat2->rows / 2; i <mat2->rows; i++)
    {
        for (size_t j = mat2->cols/2; j <mat2->cols ; j++)
        {
            B21->pData[i-mat2->rows/2][j-mat2->cols/2] = mat2->pData[i][j];
        }
    }
    S[0] = subtractMatrix(B12,B22);
    S[1] = addMatrix(A11,A12);
    S[2] = addMatrix(A21,A22);
    S[3] = subtractMatrix(B21,B11);
    S[4] = addMatrix(A11,A22);
    S[5] = addMatrix(B11,B12);
    S[6] = subtractMatrix(A12,A22);
    S[7] = addMatrix(B21,B22);
    S[8] = subtractMatrix(A11,A21);
    S[9] = addMatrix(B11,B12);

    P[0] = matmul_improved_strassen(A11,S[0]);
    P[1] = matmul_improved_strassen(S[1],B22);
    P[2] = matmul_improved_strassen(S[2],B11);
    P[3] = matmul_improved_strassen(A22,S[3]);
    P[4] = matmul_improved_strassen(S[4],S[5]);
    P[5] = matmul_improved_strassen(S[6],S[7]);
    P[6] = matmul_improved_strassen(S[8],S[9]);

    C11 = addMatrix(subtractMatrix(addMatrix(P[4],P[3]),P[1]),P[5]);
    C12 = addMatrix(P[0],P[1]);
    C21 = addMatrix(P[2],P[3]);
    C22 = subtractMatrix(subtractMatrix(addMatrix(P[4],P[0]),P[2]),P[6]);

    //give the value back to C matrix
    Matrix * mat_mul = createMatrix(mat1->rows,mat2->cols,NULL);
    for (size_t i = 0; i < mat1->rows; i++)
    {
        for (size_t j = 0; j < mat2->cols; j++)
        {
            if  (i<mat1->rows/2&j<mat2->cols/2){
                mat_mul->pData[i][j] = C11->pData[i][j];
            }
            else if (i<mat1->rows/2&j>=mat2->cols/2)
            {
                mat_mul->pData[i][j] = C12->pData[i][j-mat2->cols/2];
            }
            else if (i>=mat1->rows/2&j<mat2->cols/2)
            {
                mat_mul->pData[i][j] = C21->pData[i-mat1->rows/2][j];
            }
            else if (i>=mat1->rows/2&j>=mat2->cols/2)
            {
                mat_mul->pData[i][j] = C22->pData[i-mat1->rows/2][j-mat2->rows/2];
            }
        }
    }
    return mat_mul;
}



float matrixMax(const  Matrix * mat){
    float max = 0;
    for (size_t i = 0; i < mat->rows; i++)
    {
        for (size_t j = 0; j < mat->cols; j++)
        {
            max = (max < mat->pData[i][j])?mat->pData[i][j] : max;
        }
    }
    return max;
}

float matrixMin(const  Matrix * mat){
    float min = 0;
    for (size_t i = 0; i < mat->rows; i++)
    {
        for (size_t j = 0; j < mat->cols; j++)
        {
            min = (min < mat->pData[i][j])?min : mat->pData[i][j];
        }
    }
    return min;
}