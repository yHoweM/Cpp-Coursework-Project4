#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<errno.h>
#include"../include/matrix_improved.h"
#include<cblas.h>
#include<sys/time.h>
#include<time.h>

#define parameter 8192


int main(){
    char * fpa = "mat-A-8192.txt";
    char * fpb = "mat-B-8192.txt";
    printf("            The size of the input matrixes is %d * %d\n",parameter,parameter);
    Matrix * mat1 = createMatrix(parameter,parameter,fpa);
    Matrix * mat2 = createMatrix(parameter,parameter,fpb);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    Matrix * mul_mat1 = matmul_plain(mat1,mat2);
    gettimeofday(&end, NULL);
    int timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Plain procession: %.4f s\n",((float) timeuse / 1000000.0f));
    deleteMatrix(&mul_mat1);

    printf("------------OpenMP procession is conducting--------\n");
    gettimeofday(&start, NULL);
    Matrix * mul_mat2 = matmul_improved_omp(mat1,mat1);
    gettimeofday(&end, NULL);
    timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("OpenMP procession: %.4f s\n",((float) timeuse / 1000000.0f));
    deleteMatrix(&mul_mat2);

    printf("------------Strassen algorithm procession is conducting----------\n");
    gettimeofday(&start, NULL);
    Matrix * mul_mat3 = matmul_improved_strassen(mat1,mat2);
    gettimeofday(&end, NULL);
    timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Strassen procession: %.4f s\n",((float) timeuse / 1000000.0f));
    deleteMatrix(&mul_mat3);

    printf("------------SIMD procession is conducting----------\n");
    gettimeofday(&start, NULL);
    Matrix * mul_mat4 = matmul_improved_SIMD(mat1,mat2);
    gettimeofday(&end, NULL);
    timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("SIMD procession: %.4f s\n",((float) timeuse / 1000000.0f));
    deleteMatrix(&mul_mat4);

    deleteMatrix(&mat1);
    deleteMatrix(&mat2);
    return 0;
}