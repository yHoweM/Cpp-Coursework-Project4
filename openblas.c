#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <sys/time.h>

#define parameter 8192

int main() {
    CBLAS_ORDER Order = CblasRowMajor;
    CBLAS_TRANSPOSE TransA = CblasNoTrans;
    CBLAS_TRANSPOSE TransB = CblasNoTrans;
    const float alpha = 1;
    const float beta = 0;
    float *m1 = (float *) malloc(sizeof(float *) * (parameter*parameter));
    float *m2 = (float *) malloc(sizeof(float *) * (parameter*parameter));
    float *m3 = (float *) malloc(sizeof(float *) * (parameter*parameter));
    FILE *fpa;
    if ((fpa = fopen("mat-A-8192.txt", "r")) == NULL) {
        printf("fail\n");
        exit(0);
    } else
        printf("done\n");
    float a;
    for (int i = 0; i <parameter; i++) {
        for (int j = 0; j <parameter; j++) {
            fscanf(fpa, "%f,", &a);
            m1[i * parameter+ j] = a;
        }
    }
    fclose(fpa);

    FILE *fpb;
    if ((fpb = fopen("mat-B-8192.txt", "r")) == NULL) {
        printf("fail\n");
        exit(0);
    } else
        printf("done\n");
    float b;
    for (int i = 0; i <parameter; i++) {
        for (int j = 0; j <parameter; j++) {
            fscanf(fpb, "%f,", &b);
            m2[i * parameter+ j] = b;
        }
    }
    fclose(fpb);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    printf("start\n");
    cblas_sgemm(Order, TransA, TransB,parameter,parameter,parameter, alpha, m1,parameter, m2,parameter, beta, m3,parameter);
    cblas_sgemm(Order, TransA, TransB,parameter,parameter,parameter, alpha, m1,parameter, m2,parameter, beta, m3,parameter);
    cblas_sgemm(Order, TransA, TransB,parameter,parameter,parameter, alpha, m1,parameter, m2,parameter, beta, m3,parameter);
    cblas_sgemm(Order, TransA, TransB,parameter,parameter,parameter, alpha, m1,parameter, m2,parameter, beta, m3,parameter);
    cblas_sgemm(Order, TransA, TransB,parameter,parameter,parameter, alpha, m1,parameter, m2,parameter, beta, m3,parameter);
    gettimeofday(&end, NULL);
    int timeuse =1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("Time cost of OpenBLAS for %d-size: %.4f s\n", parameter, ((float) timeuse / (5 * 1000000.0f)));
    return 0;
}