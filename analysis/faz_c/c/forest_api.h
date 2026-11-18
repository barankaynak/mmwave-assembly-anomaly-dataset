#ifndef FOREST_API_H
#define FOREST_API_H
#define FOREST_NFEAT 21
/* dagitilan cikarim: tek pencere (21 float32 ozellik) -> sinif 0..5; esitlikte kucuk sinif */
int forest_predict(const float *x);
/* yalniz test (-DWITH_SUM): sinif basina agac-yaprak degeri toplami (double) */
void forest_sum(const float *x, double *acc);
#endif
