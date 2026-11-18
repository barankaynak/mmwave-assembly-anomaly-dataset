/* Faz C / C.1 test surucusu: float32 ozellik matrisini (N x 21, ham ikili) okur, orman kodunu cagirir.
   Cikti: <out>.pred  (N x 2 uint8: [pred_double_toplama, pred_gomulu_forest_predict])
          <out>.sum   (N x 6 float64: sinif basina agac-yaprak toplami, double). */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "forest_api.h"

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: driver feat.bin N outbase\n"); return 2; }
    long n = atol(argv[2]);
    float *X = (float *)malloc((size_t)n * FOREST_NFEAT * sizeof(float));
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 3; }
    if (fread(X, sizeof(float), (size_t)n * FOREST_NFEAT, f) != (size_t)n * FOREST_NFEAT) { fprintf(stderr, "short read\n"); return 4; }
    fclose(f);
    char p1[1024], p2[1024];
    snprintf(p1, sizeof p1, "%s.pred", argv[3]);
    snprintf(p2, sizeof p2, "%s.sum", argv[3]);
    FILE *fp = fopen(p1, "wb"), *fs = fopen(p2, "wb");
    for (long i = 0; i < n; i++) {
        const float *x = X + i * FOREST_NFEAT;
        double acc[6];
        forest_sum(x, acc);
        int b = 0;
        for (int c = 1; c < 6; c++) if (acc[c] > acc[b]) b = c;   /* esitlikte kucuk sinif */
        uint8_t pr[2] = { (uint8_t)b, (uint8_t)forest_predict(x) };
        fwrite(pr, 1, 2, fp);
        fwrite(acc, sizeof(double), 6, fs);
    }
    fclose(fp); fclose(fs); free(X);
    return 0;
}
