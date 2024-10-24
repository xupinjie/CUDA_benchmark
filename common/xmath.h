#include <math.h>

inline
int ceil(int a, int b) {
    return (a+b-1)/b;
}

inline
int round_up(int a, int b) {
    return (a+b-1)/b*b;
}

inline void dataRandom2D(float *matA, int M, int N, int lda) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matA[i * lda + j] = (float)rand() / RAND_MAX;
        }
    }
}

inline void valueSet2D(float *matA, int M, int N, int lda, float value) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matA[i * lda + j] = value;
        }
    }
}
