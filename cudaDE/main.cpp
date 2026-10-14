#include "kernel.h"

int main(int argc, char** argv) {

    float gBest[NUM_OF_DIMENSIONS];

    printf("Type \t Time \t  \t Minimum\n");

    // La population est initialisee directement sur le GPU (kernel IE)
    unsigned long long seed = (unsigned long long) time(NULL);

    clock_t begin = clock();
    cuda_de(gBest, seed);
    clock_t end = clock();

    printf("GPU \t ");
    printf("%10.3lf \t", (double)(end - begin)/CLOCKS_PER_SEC);
    printf(" %f\n", host_fitness_function(gBest));

    return 0;
}
