#include "kernel.h"

int main(int argc, char** argv) {
    if(argc == 3){
        int dim = std::stoi(argv[1]);
        int pop = std::stoi(argv[2]);
    }
    
    // Allocation sur l'hôte (CPU) - Plus de tableau de vitesses !
    float positions[NUM_OF_PARTICLES * NUM_OF_DIMENSIONS];
    float fitnessVals[NUM_OF_PARTICLES];
    float bestSolution[NUM_OF_DIMENSIONS];
    
    printf("Type \t Time \t  \t Minimum\n");
    
    // Initialisation du random
    srand((unsigned) time(NULL));

    // Initialisation aléatoire de la population de départ
    for (int i = 0; i < NUM_OF_PARTICLES * NUM_OF_DIMENSIONS; i++) {
        positions[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
    }

    clock_t begin = clock();
    
    // Appel de la fonction principale GPU optimisée (cudaDE_i)
    cuda_de_i(positions, fitnessVals, bestSolution);    
    
    clock_t end = clock();
    
    // Affichage détaillé des résultats finaux
    printf("\n========================================\n");
    printf("          RESULTATS FINAUX DE `cudaDE_i`   \n");
    printf("========================================\n");
    printf("Temps d'execution : %.3lf secondes\n", (double)(end - begin) / CLOCKS_PER_SEC);
    printf("best solution %f \n",bestSolution);
    printf("Valeur minimale   : %f\n", host_fitness_function(bestSolution));
    
    printf("Meilleure solution (Coordonnees X) :\n");
    for(int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        printf("  x[%d] = %f\n", j, bestSolution[j]);
    }
    printf("========================================\n");

    return 0;
}