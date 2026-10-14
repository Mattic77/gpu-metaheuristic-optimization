#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>

// Constantes
/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
const int SELECTED_OBJ_FUNC = 0;
const int NUM_OF_PARTICLES  = 512;   // taille de la population NP
const int NUM_OF_DIMENSIONS = 3;     // dimension D du probleme (D <= 1024)
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;
const float phi = 3.1415;

// Parametres de la DE/rand/1/bin (valeurs de l'article)
const float F_SCALE = 0.5f;          // facteur de mutation F
const float CR_RATE = 0.3f;          // probabilite de croisement CR

// Critere d'arret de l'article : 10^4 * D evaluations de la fonction objectif
// (la population initiale compte pour 1 generation d'evaluations).
// Pour un budget plus grand, augmenter MAX_GEN directement.
const int MAX_FES = 10000 * NUM_OF_DIMENSIONS;
const int MAX_GEN = MAX_FES / NUM_OF_PARTICLES;

// Fonctions utiles cote CPU
float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[]);

// Fonction externe qui va tourner sur le GPU (cudaDE_i)
// gBest : meilleur individu trouve (taille NUM_OF_DIMENSIONS)
extern "C" void cuda_de(float *gBest, unsigned long long seed);
