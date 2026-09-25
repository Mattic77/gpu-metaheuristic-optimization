#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>

// Constantes globales
/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
const int SELECTED_OBJ_FUNC = 1;
const int NUM_OF_PARTICLES = 512;
const int NUM_OF_DIMENSIONS = 10;
const int MAX_ITER = NUM_OF_DIMENSIONS * pow(10, 4); // Condition d'arrêt standard[cite: 1, 2]
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;

// Paramètres spécifiques de l'Évolution Différentielle (DE)
const float F = 0.5f;   // Facteur d'amplification de la mutation
const float CR = 0.9f;  // Taux de croisement
const float phi = 3.1415f;

// Fonctions utilitaires
float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[]);

// Nouvelle fonction externe pour le DE sur GPU
extern "C" void cuda_de_i(float *positions, float *fitnessVals, float *bestSolution);

#endif