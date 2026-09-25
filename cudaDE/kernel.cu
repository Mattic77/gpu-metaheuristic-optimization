#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>
#include <curand_kernel.h>
#include "kernel.h"

// Fonction de fitness sur le GPU
__device__ float fitness_function(float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 1;

    switch (SELECTED_OBJ_FUNC) {
        case 1: // Shifted Rastrigin's Function[cite: 1]
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i];
                res += pow(zi, 2) - 10 * cos(2 * phi * zi) + 10;
            }
            res -= 330;
            break;
        case 4: // Shifted Sphere's Function[cite: 1]
            for(int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                res += pow(x[i], 2);
            }
            res -= 450;
            break;
        default:
            break;
    }
    return res;
}

// 1. Noyau d'initialisation et d'évaluation combinés (kernel_IE)[cite: 2]
__global__ void kernel_IE(float *positions, float *fitnessVals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_PARTICLES) return;

    float ind[NUM_OF_DIMENSIONS];
    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        ind[j] = positions[idx * NUM_OF_DIMENSIONS + j];
    }
    fitnessVals[idx] = fitness_function(ind);
}

// 2. Noyau de préparation des indices de mutation r1, r2, r3 (kernel_P)[cite: 2]
__global__ void kernel_P(int *r1, int *r2, int *r3, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_PARTICLES) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    int id1, id2, id3;
    do { id1 = curand(&state) % NUM_OF_PARTICLES; } while (id1 == idx);
    do { id2 = curand(&state) % NUM_OF_PARTICLES; } while (id2 == idx || id2 == id1);
    do { id3 = curand(&state) % NUM_OF_PARTICLES; } while (id3 == idx || id3 == id1 || id3 == id2);

    r1[idx] = id1;
    r2[idx] = id2;
    r3[idx] = id3;
}

// 3. Noyau Composite Principal : Mutation + Crossover + Evaluation + Replacement (kernel_MCER)[cite: 2]
__global__ void kernel_MCER(float *positions, float *fitnessVals, int *r1, int *r2, int *r3, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_PARTICLES) return;

    curandState state;
    curand_init(seed + idx, idx, 0, &state);

    int i1 = r1[idx];
    int i2 = r2[idx];
    int i3 = r3[idx];

    int randDim = curand(&state) % NUM_OF_DIMENSIONS;
    float trial[NUM_OF_DIMENSIONS];

    // Mutation (DE/rand/1) & Croisement binomial (bin)[cite: 2]
    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        float valR1 = positions[i1 * NUM_OF_DIMENSIONS + j];
        float valR2 = positions[i2 * NUM_OF_DIMENSIONS + j];
        float valR3 = positions[i3 * NUM_OF_DIMENSIONS + j];
        float targetVal = positions[idx * NUM_OF_DIMENSIONS + j];

        float cr_rand = curand_uniform(&state);

        if (cr_rand <= CR || j == randDim) {
            // Formule de mutation différentielle : v = r1 + F * (r2 - r3)[cite: 2]
            trial[j] = valR1 + F * (valR2 - valR3);
        } else {
            trial[j] = targetVal;
        }
    }

    // Évaluation du vecteur d'essai (Trial Vector)[cite: 2]
    float trialFitness = fitness_function(trial);

    // Sélection / Remplacement[cite: 2]
    if (trialFitness < fitnessVals[idx]) {
        fitnessVals[idx] = trialFitness;
        for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
            positions[idx * NUM_OF_DIMENSIONS + j] = trial[j];
        }
    }
}

// Fonction hôte de pilotage CUDA pour le DE
extern "C" void cuda_de_i(float *positions, float *fitnessVals, float *bestSolution) {
    int sizePop = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS;

    float *devPos, *devFitness;
    int *devR1, *devR2, *devR3;

    cudaMalloc((void**)&devPos, sizeof(float) * sizePop);
    cudaMalloc((void**)&devFitness, sizeof(float) * NUM_OF_PARTICLES);
    cudaMalloc((void**)&devR1, sizeof(int) * NUM_OF_PARTICLES);
    cudaMalloc((void**)&devR2, sizeof(int) * NUM_OF_PARTICLES);
    cudaMalloc((void**)&devR3, sizeof(int) * NUM_OF_PARTICLES);

    cudaMemcpy(devPos, positions, sizeof(float) * sizePop, cudaMemcpyHostToDevice);

    // ... (début de la fonction cuda_de_i inchangé)

    int threadsNum = 64;
    int blocksNum = (NUM_OF_PARTICLES + threadsNum - 1) / threadsNum;

    // 1. Initialisation et Évaluation initiale
    kernel_IE<<<blocksNum, threadsNum>>>(devPos, devFitness);
    cudaDeviceSynchronize();

    int total_gens = MAX_ITER / NUM_OF_PARTICLES;
    
    printf("--- Debut de l'evolution differentielle (cudaDE_i) ---\n");

    // Boucle principale d'évolution (Générations)
    for (int iter = 0; iter < total_gens; iter++) {
        // 2. Préparation des indices aléatoires de mutation
        kernel_P<<<blocksNum, threadsNum>>>(devR1, devR2, devR3, time(NULL) + iter);
        
        // 3. Exécution du noyau composite MCER
        kernel_MCER<<<blocksNum, threadsNum>>>(devPos, devFitness, devR1, devR2, devR3, time(NULL) + iter);
        
        // OPTIONNEL : Afficher l'état toutes les 100 générations (pour suivre le processus)
        if (iter % 100 == 0 || iter == total_gens - 1) {
            cudaDeviceSynchronize();
            
            // Copier temporairement les fitness sur le CPU pour trouver le meilleur actuel
            cudaMemcpy(fitnessVals, devFitness, sizeof(float) * NUM_OF_PARTICLES, cudaMemcpyDeviceToHost);
            float current_best = fitnessVals[0];
            for (int i = 1; i < NUM_OF_PARTICLES; i++) {
                if (fitnessVals[i] < current_best) {
                    current_best = fitnessVals[i];
                }
            }
            printf("Generation %d / %d -> Meilleur score actuel : %f\n", iter, total_gens, current_best);
        }
    }
    cudaDeviceSynchronize();
    printf("--- Fin de l'optimisation ---\n");

    // Nettoyage de la mémoire GPU
    cudaFree(devPos);
    cudaFree(devFitness);
    cudaFree(devR1);
    cudaFree(devR2);
    cudaFree(devR3);
}