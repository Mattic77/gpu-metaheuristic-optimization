#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "kernel.h"

// Keep this implementation in sync with host_fitness_function in kernel.cpp.
__device__ float fitness_function(const float x[]) {
    float res = 0.0f;

    switch (SELECTED_OBJ_FUNC)  {
        case 0: {
            float y1 = 1 + (x[0] - 1)/4;
            float yn = 1 + (x[NUM_OF_DIMENSIONS-1] - 1)/4;

            res += pow(sin(phi*y1), 2);

            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float y = 1 + (x[i] - 1)/4;
                float yp = 1 + (x[i+1] - 1)/4;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        }
        case 1: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*phi*zi) + 10;
            }
            res -= 330;
            break;
        }
        
        case 2: {
            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i+1] - 0 + 1;
                res += 100 * ( pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        }
        case 3: {
            float somme = 0.0f;
            float produit = 1.0f;
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2)/4000;
                produit *= cos(zi/pow(i+1, 0.5));
            }
            res = somme - produit + 1 - 180;
            break;
        }
        case 4: {
            for(int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
        }
        default:
            break;
    }

    return res;
}

// Initial evaluation: one thread evaluates one individual.
__global__ void kernel_IE(const float *positions, float *fitnessVals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_PARTICLES) return;

    float ind[NUM_OF_DIMENSIONS];
    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        ind[j] = positions[idx * NUM_OF_DIMENSIONS + j];
    }
    fitnessVals[idx] = fitness_function(ind);
}

// Select three distinct mutation partners for every target individual.
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

// Composite mutation, crossover, evaluation, and selection kernel.
__global__ void kernel_MCER(const float *positions, const float *fitnessVals,
                            float *nextPositions, float *nextFitness,
                            int *r1, int *r2, int *r3, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_OF_PARTICLES) return;

    curandState state;
    curand_init(seed + idx, idx, 0, &state);

    int i1 = r1[idx];
    int i2 = r2[idx];
    int i3 = r3[idx];

    int randDim = curand(&state) % NUM_OF_DIMENSIONS;
    float trial[NUM_OF_DIMENSIONS];

    // DE/rand/1 mutation and binomial crossover.
    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        float valR1 = positions[i1 * NUM_OF_DIMENSIONS + j];
        float valR2 = positions[i2 * NUM_OF_DIMENSIONS + j];
        float valR3 = positions[i3 * NUM_OF_DIMENSIONS + j];
        float targetVal = positions[idx * NUM_OF_DIMENSIONS + j];

        float cr_rand = curand_uniform(&state);

        if (cr_rand <= CR || j == randDim) {
            // Differential mutation: v = r1 + F * (r2 - r3).
            trial[j] = valR1 + F * (valR2 - valR3);
        } else {
            trial[j] = targetVal;
        }
    }

    // Evaluate the trial vector.
    float trialFitness = fitness_function(trial);

    // Greedy selection: keep the trial when it is no worse.
    const bool accept = trialFitness <= fitnessVals[idx];
    nextFitness[idx] = accept ? trialFitness : fitnessVals[idx];
    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        const float value = accept ? trial[j] : positions[idx * NUM_OF_DIMENSIONS + j];
        nextPositions[idx * NUM_OF_DIMENSIONS + j] =
            fminf(START_RANGE_MAX, fmaxf(START_RANGE_MIN, value));
    }
}

// Fonction hôte de pilotage CUDA pour le DE
extern "C" void cuda_de_i(float *positions, float *fitnessVals, float *bestSolution) {
    int sizePop = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS;

    float *devPos, *devPosNext, *devFitness, *devFitnessNext;
    int *devR1, *devR2, *devR3;

    cudaMalloc((void**)&devPos, sizeof(float) * sizePop);
    cudaMalloc((void**)&devPosNext, sizeof(float) * sizePop);
    cudaMalloc((void**)&devFitness, sizeof(float) * NUM_OF_PARTICLES);
    cudaMalloc((void**)&devFitnessNext, sizeof(float) * NUM_OF_PARTICLES);
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

    int total_gens = MAX_EVALUATIONS > NUM_OF_PARTICLES
        ? (MAX_EVALUATIONS - NUM_OF_PARTICLES) / NUM_OF_PARTICLES : 0;
    float *currentPos = devPos;
    float *nextPos = devPosNext;
    float *currentFitness = devFitness;
    float *nextFitness = devFitnessNext;
    
    printf("--- Debut de l'evolution differentielle (cudaDE_i) ---\n");

    // Boucle principale d'évolution (Générations)
    for (int iter = 0; iter < total_gens; iter++) {
        // 2. Préparation des indices aléatoires de mutation
        kernel_P<<<blocksNum, threadsNum>>>(devR1, devR2, devR3, time(NULL) + iter);

        // 3. Exécution du noyau composite MCER
        kernel_MCER<<<blocksNum, threadsNum>>>(currentPos, currentFitness,
                                               nextPos, nextFitness,
                                               devR1, devR2, devR3,
                                               time(NULL) + iter);

        float *posSwap = currentPos;
        currentPos = nextPos;
        nextPos = posSwap;
        float *fitnessSwap = currentFitness;
        currentFitness = nextFitness;
        nextFitness = fitnessSwap;
        
        // OPTIONNEL : Afficher l'état toutes les 100 générations (pour suivre le processus)
        if (iter % 100 == 0 || iter == total_gens - 1) {
            cudaDeviceSynchronize();
            
            // Copier temporairement les fitness sur le CPU pour trouver le meilleur actuel
            cudaMemcpy(fitnessVals, currentFitness, sizeof(float) * NUM_OF_PARTICLES, cudaMemcpyDeviceToHost);
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


    // =========================================================================
    //  ADD THE FIX HERE (Copy best results back from GPU to CPU host)
    // =========================================================================
    cudaMemcpy(fitnessVals, currentFitness, sizeof(float) * NUM_OF_PARTICLES, cudaMemcpyDeviceToHost);
    cudaMemcpy(positions, currentPos, sizeof(float) * sizePop, cudaMemcpyDeviceToHost);

    // Find the particle index with the lowest (best) fitness score
    int bestIdx = 0;
    float minFit = fitnessVals[0];
    for (int i = 1; i < NUM_OF_PARTICLES; i++) {
        if (fitnessVals[i] < minFit) {
            minFit = fitnessVals[i];
            bestIdx = i;
        }
    }

    // Populate bestSolution buffer for main.cpp
    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        bestSolution[j] = positions[bestIdx * NUM_OF_DIMENSIONS + j];
    }

    // Nettoyage de la mémoire GPU
    cudaFree(devPos);
    cudaFree(devPosNext);
    cudaFree(devFitness);
    cudaFree(devFitnessNext);
    cudaFree(devR1);
    cudaFree(devR2);
    cudaFree(devR3);
}
