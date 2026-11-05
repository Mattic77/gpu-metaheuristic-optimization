#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

#include "kernel.h"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s (%s:%d)\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* ------------------------------------------------------------------------
 * Fonction objectif, decomposee par element (1 thread = 1 element d'un individu)
 *
 * Chaque thread j calcule sa contribution "sum" (et "prod" pour Griewank),
 * puis les D contributions sont reduites en memoire partagee.
 *
 * Objective function
 * 0: Levy 3-dimensional
 * 1: Shifted Rastigrin's Function
 * 2: Shifted Rosenbrock's Function
 * 3: Shifted Griewank's Function
 * 4: Shifted Sphere's Function
 * ---------------------------------------------------------------------- */
__device__ void elementTerm(const float *x, int j, float &sum, float &prod)
{
    sum = 0.0f;
    prod = 1.0f;

    switch (SELECTED_OBJ_FUNC) {
        case 0: {
            float y1 = 1 + (x[0] - 1) / 4;
            float yn = 1 + (x[NUM_OF_DIMENSIONS - 1] - 1) / 4;
            if (j == 0) {
                float s = sinf(phi * y1);
                sum += s * s;
            }
            if (j < NUM_OF_DIMENSIONS - 1) {
                float y  = 1 + (x[j] - 1) / 4;
                float yp = 1 + (x[j + 1] - 1) / 4;
                float s  = sinf(phi * yp);
                sum += (y - 1) * (y - 1) * (1 + 10 * s * s) + (yn - 1) * (yn - 1);
            }
            break;
        }
        case 1: {
            float zi = x[j];
            sum = zi * zi - 10 * cosf(2 * phi * zi) + 10;
            break;
        }
        case 2: {
            if (j < NUM_OF_DIMENSIONS - 1) {
                float zi   = x[j] + 1;
                float zip1 = x[j + 1] + 1;
                float a = zi * zi - zip1;
                sum = 100 * a * a + (zi - 1) * (zi - 1);
            }
            break;
        }
        case 3: {
            float zi = x[j];
            sum  = zi * zi / 4000;
            prod = cosf(zi / sqrtf((float)(j + 1)));
            break;
        }
        case 4: {
            float zi = x[j];
            sum = zi * zi;
            break;
        }
    }
}

// Constantes de decalage appliquees apres la reduction
__device__ float finishFitness(float sum, float prod)
{
    switch (SELECTED_OBJ_FUNC) {
        case 1:  return sum - 330;
        case 2:  return sum + 390;
        case 3:  return sum - prod + 1 - 180;
        case 4:  return sum - 450;
        default: return sum;
    }
}

/* ------------------------------------------------------------------------
 * Evaluation parallele de tous les individus du bloc.
 * Disposition : blockDim.x = membersPerBlock * D, thread tid -> individu
 * tid / D, element tid % D. Tous les threads du bloc doivent appeler cette
 * fonction (elle contient des __syncthreads).
 * La valeur retournee n'est valide que pour le thread j == 0 de chaque individu.
 * ---------------------------------------------------------------------- */
__device__ float blockFitness(const float *s_trial, float *s_term, float *s_prod,
                              int tid, int j)
{
    const float *x = s_trial + (tid - j);   // vecteur de l'individu courant

    float t, p;
    elementTerm(x, j, t, p);
    s_term[tid] = t;
    s_prod[tid] = p;

    // Reduction en arbre (D quelconque, pas forcement puissance de 2)
    int pw = 1;
    while (pw < NUM_OF_DIMENSIONS) pw <<= 1;
    for (int s = pw >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if (j < s && j + s < NUM_OF_DIMENSIONS) {
            s_term[tid] += s_term[tid + s];
            s_prod[tid] *= s_prod[tid + s];
        }
    }
    return finishFitness(s_term[tid], s_prod[tid]);
}

// Memoire partagee necessaire pour un bloc de m individus
static size_t sharedBytes(int m)
{
    return (size_t)(3 * m * NUM_OF_DIMENSIONS) * sizeof(float) + (size_t)m * sizeof(int);
}

/* ------------------------------------------------------------------------
 * kernel(IE) : initialisation de la population + evaluation (fusionnes)
 * ---------------------------------------------------------------------- */
__global__ void kernelIE(float *pop, float *fit, curandState *stE,
                         curandState *stP, unsigned long long seed)
{
    extern __shared__ float smem[];
    const int m = blockDim.x / NUM_OF_DIMENSIONS;
    float *s_trial = smem;
    float *s_term  = smem + m * NUM_OF_DIMENSIONS;
    float *s_prod  = smem + 2 * m * NUM_OF_DIMENSIONS;

    const int tid    = threadIdx.x;
    const int mem    = tid / NUM_OF_DIMENSIONS;
    const int j      = tid % NUM_OF_DIMENSIONS;
    const int member = blockIdx.x * m + mem;
    const bool valid = member < NUM_OF_PARTICLES;
    const int gid    = member * NUM_OF_DIMENSIONS + j;

    if (valid) {
        curandState s;
        curand_init(seed, gid, 0, &s);
        float x = START_RANGE_MIN + curand_uniform(&s) * (START_RANGE_MAX - START_RANGE_MIN);
        pop[gid]     = x;
        s_trial[tid] = x;
        stE[gid]     = s;

        // etat aleatoire du kernel(P) : un par individu
        if (j == 0)
            curand_init(seed, NUM_OF_PARTICLES * NUM_OF_DIMENSIONS + member, 0, &stP[member]);
    } else {
        s_trial[tid] = 0.0f;
    }
    __syncthreads();

    float f = blockFitness(s_trial, s_term, s_prod, tid, j);
    if (valid && j == 0)
        fit[member] = f;
}

/* ------------------------------------------------------------------------
 * kernel(P) : pour chaque individu i, tire r1, r2, r3 deux a deux differents
 * et differents de i. Ne depend pas de la population -> peut tourner en
 * parallele de kernel(MCER) dans un autre stream.
 * ---------------------------------------------------------------------- */
__global__ void kernelP(int *idx, curandState *stP)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_OF_PARTICLES) return;

    curandState s = stP[i];
    int r1, r2, r3;
    do { r1 = curand(&s) % NUM_OF_PARTICLES; } while (r1 == i);
    do { r2 = curand(&s) % NUM_OF_PARTICLES; } while (r2 == i || r2 == r1);
    do { r3 = curand(&s) % NUM_OF_PARTICLES; } while (r3 == i || r3 == r1 || r3 == r2);
    stP[i] = s;

    idx[3 * i + 0] = r1;
    idx[3 * i + 1] = r2;
    idx[3 * i + 2] = r3;
}

/* ------------------------------------------------------------------------
 * kernel(MCER) : Mutation + Crossover + Evaluation + Replacement fusionnes.
 * Lit la generation courante (popCur/fitCur), ecrit la suivante
 * (popNext/fitNext) : double buffering pour eviter toute course entre blocs
 * (un individu peut etre lu comme r1/r2/r3 par un autre bloc).
 * Le vecteur d'essai reste en memoire partagee : aucun aller-retour en
 * memoire globale entre les 4 operations.
 * ---------------------------------------------------------------------- */
__global__ void kernelMCER(const float *popCur, const float *fitCur,
                           float *popNext, float *fitNext,
                           const int *idx, curandState *stE)
{
    extern __shared__ float smem[];
    const int m = blockDim.x / NUM_OF_DIMENSIONS;
    float *s_trial = smem;
    float *s_term  = smem + m * NUM_OF_DIMENSIONS;
    float *s_prod  = smem + 2 * m * NUM_OF_DIMENSIONS;
    int   *s_k     = (int *)(smem + 3 * m * NUM_OF_DIMENSIONS);

    const int tid    = threadIdx.x;
    const int mem    = tid / NUM_OF_DIMENSIONS;
    const int j      = tid % NUM_OF_DIMENSIONS;
    const int member = blockIdx.x * m + mem;
    const bool valid = member < NUM_OF_PARTICLES;
    const int gid    = member * NUM_OF_DIMENSIONS + j;

    curandState s;
    float xOld = 0.0f;

    if (valid) {
        s = stE[gid];
        xOld = popCur[gid];
        // indice k : au moins un element du vecteur d'essai vient du mutant
        if (j == 0)
            s_k[mem] = curand(&s) % NUM_OF_DIMENSIONS;
    }
    __syncthreads();

    if (valid) {
        // Mutation  V = X_r1 + F * (X_r2 - X_r3)
        int r1 = idx[3 * member + 0];
        int r2 = idx[3 * member + 1];
        int r3 = idx[3 * member + 2];
        float v = popCur[r1 * NUM_OF_DIMENSIONS + j]
                + F_SCALE * (popCur[r2 * NUM_OF_DIMENSIONS + j]
                           - popCur[r3 * NUM_OF_DIMENSIONS + j]);

        // Crossover binomial
        float u = (curand_uniform(&s) <= CR_RATE || j == s_k[mem]) ? v : xOld;

        // Rester dans le domaine de recherche
        u = fminf(fmaxf(u, START_RANGE_MIN), START_RANGE_MAX);
        s_trial[tid] = u;
        stE[gid] = s;
    } else {
        s_trial[tid] = 0.0f;
    }
    __syncthreads();

    // Evaluation du vecteur d'essai
    float fTrial = blockFitness(s_trial, s_term, s_prod, tid, j);

    // Le thread j == 0 decide du remplacement et le partage via s_k
    if (valid && j == 0)
        s_k[mem] = (fTrial <= fitCur[member]) ? 1 : 0;
    __syncthreads();

    // Replacement
    if (valid) {
        bool better = (s_k[mem] == 1);
        popNext[gid] = better ? s_trial[tid] : xOld;
        if (j == 0)
            fitNext[member] = better ? fTrial : fitCur[member];
    }
}

/* ------------------------------------------------------------------------
 * Configuration d'execution automatique (section 4 de l'article) :
 * on choisit le nombre d'individus par bloc m (1 thread par element,
 * threads = m * D) qui maximise le nombre de threads residents par SM
 * en tenant compte de la memoire partagee, des registres et des limites
 * du GPU (via l'API d'occupancy). A occupancy egale, on prend le plus grand m.
 * On borne aussi m pour que la grille contienne au moins ~1 bloc par SM.
 * ---------------------------------------------------------------------- */
static void chooseConfig(int &members, int &threads, size_t &smem)
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    if (NUM_OF_DIMENSIONS > prop.maxThreadsPerBlock) {
        fprintf(stderr, "D=%d depasse maxThreadsPerBlock=%d : non supporte\n",
                NUM_OF_DIMENSIONS, prop.maxThreadsPerBlock);
        exit(EXIT_FAILURE);
    }

    int maxM = prop.maxThreadsPerBlock / NUM_OF_DIMENSIONS;
    int perSM = (NUM_OF_PARTICLES + prop.multiProcessorCount - 1) / prop.multiProcessorCount;
    if (maxM > NUM_OF_PARTICLES) maxM = NUM_OF_PARTICLES;
    if (maxM > perSM) maxM = perSM;
    if (maxM < 1) maxM = 1;

    int bestActive = -1;
    members = 1;
    for (int m = 1; m <= maxM; m++) {
        int    t = m * NUM_OF_DIMENSIONS;
        size_t s = sharedBytes(m);
        if (s > prop.sharedMemPerBlock) break;

        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kernelMCER, t, s));
        if (nb * t >= bestActive) {
            bestActive = nb * t;
            members = m;
        }
    }
    threads = members * NUM_OF_DIMENSIONS;
    smem    = sharedBytes(members);
}

/* ------------------------------------------------------------------------
 * cudaDE_i : boucle principale cote hote
 * Deux streams : kernel(MCER) de la generation g tourne pendant que
 * kernel(P) de la generation g+1 remplit l'autre matrice d'indices.
 * Deux matrices d'indices (idx[0], idx[1]) sont echangees a chaque generation ;
 * des events garantissent qu'on n'ecrase pas une matrice encore lue.
 * ---------------------------------------------------------------------- */
extern "C" void cuda_de(float *gBest, unsigned long long seed)
{
    const int NP   = NUM_OF_PARTICLES;
    const int size = NP * NUM_OF_DIMENSIONS;

    int members, threads;
    size_t smem;
    chooseConfig(members, threads, smem);
    const int blocks = (NP + members - 1) / members;

    const int pThreads = 128;
    const int pBlocks  = (NP + pThreads - 1) / pThreads;

    // Memoire GPU
    float *devPop[2], *devFit[2];
    int   *devIdx[2];
    curandState *devStE, *devStP;
    for (int b = 0; b < 2; b++) {
        CUDA_CHECK(cudaMalloc((void**)&devPop[b], sizeof(float) * size));
        CUDA_CHECK(cudaMalloc((void**)&devFit[b], sizeof(float) * NP));
        CUDA_CHECK(cudaMalloc((void**)&devIdx[b], sizeof(int) * 3 * NP));
    }
    CUDA_CHECK(cudaMalloc((void**)&devStE, sizeof(curandState) * size));
    CUDA_CHECK(cudaMalloc((void**)&devStP, sizeof(curandState) * NP));

    cudaStream_t sM, sP;
    cudaEvent_t evP[2], evM[2];
    CUDA_CHECK(cudaStreamCreate(&sM));
    CUDA_CHECK(cudaStreamCreate(&sP));
    for (int b = 0; b < 2; b++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evP[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evM[b], cudaEventDisableTiming));
    }

    // kernel(IE) puis kernel(P) de la generation 0
    kernelIE<<<blocks, threads, smem>>>(devPop[0], devFit[0], devStE, devStP, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernelP<<<pBlocks, pThreads, 0, sP>>>(devIdx[0], devStP);
    CUDA_CHECK(cudaEventRecord(evP[0], sP));

    // Boucle principale
    for (int gen = 0; gen < MAX_GEN; gen++)
    {
        const int cur = gen & 1;     // buffers lus par MCER (population et indices)
        const int nxt = cur ^ 1;     // buffers ecrits

        // kernel(P) de la generation suivante, en parallele de MCER
        if (gen + 1 < MAX_GEN) {
            if (gen >= 1)            // MCER(gen-1) doit avoir fini de lire idx[nxt]
                CUDA_CHECK(cudaStreamWaitEvent(sP, evM[nxt], 0));
            kernelP<<<pBlocks, pThreads, 0, sP>>>(devIdx[nxt], devStP);
            CUDA_CHECK(cudaEventRecord(evP[nxt], sP));
        }

        // kernel(MCER) : attend la matrice d'indices de cette generation
        CUDA_CHECK(cudaStreamWaitEvent(sM, evP[cur], 0));
        kernelMCER<<<blocks, threads, smem, sM>>>(devPop[cur], devFit[cur],
                                                  devPop[nxt], devFit[nxt],
                                                  devIdx[cur], devStE);
        CUDA_CHECK(cudaEventRecord(evM[cur], sM));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Meilleur individu de la derniere generation (un seul transfert)
    const int last = MAX_GEN & 1;
    float *hPop = (float*)malloc(sizeof(float) * size);
    float *hFit = (float*)malloc(sizeof(float) * NP);
    CUDA_CHECK(cudaMemcpy(hPop, devPop[last], sizeof(float) * size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hFit, devFit[last], sizeof(float) * NP, cudaMemcpyDeviceToHost));

    int best = 0;
    for (int i = 1; i < NP; i++)
        if (hFit[i] < hFit[best]) best = i;
    for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
        gBest[k] = hPop[best * NUM_OF_DIMENSIONS + k];

    free(hPop);
    free(hFit);

    // Nettoyage
    for (int b = 0; b < 2; b++) {
        cudaEventDestroy(evP[b]);
        cudaEventDestroy(evM[b]);
        cudaFree(devPop[b]);
        cudaFree(devFit[b]);
        cudaFree(devIdx[b]);
    }
    cudaStreamDestroy(sM);
    cudaStreamDestroy(sP);
    cudaFree(devStE);
    cudaFree(devStP);
}
