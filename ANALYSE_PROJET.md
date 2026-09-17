# Analyse Globale du Projet CUDA-PSO & Architecture GPU

## 1. Contexte Académique et Objectifs du Projet

Ce document présente une analyse exhaustive et détaillée du projet de calcul haute performance **CUDA-PSO** (Particle Swarm Optimization sur GPU), développé dans le cadre du cours de calcul parallèle (Master IM & IMDS à l'Université de Haute-Alsace - UHA, sous la direction du Pr. Lhassane Idoumghar).

### 1.1 Objectifs principaux
1. **Comprendre et analyser** la mise en œuvre de l'algorithme d'optimisation par essaim particulaire (PSO - *Particle Swarm Optimization*) sur architecture parallèle GPU via NVIDIA CUDA.
2. **Identifier les choix d'architecture**, le modèle d'exécution, la gestion mémoire, les goulots d'étranglement et les vulnérabilités de l'implémentation existante.
3. **Préparer la transition/extension** vers d'autres métaheuristiques massivement parallèles, notamment **l'Évolution Différentielle** (DE - *Differential Evolution*), en s'appuyant sur le cours CUDA et les travaux de recherche de référence (*Qin et al., "An Improved CUDA-Based Implementation of Differential Evolution on GPU"*).

---

## 2. Fondements Thématiques et Architecturaux de CUDA
*(Synthèse basée sur le support de cours `courscuda.pdf`)*

### 2.1 Comparaison des architectures : CPU (Multi-core) vs GPU (Many-core)

| Caractéristique | CPU (ex. Intel Core i7 / Xeon) | GPU (ex. NVIDIA GTX / Tesla / RTX) |
| :--- | :--- | :--- |
| **Modèle d'exécution** | **MIMD** (Multiple Instructions, Multiple Data) | **SIMD / SIMT** (Single Instruction, Multiple Threads) |
| **Philosophie des Cœurs** | Quelques cœurs hyper-optimisés (clock élevée, prédiction de branchement, exécution superscalaire) | Des milliers de petits cœurs de calcul (ALU / SP - Streaming Processors) |
| **Gestion de la Mémoire** | Latence minimale grâce à de très grands caches (L1, L2, L3, L4) | Débit maximal (High Bandwidth) au détriment de la latence individuelle |
| **Cas d'usage optimal** | Traitements séquentiels complexes et branchements conditionnels lourds | Traitements massivement parallèles sur de grands tableaux de données |

Dans le domaine de l'optimisation métaheuristique (PSO, DE), où des centaines ou milliers d'individus (particules ou vecteurs) subissent des opérations mathématiques strictement identiques à chaque génération, le modèle GPU offre un facteur d'accélération d'un à plusieurs ordres de grandeur par rapport au CPU.

---

### 2.2 Modèle Logique d'Exécution CUDA
CUDA organise l'exécution sous forme d'une hiérarchie à 3 niveaux :
1. **Thread** : L'unité élémentaire d'exécution qui exécute une instance du *kernel*. Chaque thread possède un identifiant unique accessible via la variable intégrée `threadIdx` (en 1D, 2D ou 3D).
2. **Bloc de Threads (CTA - Cooperative Thread Array)** : Un groupe de threads (jusqu'à 512, 1024 ou 2048 selon la *Compute Capability* de la carte) s'exécutant sur le même **SM** (*Streaming Multiprocessor*). Les threads d'un même bloc peuvent se synchroniser (`__syncthreads()`) et partager des données via la mémoire partagée (`__shared__`).
3. **Grille (Grid)** : L'ensemble des blocs déployés pour exécuter un kernel. Représentée par `gridDim` et indexée par `blockIdx`.

#### Formule d'Indexation Globale 1D :
Pour associer de manière unique un thread à un élément de données dans la mémoire globale :
$$\text{gtid} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

---

### 2.3 Modèle Physique et Exécution Matérielle
* **Streaming Multiprocessor (SM)** : Le GPU est composé de plusieurs SMs. Chaque SM contient ses propres cœurs d'exécution, registres, cache L1 et mémoire partagée.
* **Warps & Half-Warps** : Sur le plan matériel, les threads d'un bloc sont regroupés par paquets de **32 threads** appelés **Warps**.
  * Les 32 threads d'un warp exécutent la même instruction simultanément.
  * Si les threads d'un warp empruntent des branches conditionnelles différentes (`if/else`), l'exécution est sérialisée (**Divergence de Warp**), dégradant les performances.

---

### 2.4 Hiérarchie Mémoire CUDA et Règles de Performance

```
+-------------------------------------------------------------------+
|                        MÉMOIRE GLOBALE (DRAM)                     |
|            (Accès lent, 400-800 cycles, nécessite la coalescence) |
+-------------------------------------------------------------------+
       ^                                      ^
       |                                      |
+--------------+                      +--------------+
| CACHE L2     |                      | MÉMOIRE CONST.| (Cached, Read-Only)
+--------------+                      +--------------+
       ^
       |
+-------------------------------------------------------------------+
|                     STREAMING MULTIPROCESSOR (SM)                 |
|  +---------------------------+    +----------------------------+  |
|  | Mémoire Partagée (Shared) |    | Registres (Ultra-rapides)  |  |
|  | (~1-2 cycles, inter-thread)  |    | (Propres à chaque thread)  |  |
|  +---------------------------+    +----------------------------+  |
|  +-------------------------------------------------------------+  |
|  | Mémoire Locale (Local Mem - Spillover registres vers DRAM) |  |
|  +-------------------------------------------------------------+  |
+-------------------------------------------------------------------+
```

#### Principes d'Optimisation Mémoire :
1. **Coalescence des accès à la mémoire globale** : Tous les threads d'un *half-warp* (16 threads) ou warp (32 threads) doivent accéder à des éléments mémoire contigus (segments de 32, 64 ou 128 octets). Des accès non alignés engendrent plusieurs transactions mémoire distinctes.
2. **Utilisation de la Mémoire Partagée (`__shared__`)** : Très rapide (équivalente aux registres). Permet d'éviter d'aller chercher les mêmes données plusieurs fois en mémoire globale.
3. **Mémoire Constante (`__constant__`)** : Mise en cache dédiée. Idéale pour les paramètres algorithmiques constants lus simultanément par tous les threads.

---

### 2.5 Modèle de Compilation NVCC et API CUDA
Le compilateur `nvcc` sépare le code source :
* Code **Host** (exécuté sur le CPU) $\rightarrow$ Compilé par GCC/MSVC.
* Code **Device** (*kernels* `__global__` et fonctions `__device__`) $\rightarrow$ Compilé en code intermédiaire **PTX** puis en binaire machine GPU (**SASS**).

#### Cycle de vie synchrone standard :
1. `cudaMalloc` : Allocation de la mémoire GPU (Device).
2. `cudaMemcpy(HostToDevice)` : Copie des données initiales de la RAM vers le GPU.
3. Launch Kernel : `kernel<<<grid, block>>>(params)` (Non-bloquant pour le CPU).
4. `cudaMemcpy(DeviceToHost)` : Transfert synchrone des résultats du GPU vers la RAM.
5. `cudaFree` : Libération de la mémoire GPU.

---

## 3. Analyse Détaillée de l'Architecture du Code Source `cudapso`

Le projet est constitué de 5 fichiers principaux :
* [kernel.h](file:///c:/Users/youne/Downloads/cudapso/kernel.h) : Constantes et prototypes.
* [main.cpp](file:///c:/Users/youne/Downloads/cudapso/main.cpp) : Point d'entrée Host (CPU).
* [kernel.cpp](file:///c:/Users/youne/Downloads/cudapso/kernel.cpp) : Évaluations de fitness et fonctions utilitaires sur Host.
* [kernel.cu](file:///c:/Users/youne/Downloads/cudapso/kernel.cu) : Kernels CUDA et wrapper GPU.
* [CudaPSO.ipynb](file:///c:/Users/youne/Downloads/cudapso/CudaPSO.ipynb) : Notebook Jupyter pour exécution Google Colab (GPU T4, CUDA 10.0).

---

### 3.1 Définition des Fonctions Benchmark (Problèmes d'Optimisation)

Le projet intègre 5 fonctions objectifs d'optimisation numérique (*Shifted & Benchmark Functions*), sélectionnées via la constante `SELECTED_OBJ_FUNC` :

1. **Fonction 0 : Levy 3D** ($F_0$)
   $$y_i = 1 + \frac{x_i - 1}{4}$$
   $$f(x) = \sin^2(\pi y_1) + \sum_{i=1}^{D-1} (y_i - 1)^2 [1 + 10 \sin^2(\pi y_{i+1})] + (y_D - 1)^2$$
2. **Fonction 1 : Shifted Rastrigin** ($F_1$)
   $$f(x) = \sum_{i=1}^{D} [z_i^2 - 10 \cos(2\pi z_i) + 10] - 330, \quad z = x - o$$
3. **Fonction 2 : Shifted Rosenbrock** ($F_2$)
   $$f(x) = \sum_{i=1}^{D-1} [100(z_i^2 - z_{i+1})^2 + (z_i - 1)^2] + 390$$
4. **Fonction 3 : Shifted Griewank** ($F_3$)
   $$f(x) = \sum_{i=1}^{D} \frac{z_i^2}{4000} - \prod_{i=1}^{D} \cos\left(\frac{z_i}{\sqrt{i}}\right) + 1 - 180$$
5. **Fonction 4 : Shifted Sphere** ($F_4$)
   $$f(x) = \sum_{i=1}^{D} z_i^2 - 450$$

---

### 3.2 Modélisation Algorithmique du Particle Swarm Optimization (PSO)

Dans l'implémentation PSO du projet :
* **Nombre de particules ($N$)** : `NUM_OF_PARTICLES = 512`
* **Nombre de dimensions ($D$)** : `NUM_OF_DIMENSIONS = 3`
* **Taille totale des tableaux** : $N \times D = 512 \times 3 = 1536$ éléments.
* **Paramètres PSO** : $\omega = 0.5$ (inertie), $c_1 = 1.5$ (coefficient cognitif), $c_2 = 1.5$ (coefficient social).

#### Équations de mise à jour des particules :
1. **Mise à jour des vitesses** :
   $$v_{i,d}^{t+1} = \omega \cdot v_{i,d}^t + c_1 \cdot r_1 \cdot (pBest_{i,d} - x_{i,d}^t) + c_2 \cdot r_2 \cdot (gBest_d - x_{i,d}^t)$$
2. **Mise à jour des positions** :
   $$x_{i,d}^{t+1} = x_{i,d}^t + v_{i,d}^{t+1}$$

---

### 3.3 Organisation des Threads et Kernels GPU

Dans [kernel.cu](file:///c:/Users/youne/Downloads/cudapso/kernel.cu) :
* Configuration d'exécution :
  * `threadsNum = 32` (Taille d'un bloc = 1 warp).
  * `blocksNum = ceil(1536 / 32) = 48` blocs.
  * Nombre total de threads lancés = $48 \times 32 = 1536$ threads (1 thread par composante scalaire particule-dimension).

#### 1. Kernel `kernelUpdateParticle`
Chaque thread $i \in [0, N \cdot D - 1]$ traite une unique dimension d'une particule :
* Identifie la particule correspondante et l'indice de dimension via `i % NUM_OF_DIMENSIONS`.
* Met à jour la vitesse `velocities[i]` et la position `positions[i]`.

#### 2. Kernel `kernelUpdatePBest`
Destiné à mettre à jour les meilleures positions personnelles ($pBest$) :
* Seuls les threads avec `i % NUM_OF_DIMENSIONS == 0` (soit le premier thread de chaque particule) effectuent l'évaluation.
* Copie la position courante et le $pBest$ courant de la particule dans deux tableaux temporaires `tempParticle1` et `tempParticle2`.
* Calcule $f(\text{tempParticle1})$ et $f(\text{tempParticle2})$. Si la position courante est meilleure, la position $pBest$ de la particule est mise à jour.

#### 3. Boucle Principale Host-Device (`cuda_pso`)
Pour chaque itération `iter` de $0$ à `MAX_ITER - 1` :
1. Lancement de `kernelUpdateParticle<<<48, 32>>>`.
2. Lancement de `kernelUpdatePBest<<<48, 32>>>`.
3. **Copie Device-to-Host** : Transfert de `pBests` du GPU vers la RAM du CPU.
4. **Calcul de `gBest` sur CPU** : Parcours séquentiel des $pBest$ sur le CPU avec `host_fitness_function`.
5. **Copie Host-to-Device** : Transfert de la nouvelle valeur de `gBest` vers la mémoire du GPU.

---

## 4. Analyse Critique : Vulnérabilités, Goulots d'Étranglement et Erreurs de Conception

Bien que le code fonctionne de manière basique, une analyse approfondie révèle plusieurs **défauts majeurs de performance et de correction logicielle** :

```
                  +----------------------------------------------+
                  |  PROBLÈMES MAJEURS DANS L'ARCHITECTURE DE   |
                  |                CUDAPSO EXISTANT              |
                  +----------------------------------------------+
                                         |
     +-------------------+---------------+-------------------+-------------------+
     |                   |                                   |                   |
     v                   v                                   v                   v
+-----------------+ +-------------------------+ +------------------------+ +------------------------+
| GOULOT PCI-E    | | DATA RACE CRITIQUE      | | DIVERGENCE & SOUS-     | | GÉNÉRATEUR ALÉATOIRE   |
| 60 000 transferts | | tempParticle1/2 sont   | | UTILISATION DE THREADS | | CPU (rand()) passé en |
| Host<->Device   | | __device__ globaux !    | | seuls 1/3 threads     | | arguments identiques  |
| dans la boucle! | | Undefined Behavior !    | | travaillent pBest!    | | à tous les threads ! |
+-----------------+ +-------------------------+ +------------------------+ +------------------------+
```

### 4.1 Vulnérabilité Majeure 1 : Condition de Course (Data Race Hazard)
Dans [kernel.cu](file:///c:/Users/youne/Downloads/cudapso/kernel.cu#L8-L9) :
```cuda
__device__ float tempParticle1[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];
```
* Ces deux tableaux sont déclarés en **mémoire globale device** sans portée restreinte par thread.
* Lorsque 512 threads (1 par particule) exécutent `kernelUpdatePBest` en parallèle, **tous les threads écrivent simultanément dans les mêmes cases mémoire `tempParticle1` et `tempParticle2`** !
* **Conséquence** : Destruction des données par chevauchement (Data Race), entraînant un comportement indéterminé (*Undefined Behavior*) et des erreurs de calcul graves.
* **Correction requise** : Déclarer les tableaux temporaires en mémoire locale au thread (`float tempParticle1[NUM_OF_DIMENSIONS];` à l'intérieur du kernel) ou utiliser la mémoire partagée (`__shared__`).

---

### 4.2 Goulot d'Étranglement 2 : Transferts PCI-e Excessifs dans la Boucle Principale
Dans la boucle principal de `cuda_pso` :
```cpp
for (int iter = 0; iter < MAX_ITER; iter++) {
    ...
    cudaMemcpy(pBests, devPBest, sizeof(float)*size, cudaMemcpyDeviceToHost);
    // Calcul de gBest sur CPU...
    cudaMemcpy(devGBest, gBest, sizeof(float)*NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);
}
```
* Pour `MAX_ITER = 30 000`, le programme effectue **60 000 transferts synchrone à travers le bus PCI-e**.
* La latence du bus PCI-e ruine complètement les performances du GPU.
* **Correction requise** : Conserver l'intégralité du calcul de $gBest$ sur GPU grâce à un kernel de réduction parallèle (*Parallel Reduction*).

---

### 4.3 Inefficacité 3 : Divergence de Threads et Faible Occupancy
Dans `kernelUpdatePBest` :
```cuda
if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
    return;
```
* Seul 1 thread sur 3 (pour $D=3$) effectue un travail utile. Les 2/3 des threads créés s'arrêtent immédiatement.
* La structure de bloc `threadsNum = 32` ne permet pas d'exploiter la pleine capacité d'un SM (qui accepte jusqu'à 2048 threads résidents).

---

### 4.4 Inefficacité 4 : Génération Aléatoire Centrée CPU
* Le Host génère deux nombres aléatoires `r1 = getRandomClamped()` et `r2 = getRandomClamped()` et les passe en argument au kernel `kernelUpdateParticle`.
* Par conséquent, **tous les threads et toutes les particules reçoivent exactement les mêmes coefficients aléatoires $r_1$ et $r_2$** au cours d'une itération donnée.
* **Correction requise** : Utiliser la bibliothèque CUDA **cuRAND** (`curand_uniform`) pour fournir des nombres aléatoires indépendants par particule et par dimension.

---

## 5. Spécifications pour l'Extension vers Differential Evolution (DE) & Architecture Optimisée
*(Inspiré de l'article de référence `An_Improved_CUDA-Based-Implementation_of_Different.pdf`)*

Pour faire évoluer le projet vers l'Évolution Différentielle (DE) et corriger les défauts identifiés, l'architecture doit respecter les principes suivants :

```
                  +----------------------------------------------+
                  |    ARCHITECTURE CUDADEi AMÉLIORÉE PROPOSÉE   |
                  +----------------------------------------------+
                                         |
     +-------------------+---------------+-------------------+
     |                                   |                   |
     v                                   v                   v
+------------------------+   +-----------------------+   +-----------------------+
| FUSION DES KERNELS     |   + GÉNÉRATION ALÉATOIRE  |   | GESTION PAR STREAMS   |
| Combine Mutation,      |   | GPU VIA CURAND        |   | Executions asynchrones|
| Crossover, Évaluation, |   | Indépendance totale   |   | et masquage des       |
| Sélection (MCER)       |   | des threads GPU       |   | latences mémoire      |
+------------------------+   +-----------------------+   +-----------------------+
```

### 5.1 Principes Algorithmiques de l'Évolution Différentielle (DE/rand/1/bin)
Pour chaque vecteur cible $x_i$ ($i = 1 \dots N$) :
1. **Mutation** : Choix de 3 individus distincs $r1, r2, r3 \neq i$ :
   $$v_i = x_{r1} + F \cdot (x_{r2} - x_{r3})$$
2. **Crossover Binomial** :
   $$u_{i,j} = \begin{cases} v_{i,j} & \text{si } \text{rand}_j(0,1) \le CR \text{ ou } j == k \\ x_{i,j} & \text{sinon} \end{cases}$$
3. **Sélection / Remplacement** :
   $$x_i^{t+1} = \begin{cases} u_i & \text{si } f(u_i) \le f(x_i) \\ x_i^t & \text{sinon} \end{cases}$$

---

### 5.2 Optimisations Architecturales GPU Recommandées

#### 1. Composite Kernel (`kernelMCER`)
Au lieu d'exécuter 6 kernels séparés (conduisant à des accès répétitifs à la mémoire globale), fusionner les opérations dans un **Kernel Composite (Mutation, Crossover, Evaluation, Replacement - MCER)**.
* Charge la population en **Mémoire Partagée (`__shared__`)**.
* Réalise la mutation et le croisement en registres/mémoire partagée.
* Calcule la fitness et effectue le remplacement sans réécrire inutilement dans la mémoire globale DRAM.

#### 2. Calcul du $gBest$ par Réduction Parallèle sur GPU
Remplacer le rapatriement Host par une réduction parallèle GPU :
* Chaque bloc calcule son minimum local en mémoire partagée.
* Une réduction globale détermine l'optimum en une fraction de milliseconde sans aucun transfert PCI-e.

#### 3. Configuration Automatique de la Taille des Blocs (SM Occupancy)
Ajuster dynamiquement `blockDim` et `gridDim` en fonction du nombre de dimensions $D$ et de la taille de population $N$ pour garantir une **Occupancy de 100% sur chaque SM**.

---

## 6. Protocole Expérimental d'Évaluation

Pour valider les performances théoriques et pratiques de l'implémentation :

### 6.1 Matrice de Tests Expérimentaux
* **Dimensions du problème ($D$)** : $10, 50, 100$
* **Taille de population ($N$)** : $50, 100, 500$
* **Nombre maximal d'évaluations** : $10^4 \times D$ (soit $10^5$ pour $D=10$, $10^6$ pour $D=100$)
* **Nombre de runs indépendants** : 10 à 25 exécutions par configuration (avec graines aléatoires distinctes).

### 6.2 Métriques à Reporter
1. **EFV (Error Function Value)** : $|f(x^*_{\text{trouvé}}) - f(x^*_{\text{optimum}})|$ (Moyenne et Écart-type).
2. **Temps d'exécution (en millisecondes/secondes)** : Mesuré via `cudaEventRecord` et `cudaEventElapsedTime`.
3. **Taux de Succès (SR - Success Rate)** : Pourcentage de runs atteignant un EFV $< 10^{-8}$.
4. **Analyse Statistique** : Tests de comparaison non paramétriques de **Wilcoxon** ou **Kruskal-Wallis** pour valider la significativité statistique des accélérations obtenues par rapport à la version séquentielle CPU.

---

## 7. Synthèse des Fichiers et Chemins d'Accès du Projet

| Composant | Emplacement du Fichier | Description |
| :--- | :--- | :--- |
| **En-tête GPU** | [kernel.h](file:///c:/Users/youne/Downloads/cudapso/kernel.h) | Déclaration des constantes, prototypes et paramètres PSO |
| **Source CPU** | [main.cpp](file:///c:/Users/youne/Downloads/cudapso/main.cpp) | Initialisation, boucle d'exécution CPU et timing |
| **Fitness Host** | [kernel.cpp](file:///c:/Users/youne/Downloads/cudapso/kernel.cpp) | Évaluation des 5 fonctions benchmark sur CPU |
| **Kernels CUDA** | [kernel.cu](file:///c:/Users/youne/Downloads/cudapso/kernel.cu) | Kernels de mise à jour des positions, $pBest$ et wrapper Host-Device |
| **Notebook Jupyter**| [CudaPSO.ipynb](file:///c:/Users/youne/Downloads/cudapso/CudaPSO.ipynb) | Environnement de compilation et d'exécution Colab |
| **Cours CUDA** | [courscuda.pdf](file:///c:/Users/youne/Downloads/cudapso/courscuda.pdf) | Support de cours théorique CUDA |
| **Sujet Master** | [MasterworkcudaPSODE.pdf](file:///c:/Users/youne/Downloads/cudapso/MasterworkcudaPSODE.pdf) | Cahier des charges du projet de Master |
| **Papier Référence**| [An_Improved_CUDA-Based-Implementation...pdf](file:///c:/Users/youne/Downloads/cudapso/An_Improved_CUDA-Based-Implementation_of_Different.pdf) | Article de recherche de référence pour l'optimisation DE sur GPU |
