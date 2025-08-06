# Apprentissage par Renforcement sur les Diatomées Bacillaria

## Description

Ce projet implémente des algorithmes d'apprentissage par renforcement (Q-learning et PPO) pour contrôler la locomotion de colonies de diatomées Bacillaria. Le système utilise une simulation multi-corps pour modéliser les interactions hydrodynamiques entre les composants de la colonie.

## Auteur

**Ely Cheikh Abass** - Juin-Juillet 2025

## Structure du Projet

```
RL_diatoms/
├── q_learning/           # Algorithmes Q-learning
│   ├── train.py         # Entraînement Q-learning
│   ├── validate.py      # Validation des politiques
│   ├── main.py          # Point d'entrée Q-learning
│   ├── hyperparams_tunning.py  # Optimisation hyperparamètres
│   └── config.yaml      # Configuration Q-learning
├── ppo/                 # Algorithmes PPO
│   ├── train.py         # Entraînement PPO
│   ├── hyperparams_tunning.py  # Optimisation hyperparamètres
│   └── config.yaml      # Configuration PPO
├── common/              # Modules partagés
│   ├── sim_env.py       # Environnement de simulation
│   ├── utils.py         # Utilitaires et configuration
│   ├── multi_bodies_bacillaria1.py  # Simulation physique
│   ├── multi_bodies_utilities1.py   # Utilitaires simulation
│   ├── post_process_imagery.py      # Visualisation résultats
│   └── copy_sim_folders_remote.py   # Gestion fichiers distants
├── requirements.txt     # Dépendances Python
├── results.zip         # Archive des résultats (538MB)
└── README.md           # Ce fichier
```

## Fonctionnalités Principales

### Algorithmes d'Apprentissage
- **Q-learning** : Algorithme tabulaire pour l'apprentissage par renforcement
- **PPO (Proximal Policy Optimization)** : Algorithme de politique continue

### Méthodes de Récompense
- `VELOCITY` : Récompense basée sur la vitesse du centre de masse
- `CM_DISTANCE` : Récompense basée sur la distance parcourue
- `X_DISPLACEMENT` : Récompense basée sur le déplacement en X
- `CIRCULAR_ZONES` : Récompense basée sur les zones circulaires
- `FORWARD_PROGRESS` : Récompense basée sur la progression directionnelle

### Environnement de Simulation
- Interface Gymnasium pour l'apprentissage par renforcement
- Simulation multi-corps avec interactions hydrodynamiques
- Modélisation des contraintes entre les bâtonnets
- Gestion des états de la colonie (gaps entre bâtonnets)

## Installation

### Prérequis
- Python 3.8+
- CUDA (optionnel, pour accélération GPU)

### Installation des Dépendances

```bash
pip install -r requirements.txt
```

### Dépendances Principales
- `gymnasium==1.1.1` : Environnements d'apprentissage par renforcement
- `stable_baselines3==2.6.0` : Implémentation PPO
- `numpy==1.26.4` : Calculs numériques
- `scipy==1.11.4` : Calculs scientifiques
- `matplotlib==3.10.0` : Visualisation
- `PyYAML==6.0.2` : Configuration YAML
- `torch==2.7.1` : Deep learning (pour PPO)

## Utilisation

### Configuration

Les paramètres sont définis dans les fichiers `config.yaml` :

```yaml
input_file_path: "./inputfile_bacillaria.dat"
output_directory: "./"
nb_blobs: 10          # Nombre de blobs par bâtonnet
nb_rods: 9            # Nombre de bâtonnets
dt: 0.0025           # Pas de temps
nb_step: 200         # Nombre d'étapes par épisode
nb_episodes: 2000    # Nombre d'épisodes d'entraînement
learning_rate: 0.1   # Taux d'apprentissage
discount_factor: 0.6 # Facteur de décote
reward_method: "FORWARD_PROGRESS"  # Méthode de récompense
reward_angle: 90.0   # Angle pour FORWARD_PROGRESS
```

### Exécution Simplifiée (Recommandée)

Depuis la racine du projet :

```bash
# Entraînement Q-learning
python3 -m q_learning.main --config q_learning/config.yaml

# Entraînement PPO
python3 -m ppo.main --config ppo/config.yaml
```

### Exécution Directe (Alternative)

```bash
# Q-learning
cd q_learning
python train.py --input_file_path "./inputfile_bacillaria.dat" \
                --output_directory "./results" \
                --nb_blobs 10 \
                --nb_rods 9 \
                --nb_episodes 1000 \
                --reward_method "FORWARD_PROGRESS"

# PPO
cd ppo
python train.py --input_file_path "./inputfile_bacillaria.dat" \
                --output_directory "./results" \
                --nb_blobs 10 \
                --nb_rods 9 \
                --nb_episodes 1000 \
                --reward_method "FORWARD_PROGRESS"
```

### Validation

```bash
cd q_learning
python validate.py --config config.yaml --qtable q_table.pkl --episodes 5
```

### Optimisation des Hyperparamètres

```bash
cd q_learning
python hyperparams_tunning.py
```

## Architecture Technique

### Environnement de Simulation (`sim_env.py`)

L'environnement `DiatomEnv` implémente l'interface Gymnasium et gère :

- **État** : Gaps entre les bâtonnets de la colonie
- **Actions** : Déplacement d'un bâtonnet dans une direction
- **Récompenses** : Basées sur le mouvement du centre de masse
- **Simulation** : Intégration avec le moteur physique multi-corps

### Simulation Multi-corps (`multi_bodies_bacillaria1.py`)

- Calcul des interactions hydrodynamiques
- Gestion des contraintes entre bâtonnets
- Intégration temporelle avec schémas numériques
- Support pour différents types de mobilité (Python, C++, CUDA)

### Méthodes de Récompense

1. **VELOCITY** : `||v_cm||` - Norme de la vitesse du centre de masse
2. **CM_DISTANCE** : `||Δr_cm||` - Distance parcourue par le centre de masse
3. **X_DISPLACEMENT** : `Δx_cm` - Déplacement en direction X
4. **CIRCULAR_ZONES** : Distance depuis la position initiale
5. **FORWARD_PROGRESS** : Progression dans une direction spécifiée

## Résultats

Les résultats d'entraînement sont sauvegardés dans les dossiers spécifiés par `output_directory` dans les fichiers de configuration. Les Q-tables et modèles PPO sont sauvegardés avec des noms structurés incluant les paramètres d'entraînement.

Un fichier `results.zip` (538MB) est également disponible contenant les résultats d'analyses précédentes.

## Visualisation

Le module `post_process_imagery.py` permet de :
- Générer des graphiques de trajectoires
- Visualiser le centre de masse
- Analyser les performances des politiques apprises

## Notes de Développement

- **Date de création** : 9 juin 2025 (Q-learning, environnement)
- **Date PPO** : 31 juillet 2025
- **Inspiration** : Code de Julien/Stefanie pour la structure de base
- **Simulation** : Basée sur le framework RigidMultiblobsWall

## Licence

Projet académique - Tous droits réservés à l'auteur.

## Contact

**Ely Cheikh Abass** - Projet d'apprentissage par renforcement sur les diatomées 