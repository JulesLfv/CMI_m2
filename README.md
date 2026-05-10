# Prédiction de trajectoires d'ouragans (Atlantic 1851–2017)

Projet ML/DL reproductible pour prédire la prochaine position `(latitude, longitude)` d'un ouragan à partir de son historique.

## Objectifs
- Construire un pipeline complet: QA data, preprocessing, fenêtrage, entraînement, évaluation.
- Comparer baselines et modèles deep learning.
- Étudier l'impact de la taille de fenêtre et du feature engineering.

## Arborescence
```text
data/
notebooks/
src/
  preprocessing.py
  features.py
  datasets.py
  models.py
  train.py
  evaluate.py
  utils.py
configs/
  config.yaml
results/
  models/
  metrics/
  figures/
requirements.txt
README.md
```

## Dataset
Fichier attendu: `data/Atlantic Hurricane Tracks 1851 to 2017.csv`
Colonnes:
- `Date`
- `Storm_ID`
- `Observation_Latitude`
- `Observation_Longitude`

## Choix méthodologiques
- **Split principal choisi**: split par `Storm_ID` (anti-fuite inter-trajectoires).
- Fenêtres testées: `3, 5, 8, 12`.
- Cible par défaut: prédiction directe `[lat_t+1, lon_t+1]`.
- Features:
  - mode `basic`: lat/lon passées.
  - mode `enriched`: deltas, vitesse approx., cap, delta temps, saisonnalité cyclique.

## Modèles comparés
### Baselines
- Persistence
- Mean displacement
- MLP (fenêtre aplatie)

### Modèles séquentiels
- RNN simple
- LSTM
- GRU
- TCN (CNN 1D temporelle)

## Métriques
- MAE lat/lon
- RMSE lat/lon
- Erreur euclidienne moyenne (degrés)
- Distance haversine moyenne (km)
- Temps d'entraînement / inférence
- Nombre de paramètres

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution
```bash
python src/train.py --config configs/config.yaml
```

Sorties:
- `results/metrics/summary_metrics.csv`
- `results/metrics/data_quality.json`
- `results/models/*.pt`
- `results/figures/loss_*.png`

## Expérimentations recommandées
1. `features_mode: basic` puis `enriched`.
2. `target_mode: direct` puis extension delta.
3. Variation `hidden_size`, `num_layers`, `dropout`, `batch_size`, `learning_rate`.
4. Multi-runs via `runs_per_model` (extension possible).

## Limites actuelles
- Pas de carte géospatiale dédiée (simplement plots de loss à ce stade).
- Pas encore de stratégie chrono stricte (option à ajouter si besoin de simulation forecasting opérationnel).
- Pas encore de recherche hyperparamétrique automatisée (grid/random à brancher).

## Extensions suggérées
- Ajout d'un Transformer léger.
- Prédiction multi-horizon.
- Calibration incertitude (ensembles, quantiles).
- Validation chronologique backtesting.
