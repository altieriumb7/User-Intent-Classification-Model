# Analisi stato attuale del progetto e possibili sviluppi futuri

## Executive summary

Il repository è in uno stato **solido come baseline demo/portfolio**: pipeline completa, test automatici verdi, artefatti di valutazione disponibili, demo Streamlit funzionante. Tuttavia, la qualità misurata (metriche perfette) deriva da un dataset sintetico piccolo e bilanciato; quindi il progetto non è ancora pronto per un contesto operativo reale.

---

## 1) Stato attuale (fotografia tecnica)

### Architettura e componenti già presenti

- **Training end-to-end** con split stratificato train/validation/test (`src/train.py`, `src/data.py`).
- **Modello baseline** TF-IDF (word + char n-gram) + Logistic Regression bilanciata (`src/model.py`).
- **Valutazione** con accuracy, macro/micro-F1, classification report e confusion matrix (`src/evaluate.py`).
- **Routing simulato** intent -> team di supporto e misura misrouting (`src/routing.py`).
- **Inference CLI** per singolo testo con top-k e team suggerito (`src/predict_intent.py`).
- **App Streamlit** con metriche, report per classe, confusion matrix e messaggi esempio (`app.py`).
- **Test suite** su preprocessing, train/evaluate, prediction, routing (`tests/`).

### Evidenze misurabili correnti

- Dataset demo: **80 righe**, 8 classi, **10 esempi per classe**, colonna `is_synthetic` presente.
- Report versionati nel repo:
  - accuracy = 1.0
  - macro_f1 = 1.0
  - micro_f1 = 1.0
  - misrouting_rate = 0.0
- Test automatici: **7/7 passati**.

### Valutazione critica (cosa significa davvero)

Le metriche perfette indicano che la pipeline è corretta, ma non validano la robustezza reale del classificatore. Con dati sintetici/bilanciati il rischio principale è sovrastimare la generalizzazione su ticket reali (rumore linguistico, intent ambigui, class imbalance, drift nel tempo).

---

## 2) Gap principali verso produzione

1. **Data realism gap**
   - Manca un dataset reale con etichette affidabili.
2. **Evaluation gap**
   - Non ci sono benchmark out-of-domain, CV multipla, o breakdown temporali.
3. **Decision-quality gap**
   - Non c'è una soglia operativa di confidenza con fallback umano.
4. **MLOps gap**
   - Assenti monitoraggio continuo, allarmi drift, retraining governato da KPI.
5. **Product/API gap**
   - Presente demo UI, ma non endpoint di serving e policy di versioning deployment-ready.

---

## 3) Sviluppi futuri consigliati (prioritizzati)

## Fase A — Dati reali e qualità etichette (priorità massima)

- Creare dataset reale anonimizzato (ticket storici), con policy privacy.
- Definire guideline di annotazione intent + calcolo accordo annotatori.
- Introdurre classi “hard” (multi-intent, frasi brevi, typo, tono emotivo).

## Fase B — Valutazione robusta e criteri di go/no-go

- K-fold cross-validation + holdout temporale.
- Metriche per classe (precision/recall/F1) con focus su classi business-critical.
- Calibration delle probabilità + threshold tuning per copertura/precision target.
- KPI di rilascio (es. macro-F1 minimo, misrouting massimo tollerato, coverage minima).

## Fase C — Evoluzione modello

- Benchmark baseline vs modelli transformer leggeri (DistilBERT/MiniLM).
- Esperimenti cost-sensitive (penalità maggiore su intent/team più critici).
- Routing assistito con top-2/top-3 e regole decisionali business-aware.

## Fase D — MLOps e integrazione prodotto

- API di inferenza versionata (es. `/predict`, `/health`, `/model-info`).
- Dashboard monitoraggio: distribuzione intent, confidenza media, misrouting per team.
- Pipeline retraining periodico con validazione automatica e rollback.
- Feedback loop human-in-the-loop (correzioni agenti -> nuovo training set).

---

## 4) Piano operativo 30-60-90 giorni

- **30 giorni**: dataset reale v1, guideline labeling, nuova baseline addestrata su dati reali.
- **60 giorni**: benchmark con transformer + soglia confidenza + fallback triage umano.
- **90 giorni**: monitoraggio produzione, retraining schedulato, KPI di business in dashboard.

---

## 5) KPI suggeriti (da concordare con il business)

- Macro-F1 su dati reali >= target concordato (es. 0.80+ in fase iniziale).
- Riduzione misrouting rispetto al processo attuale.
- Coverage automatica con confidenza sopra soglia senza degradare precisione classi critiche.
- Riduzione tempo medio di assegnazione ticket e miglioramento SLA primo contatto.

