# Polymarket Reddit Sentiment

**Kurs:** Data Wrangling & Engineering (FHNW)
**Thema:** Stimmungsanalyse von Reddit-Posts korreliert mit Polymarket-Vorhersagemärkten

---

## Was macht dieses Projekt?

- Lädt Polymarket-Märkte (z. B. „Will Bitcoin hit $150k?") via öffentlicher API
- Sucht passende Reddit-Posts (+ Kommentare) per Keyword-Extraktion
- Analysiert das Sentiment (VADER / FinBERT / Twitter-RoBERTa)
- Korreliert Reddit-Stimmung mit Polymarket-Wahrscheinlichkeiten
- Beantwortet 3 Forschungsfragen in Jupyter Notebooks (EDA → Bereinigung → Pipeline → Analyse)
- Streamlit-Dashboard zur interaktiven Exploration

**Kein API-Key erforderlich** – Reddit wird über die öffentliche JSON-API abgefragt.

---

## Voraussetzungen

| Tool | Version |
|------|---------|
| Python | 3.10 oder neuer |
| pip | aktuell |

---

## Setup (Schritt für Schritt)

### 1. Projekt herunterladen / klonen

```bash
# Option A: ZIP herunterladen und entpacken
# Option B: Git
git clone <repo-url>
cd polymarket-reddit-sentiment
```

### 2. Virtuelle Umgebung erstellen und aktivieren

```bash
# Erstellen
python -m venv .venv

# Aktivieren – Windows (CMD)
.venv\Scripts\activate.bat

# Aktivieren – Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Aktivieren – macOS / Linux
source .venv/bin/activate
```

### 3. Pakete installieren

```bash
pip install -r requirements.txt
```

### 4. Jupyter-Kernel registrieren

Damit VS Code / Jupyter das richtige Python (aus der .venv) findet:

```bash
python -m ipykernel install --user --name polymarket --display-name "Python (polymarket)"
```

### 5. Optional: FinBERT / Twitter-RoBERTa (ca. 500 MB)

Nur nötig wenn in den Notebooks `SENTIMENT_MODEL = sentiment.MODEL_FINBERT` gesetzt ist:

```bash
pip install transformers torch
```

---

## Notebooks ausführen (Reihenfolge!)

Öffne VS Code → wähle Kernel `Python (polymarket)` → **Kernel → Restart & Run All**

| Notebook | Inhalt |
|----------|--------|
| `notebooks/01_EDA.ipynb` | Explorative Datenanalyse – Verteilungen, Sentiment, Zeitverlauf |
| `notebooks/02_Datenbereinigung.ipynb` | Missing Values (MCAR/MAR/MNAR), Duplikate, Winsorisierung → speichert `data/reddit_clean.csv` |
| `notebooks/03_Pipeline.ipynb` | `SentimentPipeline`-Klasse, Multi-Topic-Vergleich |
| `notebooks/04_Analyse.ipynb` | 3 Forschungsfragen: Korrelation, Subreddit-Vergleich, Zeitliche Muster |

> **Tipp:** Notebook 02 muss vor 04 laufen, damit `data/reddit_clean.csv` existiert.

---

## Streamlit-App starten

```bash
python -m streamlit run app.py --server.headless=true
```

Öffne dann: [http://localhost:8501](http://localhost:8501)

---

## Projektstruktur

```
polymarket-reddit-sentiment/
├── app.py                    # Streamlit-Dashboard
├── requirements.txt          # Abhängigkeiten
├── src/
│   ├── reddit.py             # Reddit JSON API (kein Account nötig)
│   ├── polymarket.py         # Polymarket API mit Fallback
│   └── sentiment.py          # VADER / FinBERT / RoBERTa
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Datenbereinigung.ipynb
│   ├── 03_Pipeline.ipynb
│   └── 04_Analyse.ipynb
└── data/                     # Wird automatisch erstellt (in .gitignore)
    ├── reddit_clean.csv
    ├── polymarket_clean.csv
    └── correlation_pairs.csv
```

---

## Sentiment-Modell wechseln

In `notebooks/04_Analyse.ipynb` → cell-5:

```python
# Schnell (kein Download, VADER):
SENTIMENT_MODEL = sentiment.MODEL_VADER

# Finanztexte (~440 MB, benötigt transformers):
SENTIMENT_MODEL = sentiment.MODEL_FINBERT

# Social Media (~500 MB, benötigt transformers):
SENTIMENT_MODEL = sentiment.MODEL_ROBERTA
```

---

## Bekannte Einschränkungen

- **Polymarket API** ist von diesem Netz ggf. nicht erreichbar → Demo-Datensatz (40 Märkte) wird automatisch als Fallback verwendet
- **Reddit Rate Limit:** Bei 40 Märkten × 50 Posts ca. 2–5 Minuten Laufzeit (VADER) bzw. 15–30 Min (FinBERT)
- **Plotly** rendert in VS Code Jupyter ohne Interactive-Widgets-Extension nicht → Notebooks nutzen Matplotlib
