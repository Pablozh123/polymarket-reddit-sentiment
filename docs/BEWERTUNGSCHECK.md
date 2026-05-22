# Bewertungscheck DWaE FS26

Diese Checkliste uebersetzt die Bewertungskriterien in konkrete Abgabe-Artefakte.

| Bereich | Punkte | Nachweis im Projekt | Status vor Abgabe |
|---|---:|---|---|
| A Konzept, Fragestellung & Datenauswahl | 15 | Klare Forschungsfragen F1-F4, Motivation Prediction Markets vs. Reddit, zwei heterogene Datenquellen. | Erfuellt in `reports/FINAL_REPORT.pdf`. |
| B Datenbereinigung & Qualitaetspruefung | 20 | Missing Values, Duplikate, Ausreisser, API-Ausfaelle, Rate Limits, Demo-Fallback-Markierung, Bulk-Filterstatus. | Erfuellt mit Datenqualitaetsfigur und Text. |
| C Datentransformation & Harmonisierung | 15 | JSON zu DataFrame/CSV, Keyword-Mapping, Kategorie-Taxonomie, Sentiment-Skalen, Polarity-Korrektur, Stance, `clob_token_id`. | Erfuellt, Data Dictionary aktuell. |
| D Analyse & Erkenntnisse | 23 | Pearson/Spearman, Kategorievergleich, Kruskal-Wallis, Zeitmuster, Stance-Vergleich. | Erfuellt mit finalen p-Werten und Effektstaerken. |
| E Visualisierungen | 10 | Pipeline, Datenqualitaet, Scatter, Kategorieplot, Boxplot, Zeitreihe, Stance-Vergleich. | Erfuellt in `reports/figures/`. |
| F Werkzeugwahl & technische Umsetzung | 7 | Python, pandas, APIs, NLP-Modelle, Streamlit, reproduzierbare Scripts. | Erfuellt, README-Workflow aktualisiert. |
| G Bericht & Kommunikation | 10 | Bericht entlang A-G, Limitationen, Interpretation ohne Code lesbar. | Erfuellt in `reports/FINAL_REPORT.pdf`. |

## Finaler Abgabe-Workflow

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python run_bulk.py
python scripts/add_stance_scores.py
python scripts/compare_sentiment_models.py
python scripts/generate_report_assets.py
python scripts/validate_outputs.py
python scripts/render_final_report.py
```

Optional fuer tieferen Run mit Kommentaren und semantischem Filter:

```bash
python run_analysis.py
python scripts/generate_report_assets.py
```

Stance Detection fuer den finalen Bulk-Datensatz:

```bash
python scripts/add_stance_scores.py
```

## Akzeptanz vor Moodle-Abgabe

- `data/correlation_pairs_bulk.csv` enthaelt mindestens 25 auswertbare Maerkte.
- `is_demo_market` ist im finalen Hauptergebnis ueberall `False` oder Demo-Fallback wird klar als technische Ersatzdaten deklariert.
- `data/posts_per_market.csv` enthaelt genug Posts pro Markt fuer eine nachvollziehbare Aggregation.
- `reports/report_summary.md` enthaelt Pearson/Spearman-Werte.
- `reports/validation_checks.md` enthaelt die finalen CSV-Qualitaetschecks.
- `data/correlation_pairs_bulk.csv` und `data/posts_per_market.csv` enthalten `stance_score`.
- Alle Figuren in `reports/figures/` sind im Bericht enthalten.
- Bericht nennt Limitationen und vermeidet kausale Aussagen.
