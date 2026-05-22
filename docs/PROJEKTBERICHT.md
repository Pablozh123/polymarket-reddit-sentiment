# Projektbericht: Polymarket Reddit Sentiment

**Kurs:** Data Wrangling & Engineering FS26  
**Projekt:** Reddit-Sentiment zu Polymarket-Vorhersagemaerkten  
**Ziel:** Explorativ pruefen, ob Reddit-Stimmung mit Polymarket-Wahrscheinlichkeiten zusammenhaengt.

> Abgabehinweis: Diese Datei ist die Berichtsvorlage. Der aktuelle finale Abgabebericht ist `reports/FINAL_REPORT.pdf` bzw. `reports/FINAL_REPORT.md`.

## 1. Konzept, Fragestellung und Datenauswahl

### Motivation
Prediction Markets wie Polymarket aggregieren Markterwartungen als Preise zwischen 0 und 1. Reddit enthaelt oeffentliche Diskussionen, Meinungen und Stimmungen zu denselben Themen. Das Projekt untersucht, ob diese beiden Signale auf explorativer Ebene zusammenpassen.

### Forschungsfragen
1. **F1:** Korrelieren Reddit-Sentiment-Scores mit Polymarket-Wahrscheinlichkeiten?
2. **F1b:** Unterscheidet sich diese Korrelation nach Marktkategorie?
3. **F2:** Unterscheidet sich das Sentiment signifikant zwischen Subreddits?
4. **F3:** Gibt es zeitliche Muster im Reddit-Sentiment?
5. **F4:** Erklaert Stance Detection die Marktwahrscheinlichkeit besser als reines Sentiment?

### Datenquellen
- **Polymarket Gamma/CLOB API:** Marktfrage, aktuelle Wahrscheinlichkeit, Volumen, Enddatum, Markt-URL, `clob_token_id`.
- **Reddit Public JSON API oder PRAW:** Posts und optional Kommentare aus thematisch passenden Subreddits.
- **Generierte CSVs:** `data/correlation_pairs_bulk.csv`, `data/posts_per_market.csv`, optional `data/correlation_pairs.csv` und `data/posts_per_market_detail.csv`.

### Finale Stichprobe
Der finale Bulk-Run vom 2026-05-22 enthaelt 29 auswertbare Live-Polymarket-Maerkte und 725 Reddit-Posts. Die Live-Polymarket-API lieferte 100 aktive Maerkte; aus 30 ausgewaehlten relevanten Maerkten wurde ein Markt ausgeschlossen, weil keine ausreichenden Reddit-Posts gefunden wurden. Demo-Fallback-Daten wurden nicht fuer die finalen Hauptergebnisse verwendet (`is_demo_market=False` fuer alle finalen Maerkte).

## 2. Datenbeschaffung und Pipeline

![Pipeline](../reports/figures/pipeline_diagram.png)

Die Pipeline folgt dem ETL-Prinzip:

1. **Extract:** Polymarket-Maerkte per API laden; pro Markt Keywords aus der Frage extrahieren; Reddit-Posts suchen.
2. **Transform:** JSON-Antworten in DataFrames ueberfuehren; Datentypen harmonisieren; Sentiment, Polarity-Korrektur und Stance berechnen.
3. **Load:** Aggregierte Markt-Paare und Post-Level-Daten als CSV speichern.

Wichtige technische Entscheidungen:
- Reddit wird ueber Public JSON oder PRAW abgefragt. PRAW ist optional und verbessert Rate Limits.
- Polymarket-Preis-Historie verwendet `clob_token_id`, weil die CLOB-API eine Asset ID erwartet.
- `collected_at_utc`, `api_source`, `market_id`, `market_url` und `clob_token_id` werden gespeichert, damit der Run reproduzierbar dokumentiert ist.

## 3. Datenbereinigung und Qualitaetspruefung

![Datenqualitaet](../reports/figures/data_quality_summary.png)

### Fehlende Werte
- Leere Reddit-Textfelder entstehen oft bei Link-Posts und werden als leerer String behandelt.
- Fehlende Polymarket-Wahrscheinlichkeiten werden nicht imputiert, sondern aus der finalen Korrelation ausgeschlossen.
- Fehlende Kategorien werden mit einer dokumentierten Keyword-Taxonomie inferiert.

### Duplikate
- Reddit-Posts werden ueber `post_id` geprueft.
- Inhaltliche Duplikate durch Crossposts werden im Bericht als Limitation genannt, wenn sie nicht vollstaendig entfernt werden.

### Ausreisser
- Reddit-Scores und Kommentarzahlen sind typischerweise stark rechtsschief.
- Fuer Analyseplots werden log-Gewichte (`log1p(score)`) genutzt, damit virale Posts nicht die gesamte Metrik dominieren.
- In den Notebooks wird Winsorisierung als alternative Bereinigungsstrategie dokumentiert.

### Qualitaetskriterien
- Mindestens 25-30 Maerkte im finalen Bulk-Run; final erreicht: 29 auswertbare Maerkte.
- Idealerweise mindestens 10 relevante Reddit-Beitraege pro Markt.
- Keine finalen Hauptergebnisse aus unmarkierten Demo-Daten.
- Dokumentation von Rate Limits, API-Ausfaellen und Filterquote; im finalen Bulk-Run wurde kein semantischer Filter angewendet.
- Relevanz-Audit einer 50er-Stichprobe der Reddit-Treffer; final erreicht: 0 direkt relevante, 13 teilweise relevante und 37 irrelevante Treffer.

## 4. Datentransformation und Harmonisierung

### Harmonisierung der Datenquellen
Polymarket liefert strukturierte Marktdaten, Reddit liefert unstrukturierte Texte. Die Harmonisierung erfolgt ueber die Marktfrage:

1. Marktfrage wird in eine kompakte Reddit-Suchanfrage transformiert. Dabei werden schwache Fragewoerter entfernt, Zahlen/Jahre und kurze Kontexttokens wie `FIFA`, `NHL`, `GTA`, `VI`, `win` oder `Cup` aber erhalten. Markt-Benchmarks wie `before GTA VI` werden nur entfernt, wenn sie fuer Reddit als Suchbegriff eher stoeren und der Markt nicht selbst ueber GTA VI handelt.
2. Reddit-Posts werden pro Markt gesammelt.
3. Im Detail-Run kann `sentence-transformers` irrelevante Posts per semantischer Aehnlichkeit filtern; im finalen Bulk-Run wurde dieser Schritt aus Laufzeit- und Stabilitaetsgruenden nicht angewendet.
4. Sentiment wird pro Post berechnet und pro Markt aggregiert.
5. Markt-Wahrscheinlichkeit und aggregiertes Sentiment bilden ein Analysepaar.

### Kategorie-Taxonomie
Falls Polymarket keine Kategorie liefert, wird die Kategorie mit `src/market_metadata.py` aus Keywords inferiert. Die finale Stichprobe enthaelt Sports, Legal, Geopolitics, Entertainment, Politics, Crypto und Other.

### Sentiment-Skalen
Das Projekt unterstuetzt VADER, FinBERT und Twitter-RoBERTa. Fuer Reddit ist Twitter-RoBERTa der Standard, weil es auf Social-Media-Sprache trainiert ist. Der Score wird als `compound` zwischen -1 und +1 gespeichert.

### Polarity-Korrektur
Bei negativ gerahmten Fragen bedeutet positives Sentiment oft, dass das negative Ereignis weniger wahrscheinlich wird. Beispiel: "Will there be a recession?" Bei Optimismus wird das Vorzeichen invertiert. Deshalb werden `adjusted_compound` und `adjusted_weighted` berechnet.

### Stance Detection
Stance Detection prueft, ob ein Text die Aussage "Ereignis wird eintreten" eher stuetzt oder ablehnt. Das ist methodisch passender als reines Sentiment, weil es die Richtung der Marktfrage direkter abbildet.

## 5. Analyse und Erkenntnisse

### F1: Korrelation Sentiment und Polymarket
![Korrelation](../reports/figures/correlation_scatter.png)

Zu berichten:
- Pearson r und p-Wert fuer lineare Zusammenhaenge.
- Spearman rho und p-Wert als robuste Rangkorrelation.
- Unterschied zwischen `mean_compound`, `weighted_compound` und polarity-adjustierten Metriken.
- Ergebnis nur als explorative Korrelation interpretieren, nicht kausal.

### F1b: Kategorieanalyse
![Kategorie-Korrelation](../reports/figures/category_correlation.png)

Zu berichten:
- Kategorien mit ausreichendem n separat zeigen.
- Kleine Kategoriegruppen als explorativ markieren.
- Keine Ueberinterpretation einzelner Kategorien bei n < 5.

### F2: Subreddit-Unterschiede
![Subreddit Boxplot](../reports/figures/subreddit_boxplot.png)

Zu berichten:
- Deskriptive Mittelwerte/Mediane pro Subreddit.
- Kruskal-Wallis-Test, falls mindestens zwei Subreddits mit genuegend Daten vorhanden sind.
- Inhaltliche Interpretation: Finanz-Subreddits, Politik-Subreddits und News-Subreddits haben unterschiedliche Diskussionskulturen.

### F3: Zeitliche Muster
![Zeitreihe](../reports/figures/sentiment_timeline.png)

Zu berichten:
- Zeitraum der Reddit-Daten.
- Tagesaggregation nach Kategorie.
- Wochentagseffekte nur vorsichtig interpretieren.
- Echte Lag-Analyse braucht wiederholte Polymarket-Snapshots.

### F4: Stance statt Sentiment
Im finalen Run wurde Stance Detection mit `scripts/add_stance_scores.py` berechnet. Zu berichten:
- Stance r vs. Sentiment r vergleichen.
- Vorteil: Stance ist naeher an der Marktfrage.
- Nachteil: deutlich laengere Laufzeit und Modellabhaengigkeit.

## 6. Visualisierung

Die finalen Visualisierungen muessen fuer Leserinnen und Leser ohne Code verstaendlich sein. Jede Abbildung sollte enthalten:
- sprechenden Titel,
- Achsenbeschriftung,
- Stichprobengroesse `n`,
- Modellangabe,
- kurze Interpretation direkt im Bericht.

Pflichtfiguren:
- Pipeline-Diagramm,
- Datenqualitaetsuebersicht,
- Sentiment-vs.-Polymarket-Scatter,
- Kategorie-Korrelation,
- Subreddit-Boxplot,
- Sentiment-Zeitreihe.

## 7. Werkzeugwahl und technische Umsetzung

Die Loesung nutzt Python, pandas, requests, scipy, matplotlib/Plotly, Streamlit sowie optionale NLP-Modelle. Python ist hier passend, weil APIs, Textdaten, DataFrames, Statistik und Visualisierung in einem reproduzierbaren Workflow verbunden werden koennen. Low-Code-Tools waeren fuer diese flexible API- und NLP-Pipeline weniger geeignet.

Die technische Umsetzung ist modular:
- `src/polymarket.py`: Polymarket API und Preis-Historie.
- `src/reddit.py`: Reddit-Datenbeschaffung.
- `src/sentiment.py`: Sentiment, Semantic Filter und Stance Detection.
- `src/market_metadata.py`: gemeinsame Taxonomie, Polarity und stabile Output-Spalten.
- `run_bulk.py`: reproduzierbarer Hauptlauf.
- `scripts/add_stance_scores.py`: Stance Scores fuer den finalen Datensatz.
- `scripts/compare_sentiment_models.py`: Vergleich von VADER und Twitter-RoBERTa, optional FinBERT.
- `scripts/generate_report_assets.py`: druckreife Berichtsfiguren.

## 8. Limitationen

- Reddit ist kein repraesentatives Abbild der Gesamtbevoelkerung.
- Polymarket-Preise sind Marktpreise, keine objektiven Wahrheiten.
- Suchbegriffe koennen relevante Posts verpassen oder irrelevante Posts einschliessen.
- Sentiment ist nicht identisch mit Zustimmung zur Marktfrage.
- Kleine Kategoriegruppen liefern nur explorative Resultate.
- Ohne wiederholte Polymarket-Snapshots ist keine robuste Lag- oder Lead-Analyse moeglich.

## 9. Fazit

Das Projekt demonstriert die zentralen Inhalte des Moduls: API-Datenbeschaffung, Data Cleaning, Transformation, Harmonisierung heterogener Quellen, Text-Wrangling, Pipeline-Denken, Statistik und Visualisierung. Fuer die Note ist entscheidend, dass die finalen Ergebnisse transparent aus einer reproduzierbaren Pipeline stammen und im Bericht kritisch interpretiert werden.
