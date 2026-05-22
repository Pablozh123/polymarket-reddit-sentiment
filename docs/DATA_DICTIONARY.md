# Data Dictionary

Dieses Data Dictionary beschreibt die finalen CSV-Outputs und Audit-Artefakte. Es gehört in den Anhang bzw. ins Repository, damit die Datenharmonisierung ohne Code nachvollziehbar bleibt.

## `data/correlation_pairs_bulk.csv`

Aggregierte Marktebene für die Hauptkorrelation.

| Spalte | Bedeutung |
|---|---|
| `market_rank` | Reihenfolge des Marktes im Run. |
| `api_source` | `polymarket_live` oder `demo_fallback`. |
| `is_demo_market` | True, wenn der Markt aus dem Demo-Fallback stammt. |
| `collected_at_utc` | Zeitpunkt der Datenerhebung in UTC. |
| `market_id` | Polymarket condition ID oder Demo-ID. |
| `clob_token_id` | Polymarket CLOB Asset ID; für Preis-Historie relevant. |
| `market_url` | Polymarket-Webseite zum Markt. |
| `question` | Marktfrage. |
| `probability` | Aktuelle Yes-Wahrscheinlichkeit als Dezimalwert 0-1. |
| `category` | API-Kategorie oder inferierte Taxonomie. |
| `volume` | Polymarket-Volumen, falls verfügbar. |
| `end_date` | Markt-Enddatum, falls verfügbar. |
| `reddit_query` / `keywords` | Aus der Marktfrage extrahierte Reddit-Suchbegriffe. |
| `subreddits` | Kommagetrennte Liste der abgefragten Subreddits. |
| `sentiment_model` | Verwendetes Sentiment-Modell, z.B. `twitter-roberta`. |
| `semantic_threshold` | Schwellenwert für semantischen Filter; im Bulk-Run leer/NaN. |
| `mean_compound` | Durchschnittlicher Sentiment-Score aller Posts zum Markt. |
| `weighted_compound` | Sentiment gewichtet mit `log1p(score)`. |
| `adjusted_compound` | `mean_compound` nach Polarity-Korrektur. |
| `adjusted_weighted` | `weighted_compound` nach Polarity-Korrektur. |
| `polarity` | +1 für positiv gerahmte, -1 für negativ gerahmte Fragen. |
| `stance_score` | Stance-Score nach `scripts/add_stance_scores.py`; NaN nur, wenn dieser Add-on-Schritt nicht ausgeführt wurde. |
| `n_posts` | Anzahl Posts nach finalem Filter. |
| `n_comments` | Anzahl Kommentare im Run. Im Bulk-Run 0. |
| `n_total` | Posts plus Kommentare nach finalem Filter. |
| `n_raw_posts` | Ursprüngliche Anzahl gefundener Reddit-Zeilen. |
| `n_after_semantic_filter` | Anzahl nach semantischem Filter. Im finalen Bulk-Run wurde kein semantischer Filter angewendet; die Spalte bleibt aus Schema-Gründen gleich `n_total`. |

## `data/posts_per_market.csv`

Postebene für Subreddit-, Zeit- und Qualitätsanalysen.

| Spalte | Bedeutung |
|---|---|
| `market_id`, `clob_token_id`, `market_question`, `market_url` | Bezug zum Polymarket-Markt. |
| `probability`, `category`, `api_source`, `collected_at_utc` | Markt- und Run-Metadaten. |
| `reddit_query` | Suchquery, die den Post gefunden hat. |
| `sentiment_model`, `semantic_threshold` | Modell- und Filterkonfiguration. |
| `n_raw_posts_for_market`, `n_after_filter_for_market` | Marktweite Filterbilanz. |
| `post_id` | Reddit-ID des Posts oder Kommentar-Pseudo-ID. |
| `content_type` | `post` oder `comment`. |
| `title` | Reddit-Post-Titel. |
| `text` | Reddit-Post-Text oder Kommentartext. |
| `text_for_sentiment` | Konkatenierter und whitespace-bereinigter Text für Sentiment. Wird aus `title + " " + text` gebildet. |
| `subreddit` | Herkunfts-Subreddit. |
| `score` | Reddit-Score/Upvotes. |
| `num_comments` | Kommentarzahl beim Post. |
| `compound` | Sentiment-Score -1 bis +1. |
| `stance_score` | Post-level Stance Score nach `scripts/add_stance_scores.py`; NaN nur vor Ausführung dieses Add-on-Schritts. |
| `sentiment_label` | `positive`, `neutral` oder `negative`. |
| `created_utc` | Reddit-Zeitstempel. |
| `url` | Reddit-Link. |

## `data/correlation_pairs.csv`

Detail-Run mit Kommentaren und semantischer Filterung. Schema entspricht `correlation_pairs_bulk.csv`, aber `semantic_threshold` ist gesetzt und `n_comments` kann grösser als 0 sein.

## `data/posts_per_market_detail.csv`

Post-/Kommentar-Ebene des Detail-Runs. Schema entspricht `posts_per_market.csv`; zusätzlich kann `semantic_score` enthalten sein.

## `reports/relevance_audit_sample.csv`

Manuell codierte Qualitätsstichprobe der Reddit-Treffer. Die Stichprobe umfasst 50 reproduzierbar gezogene Zeilen aus `data/posts_per_market.csv` (`random_state=26`).

| Spalte | Bedeutung |
|---|---|
| `sample_id` | Laufende Nummer der Audit-Stichprobe. |
| `audit_label` | Manuelle Relevanzklasse: `relevant`, `teilweise relevant` oder `irrelevant`. |
| `audit_note` | Kurze Begründung der Codierung. |
| `market_id`, `market_question`, `reddit_query` | Bezug zur Polymarket-Frage und Suchquery. |
| `post_id`, `subreddit`, `title`, `excerpt`, `url` | Reddit-Treffer, der manuell beurteilt wurde. |

## `reports/semantic_filter_comparison.csv`

Vergleich zwischen reiner Keyword-Suche und einem simulierten semantischen Filter auf fünf Beispielmärkten. Der finale Bulk-Datensatz bleibt unverändert; diese Datei dient als Qualitätsanalyse.

| Spalte | Bedeutung |
|---|---|
| `market_question` | Polymarket-Frage. |
| `reddit_query` | Keyword-Query aus der Marktfrage. |
| `raw_posts` | Anzahl Posts ohne semantischen Filter. |
| `retained_posts_threshold_0_20` | Anzahl Posts mit `semantic_score >= 0.20`. |
| `retention_pct` | Anteil der Posts, die der semantische Filter behalten würde. |
| `mean_semantic_score` | Durchschnittliche semantische Ähnlichkeit zwischen Marktfrage und Reddit-Texten. |
| `median_semantic_score` | Median der semantischen Ähnlichkeit. |
| `example_removed_title` | Beispiel für einen Treffer, der vom Filter entfernt würde. |
| `example_removed_score` | Semantischer Score dieses entfernten Beispiels. |

## `reports/validation_checks.md`

Automatisch erzeugte Validierung der finalen CSVs. Enthält unter anderem Mindestanzahl Märkte, Demo-Fallback-Prüfung, Missing-Value-Checks, Wertebereichsprüfung für `probability`, Duplikatcheck auf `(market_id, post_id)`, Stance-Vollständigkeit und den Abgleich der gespeicherten Reddit-Queries mit der aktuellen Keyword-Logik.

## `reports/FINAL_REPORT.pdf`

Finaler Abgabebericht. Die wichtigsten Plots und Tabellen sind direkt enthalten, damit der Bericht ohne Code verständlich ist.
