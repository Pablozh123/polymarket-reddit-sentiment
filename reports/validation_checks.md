# Validierungschecks fuer finale CSVs

| Check | Ergebnis | Status |
|---|---:|---|
| Mindestens 25 auswertbare Maerkte | 29 | OK |
| Keine Demo-Fallback-Maerkte im Hauptergebnis | 0 | OK |
| Keine fehlenden Polymarket-Wahrscheinlichkeiten | 0 | OK |
| Wahrscheinlichkeiten im Wertebereich [0, 1] | 0.0015 bis 0.7720 | OK |
| Keine doppelten Markt-Post-Zeilen | 0 | OK |
| Keine fehlenden `text_for_sentiment`-Werte | 0 | OK |
| Keine fehlenden Markt-Stance-Scores | 0 | OK |
| Keine fehlenden Post-Stance-Scores | 0 | OK |
| Reddit-Queries passen zur aktuellen Keyword-Logik | 0 | OK |
| Semantischer Filter im finalen Bulk-Run | nein | OK |

Diese Checks pruefen zentrale Anforderungen aus Datenbereinigung, Qualitaetspruefung und Reproduzierbarkeit. Sie ersetzen keine inhaltliche Interpretation, zeigen aber, dass die finalen CSVs konsistent und abgabefaehig sind.
