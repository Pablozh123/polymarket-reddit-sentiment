# Reddit-Trefferqualitaet: Relevanz-Audit und Semantic-Filter-Vergleich

Die Stichprobe umfasst 50 reproduzierbar gezogene Reddit-Treffer aus dem finalen Post-Level-Datensatz (`random_state=26`). Jeder Treffer wurde anhand der Marktfrage in drei Kategorien codiert.

| Kategorie | Anzahl | Anteil | Bedeutung |
|---|---:|---:|---|
| relevant | 0 | 0.0% | Post passt direkt zur Marktfrage. |
| teilweise relevant | 13 | 26.0% | Post passt zum Thema, aber nicht genau zur Frage. |
| irrelevant | 37 | 74.0% | Post enthält Query-Wörter, aber falschen Kontext. |

Interpretation: Die Stichprobe zeigt, dass die Keyword-Suche zwar Abdeckung erzeugt, aber viele Treffer nur oberflächlich passen. Das ist ein klassisches Text-Wrangling-Problem und spricht für zusätzliche Relevanzprüfung.

## Semantic-Filter-Vergleich

Für fünf Beispielmärkte wurde ein `sentence-transformers`-Filter mit Schwelle 0.20 simuliert. Der finale Bulk-Run bleibt unverändert; der Vergleich zeigt nur, wie stark ein semantischer Filter die Trefferzahl reduzieren würde.

| Marktfrage | Ohne Filter | Mit Filter | Retention | Ø Similarity | Beispiel entfernter Treffer |
|---|---:|---:|---:|---:|---|
| Trump out as President before GTA VI? | 25 | 19 | 76.0% | 0.236 | The physical oil market is screaming there is a supply shock yet equities still seem calm? |
| Will China invades Taiwan before GTA VI? | 25 | 25 | 100.0% | 0.500 | The Recent U.S. visit to China and What it means for Investors |
| Will bitcoin hit $1m before GTA VI? | 25 | 24 | 96.0% | 0.302 | Mr Market Round 7, still undervalued but getting harder to explain why |
| Will Harvey Weinstein be sentenced to no prison time? | 25 | 12 | 48.0% | 0.219 | Anyone else notice how ‘quality’ companies can be totally different investments? |
| Will the Oklahoma City Thunder win the 2026 NBA Finals? | 25 | 5 | 20.0% | 0.130 | China Controls 65% of Global Titanium Production and the U.S. Makes None. DD on the Compan |

## Text-Cleaning-Notizen

- `title` und `text` werden zu `text_for_sentiment` kombiniert, damit Link-Posts ohne Body nicht verloren gehen.
- Leere Bodies werden als leerer String behandelt; der Titel bleibt als auswertbarer Text erhalten.
- `[deleted]` und `[removed]` werden bei Kommentaren verworfen; sehr kurze Kommentare unter 10 Zeichen werden nicht aufgenommen.
- URLs und Markdown bleiben im finalen Text weitgehend erhalten. Das ist transparent, kann aber Modell-Scores beeinflussen und wird als Limitation dokumentiert.
- Die eingesetzten Modelle sind englischsprachig bzw. Social-Media-orientiert. Nicht-englische oder stark gemischte Posts können daher ungenauer bewertet werden.
