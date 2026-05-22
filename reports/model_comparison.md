# Sentiment-Modellvergleich

Kein Modell kann hier als objektiv 'bestes' Modell bewiesen werden, weil keine manuell gelabelten Reddit-Sentiment-Labels vorliegen. Der Vergleich nutzt deshalb operative Kriterien: Korrelation mit Polymarket, Richtungstrefferquote und Laufzeit.

| Modell | Maerkte | Pearson r | Spearman rho | Richtungstrefferquote | Laufzeit s |
|---|---:|---:|---:|---:|---:|
| vader | 29 | +0.0690 | +0.1146 | 27.6% | 13.4 |
| twitter-roberta | 29 | +0.0791 | +0.1508 | 44.8% | 0.0 |

Interpretation: VADER ist schnell und transparent, aber lexikonbasiert. FinBERT ist fuer Finanztexte trainiert und kann bei Reddit-Slang oder Sport-/Legal-Maerkten weniger passend sein. Twitter-RoBERTa ist fuer Social-Media-Sprache trainiert und wurde deshalb fuer den finalen Run gewaehlt.
