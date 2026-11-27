# project-data-science-use-case

`classification.py` analysiert Einkäufe mit **Supervised Learning** (Klassifikation der Käuferklassen `MEN` und `WOMAN`) und/oder **Unsupervised Learning** (Clustering ähnlicher Warenkörbe). Produkte werden anhand der `receipt_id` zu Einkäufen zusammengefasst. Optional wird der Einkaufsort als Merkmal berücksichtigt.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Verwendung

```bash
# Supervised Learning
python classification.py --labelled labelled_data.csv --supervised-model svm

# Unsupervised Learning
python classification.py --unlabelled unlabelled_data.csv --unsupervised-model gmm

# Unsupervised & Supervised, jeweils mit und ohne Filialadresse
python classification.py --labelled labelled_data.csv --unlabelled unlabelled_data.csv --supervised-model logistic-regression --unsupervised-model agglomerative --address both --output analyse

# Unsupervised & Supervised, jeweils mit und ohne Filialadresse (SVM & agglomerative)
python classification.py --labelled labelled_data.csv --unlabelled unlabelled_data.csv --supervised-model svm --unsupervised-model agglomerative --address both --clusters 4 --fixed-clusters --output results_folder
```

Mindestens eine der Optionen `--labelled` oder `--unlabelled` ist erforderlich. Zum Testen kann sich an den Dateien `data_short_labelled` und `data_short_unlabelled` orientiert werden. Hinweis: Die Daten sind bewusst nicht umfangreich gehalten. Sie dienen nur dazu die Funktionsweise des Skriptes zu evaluieren.

## CLI-Parameter

| Parameter | Bedeutung | Standard |
|---|---|---|
| `--labelled DATEIPFAD` | CSV mit Käuferlabels für Supervised Learning. | – |
| `--unlabelled DATEIPFAD` | CSV ohne Käuferlabels für Unsupervised Learning. | – |
| `--supervised-model MODELL` | Klassifikator: `logistic-regression`, `svm`, `knn`, `decision-tree`. | `logistic-regression` |
| `--unsupervised-model MODELL` | Clustering: `kmeans`, `agglomerative`, `gmm`. | `kmeans` |
| `--address WERT` | Filialadresse als Merkmal: `none` (ohne), `yes` (mit), `both` (beide Varianten getrennt auswerten). | `none` |
| `--clusters N` | Clusteranzahl - ohne `--fixed-clusters` wird sie per Silhouette-Score optimiert (Suchbereich ab 2 bis mindestens 6, ggf. bis `N`, begrenzt durch die Anzahl der Einkäufe). | `2` |
| `--fixed-clusters` | Deaktiviert die automatische Silhouette-Optimierung und verwendet genau `--clusters N`. | deaktiviert |
| `--output ORDNER` | Unterordner innerhalb von `results/` für die Ausgabedateien. | `results/` |
| `--help` | Zeigt die CLI-Hilfe an. | – |

## Eingabedaten

Die CSV-Dateien benötigen die Spalten `receipt_id`, `product_name`, `unit_price`, `quantity`, `date` und `time`. Für `--labelled` ist zusätzlich `buyer` (`MEN` oder `WOMAN`) erforderlich. Für die Adressauswertung können `street`, `zipcode` und `city` angegeben werden. Das Datum wird im Format `TT.MM.JJJJ`, die Uhrzeit als `HH:MM:SS` erwartet.

## Ergebnisse

Im Ausgabeordner entstehen `auswertung.pdf` (Zusammengefasster Report), `uebersicht.csv` (Bewertungsmetriken), PNG-Grafiken und CSV-Dateien mit den Einkauf-Zuordnungen. Supervised Learning verwendet einen stratifizierten 80/20-Train/Test-Split sowie Cross-Validation. Unsupervised Learning bewertet die Cluster unter anderem mit dem Silhouette-Score.
