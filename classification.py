'''Analyse von Einkaufsdaten mit supervised und unsupervised Methoden.

Beispiel:
    python classification.py --labelled rewe_data_labelled.csv --unlabelled rewe_data.csv \
        --supervised-model svm --unsupervised-model gmm --address both --clusters 2 --output results
'''

# python default libs
import argparse
from pathlib import Path
# sklearn imports
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, calinski_harabasz_score,
    confusion_matrix, davies_bouldin_score, f1_score, matthews_corrcoef,
    precision_recall_fscore_support, precision_score, recall_score,
    roc_auc_score, silhouette_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# numpy and pandas import
import numpy as np
import pandas as pd
# matplotlib imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# use non interactive matplolob backend
matplotlib.use('Agg')


# constants used for configuration in the script
REQUIRED_ROWS_UNLABELLED = {
    'receipt_id', 'product_name', 'unit_price', 'quantity', 'date', 'time'}
REQUIRED_ROWS_LABELLED = {
    'receipt_id', 'product_name', 'unit_price', 'quantity', 'date', 'time', 'buyer'}
LABELS = ['MEN', 'WOMAN']
NUMERIC_FEATURES = ['positions', 'items',
                    'total', 'avg_price', 'hour', 'weekday']
ADDRESS_COLUMNS = ['street', 'zipcode', 'city']
TEST_SIZE = 0.20
# number of cross validation splits
CROSS_VALIDATION_SPLITS = 5
# max cluster size to search for as optimum
MAX_CLUSTER_SEARCH = 6
RANDOM_STATE = 42
MIN_SUPERVISED_RECEIPTS = 20
MIN_RECEIPTS_PER_CLASS = 5
# sets a max value for terms in vocabluray - speeds up training and reduces memory usage
MAX_TFIDF_FEATURES = 3000
OUTPUT_PDF = 'auswertung.pdf'
OUTPUT_SUMMARY = 'uebersicht.csv'
COLOR = '#000000'
OUTPUT_BASE_DIR = Path('results')
CLUSTER_COLORS = ['#DE0016', '#5AF4D2', '#3671C6', '#FF8000']
SUPERVISED_MODEL_NAMES = {
    'logistic-regression': 'Logistic Regression',
    'svm': 'Support Vector Machine (SVM)',
    'knn': 'k-Nearest Neighbors (kNN)',
    'decision-tree': 'Decision Tree',
}
UNSUPERVISED_MODEL_NAMES = {
    'kmeans': 'K-Means',
    'agglomerative': 'Agglomerative Clustering',
    'gmm': 'Gaussian Mixture Model',
}


def parse_arguments():
    '''
    parsing Command Line arguments and perform sanity checks
    '''

    parser = argparse.ArgumentParser(
        description='Klassifikation / Clustering von Einkaufsdaten')

    parser.add_argument('--labelled', type=Path,
                        help='CSV mit labelled Daten: Supervised')

    parser.add_argument('--unlabelled', type=Path,
                        help='CSV ohne Label: Unsupervised')

    parser.add_argument('--address', choices=['none', 'yes', 'both'], default='none',
                        help='Filialadresse verwenden: none, yes oder both')

    parser.add_argument(
        '--supervised-model', choices=list(SUPERVISED_MODEL_NAMES),
        default='logistic-regression',
        help='Supervised-Klassifikator (Standard: logistic-regression)',
    )
    parser.add_argument(
        '--unsupervised-model', choices=list(UNSUPERVISED_MODEL_NAMES),
        default='kmeans',
        help='Clustering-Algorithmus (Standard: kmeans)',
    )

    parser.add_argument('--clusters', type=int, default=2,
                        help='Anzahl Cluster (Standard: 2)'
                        '(Anzahl der unterschiedlichen Personen, die einkaufen.)')

    # if active, cluster size (k) is optimized
    parser.add_argument('--fixed-clusters', action='store_true',
                        help='Ausgewähltes Clustering-Modell mit --clusters ausführen, ohne Silhouette-Optimierung')

    parser.add_argument('--output', type=Path, default=Path(''),
                        help='Ausgabeordner (Standard: results/results)')

    args = parser.parse_args()
    # sanity check, either one labelled or unlabelled data set is required
    if not args.labelled and not args.unlabelled:
        parser.error('Bitte mindestens --labelled oder --unlabelled angeben.')
    # sanity check, for KMeans at least 2 clusters are required
    if args.clusters < 2:
        parser.error('--clusters muss mindestens 2 sein.')

    return args


def read_product_data(csv_path, labelled):
    '''
    read CSV file and return DataFrame
    '''

    rows = pd.read_csv(csv_path, dtype={'receipt_id': str, 'zipcode': str})

    # sanity check, labelled data requires column with label
    missing = (
        REQUIRED_ROWS_LABELLED if labelled else REQUIRED_ROWS_UNLABELLED) - set(rows.columns)
    if missing:
        raise ValueError(f'{csv_path}: Fehlende Spalten: {sorted(missing)}')

    return rows


def aggregate_receipts(rows, labelled):
    '''
    transform the data where every row is one product to data
    where every row is one shopping basket (receipt)
    '''

    rows = rows.copy()
    # convert price columns to suit float dtype
    for column in ['unit_price', 'quantity']:
        rows[column] = pd.to_numeric(
            rows[column].astype(str).str.replace(',', '.'))

    # calculate total price (product price * units)
    line_value = rows['unit_price'] * rows['quantity']
    if 'total_price' in rows:
        supplied = pd.to_numeric(
            rows['total_price'].astype(str).str.replace(',', '.'), errors='coerce'
        )
        rows['line_total'] = supplied.fillna(line_value)
    else:
        rows['line_total'] = line_value

    # convert time and date column to timestamp column
    rows['timestamp'] = pd.to_datetime(
        rows['date'].astype(str) + ' ' + rows['time'].astype(str),
        format='%d.%m.%Y %H:%M:%S',
    )

    # process address columns and combine them to one column
    for column in ADDRESS_COLUMNS:
        if column not in rows.columns:
            rows[column] = 'Unbekannt'
    rows['store'] = rows[ADDRESS_COLUMNS].fillna(
        '').astype(str).agg(' '.join, axis=1)

    # group purchases to one receipts
    receipts = rows.groupby('receipt_id', sort=False).agg(
        products=('product_name', lambda values: ' '.join(values.astype(str))),
        positions=('product_name', 'size'),
        items=('quantity', 'sum'),
        total=('line_total', 'sum'),
        avg_price=('unit_price', 'mean'),
        timestamp=('timestamp', 'first'),
        store=('store', 'first'),
    ).reset_index()

    # add the time of purchase (hour of the day)
    receipts['hour'] = receipts['timestamp'].dt.hour + \
        receipts['timestamp'].dt.minute / 60
    # add the weekday of the purchase
    receipts['weekday'] = receipts['timestamp'].dt.weekday

    # add buyer column to receipts if the script is used with labelled data
    if labelled:
        rows['buyer'] = (rows['buyer'].astype(
            str).str.strip().str.upper().replace({'WOMEN': 'WOMAN'}))
        if not set(rows['buyer'].unique()).issubset(LABELS):
            raise ValueError(
                'Die Spalte buyer muss MEN oder WOMAN/WOMEN enthalten.')

        # each row in the data has a label for buyer, here it is mapped to the receipts
        targets = rows.groupby('receipt_id')['buyer'].agg(
            lambda values: values.mode().iloc[0])
        receipts['buyer'] = receipts['receipt_id'].map(targets)

    return receipts


def make_feature_transformer(with_address):
    '''
    transforms each receipts into numeric values, standardizes numeric values
    '''
    transformers = [
        ('products', TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_TFIDF_FEATURES),
         'products'),
        ('numeric', StandardScaler(), NUMERIC_FEATURES),
    ]
    weights = {'products': 1.0, 'numeric': 0.3}
    if with_address:
        transformers.append(('store', OneHotEncoder(
            handle_unknown='ignore'), ['store']))
        weights['store'] = 0.6
    return ColumnTransformer(transformers, transformer_weights=weights)


def evaluate_supervised(y_true, y_pred, probability_woman):
    '''
    evaluation of supervised approach
    '''
    # evaluation values
    values = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'roc_auc': (roc_auc_score((y_true == 'WOMAN').astype(int), probability_woman)
                    if len(np.unique(y_true)) == 2 else None),
    }
    # confusion matrix
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    # precision, recall, fscore, support
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    class_scores = {
        label: {'precision': p[i], 'recall': r[i],
                'f1': f[i], 'support': support[i]}
        for i, label in enumerate(LABELS)
    }

    return values, matrix, class_scores


def make_supervised_model(with_address, model_name, n_train):
    '''
    create supervised models
    '''
    classifiers = {
        'logistic-regression': LogisticRegression(
            max_iter=1000, class_weight='balanced'),
        'svm': SVC(
            kernel='linear', probability=True,
            class_weight='balanced', random_state=RANDOM_STATE),
        'knn': KNeighborsClassifier(
            n_neighbors=min(5, n_train)),
        'decision-tree': DecisionTreeClassifier(
            max_depth=8, class_weight='balanced', random_state=RANDOM_STATE),
    }

    return Pipeline([('features', make_feature_transformer(with_address)),
                    ('classifier', classifiers[model_name])])


def split_supervised_receipts(receipts):
    '''
    splits data into train and test data
    '''
    return train_test_split(receipts, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=receipts['buyer'])


def cross_validate_supervised_model(model, train):
    '''
    checks cross validation for supervised models
    '''
    minimum_class_size = train['buyer'].value_counts().min()
    cv_splits = min(CROSS_VALIDATION_SPLITS, minimum_class_size)
    if cv_splits < 2:
        return None, None
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                         random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, train, train['buyer'], cv=cv, scoring='f1_macro')
    return float(scores.mean()), float(scores.std())


def train_supervised_model(train, with_address, model_name):
    '''
    executes supervised model and performs cross validation
    '''
    model = make_supervised_model(with_address, model_name, len(train))
    cv_mean, cv_std = cross_validate_supervised_model(model, train)
    model.fit(train, train['buyer'])
    return model, cv_mean, cv_std


def predict_supervised_receipts(model, test):
    '''
    performs prediction on test data
    '''
    predictions = model.predict(test)
    classes = list(model.named_steps['classifier'].classes_)
    probabilities = model.predict_proba(test)[:, classes.index('WOMAN')]
    return predictions, probabilities


def run_supervised(receipts, with_address, model_name):
    '''
    splits data, traines models and evaluates the classifications
    '''
    result = {
        'method': 'Supervised', 'model': SUPERVISED_MODEL_NAMES[model_name],
        'split': '80 % Training / 20 % Test', 'n_train': 0, 'n_test': 0,
        'metrics': {}, 'confusion': None, 'per_class': {}, 'counts': None,
        'note': '', 'cv_mean': None, 'cv_std': None,
    }
    class_counts = receipts['buyer'].value_counts()
    if (len(receipts) < MIN_SUPERVISED_RECEIPTS or len(class_counts) != 2
            or class_counts.min() < MIN_RECEIPTS_PER_CLASS):
        result['split'] = 'Nicht durchgeführt'
        result['note'] = ('Keine Bewertung: erforderlich sind mindestens '
                          f'{MIN_SUPERVISED_RECEIPTS} Bons und '
                          f'{MIN_RECEIPTS_PER_CLASS} je Klasse.')
        return result, pd.DataFrame()

    train, test = split_supervised_receipts(receipts)
    model, cv_mean, cv_std = train_supervised_model(
        train, with_address, model_name)
    predictions, probability_woman = predict_supervised_receipts(model, test)
    scores, matrix, class_scores = evaluate_supervised(
        test['buyer'], predictions, probability_woman)
    result.update(n_train=len(train), n_test=len(test), metrics=scores,
                  confusion=matrix, per_class=class_scores,
                  cv_mean=cv_mean, cv_std=cv_std)
    assignments = test[['receipt_id', 'timestamp', 'buyer']].copy()
    assignments['prediction'] = predictions
    assignments['probability_woman'] = probability_woman
    return result, assignments


def evaluate_unsupervised(representation, cluster_labels):
    '''
    evaluation of unsupervised approach
    '''
    # if there is only one cluster -> no metrics can be calculated
    unique = len(np.unique(cluster_labels))
    if not 1 < unique < len(cluster_labels):
        return {}

    dense = representation.toarray() if hasattr(
        representation, 'toarray') else representation

    # unsupervised evaluation metrics
    return {
        'silhouette': silhouette_score(representation, cluster_labels),
        'davies_bouldin': davies_bouldin_score(dense, cluster_labels),
        'calinski_harabasz': calinski_harabasz_score(dense, cluster_labels),
    }


def make_unsupervised_model(model_name, clusters):
    '''
    creates selected unsupervised model.
    '''
    models = {
        'kmeans': lambda: KMeans(n_clusters=clusters, n_init=10, random_state=RANDOM_STATE),
        'agglomerative': lambda: AgglomerativeClustering(n_clusters=clusters),
        'gmm': lambda: GaussianMixture(n_components=clusters, random_state=RANDOM_STATE),
    }
    return models[model_name]()


def prepare_clustering_features(representation, model_name):
    '''
    converts sparse matrix to dense matrix
    '''
    if model_name in ('agglomerative', 'gmm') and hasattr(representation, 'toarray'):
        return representation.toarray()
    return representation


def cluster_receipts(representation, clusters, model_name):
    '''
    use unsupervised model.
    '''
    model = make_unsupervised_model(model_name, clusters)
    return model.fit_predict(representation)


def select_cluster_count(representation, requested_clusters, model_name, fixed_clusters=False):
    '''
    selects cluster size with best silhouette score
    '''
    n_receipts = representation.shape[0]
    if requested_clusters > n_receipts and fixed_clusters:
        raise ValueError(
            f'--clusters={requested_clusters}, aber nur {n_receipts} Bons vorhanden.')
    if fixed_clusters:
        return requested_clusters, {}

    candidates = range(
        2, min(max(MAX_CLUSTER_SEARCH, requested_clusters), n_receipts - 1) + 1)
    scores = {}
    for k in candidates:
        labels = cluster_receipts(representation, k, model_name)
        if 1 < len(np.unique(labels)) < n_receipts:
            scores[k] = silhouette_score(representation, labels)
    if not scores:
        raise ValueError(
            'Keine gültige Clusteraufteilung für die Silhouette-Optimierung gefunden.')
    return max(scores, key=scores.get), scores


def run_unsupervised(receipts, with_address, clusters, model_name, fixed_clusters=False):
    '''
    executes and eveluates unsupervised models
    '''
    if len(receipts) < 3:
        raise ValueError(
            'Für das Clustering werden mindestens 3 Bons benötigt.')

    representation = make_feature_transformer(
        with_address).fit_transform(receipts)
    representation = prepare_clustering_features(representation, model_name)
    best_k, silhouette_candidates = select_cluster_count(
        representation, clusters, model_name, fixed_clusters)
    predicted = cluster_receipts(representation, best_k, model_name)
    counts = np.bincount(predicted, minlength=best_k)
    result = {
        'method': 'Unsupervised',
        'model': f'{UNSUPERVISED_MODEL_NAMES[model_name]} (k={best_k})',
        'split': '100 % Clustering; kein Train/Test-Split',
        'n_train': len(receipts), 'n_test': 0,
        'metrics': evaluate_unsupervised(representation, predicted),
        'confusion': None, 'per_class': {},
        'counts': counts,
        'selected_clusters': best_k,
        'silhouette_candidates': silhouette_candidates,
    }
    assignments = receipts[['receipt_id', 'timestamp']].copy()
    assignments['cluster'] = predicted
    return result, assignments


def print_section(title, char='#'):
    '''
    print secion for console output
    '''
    if char == ' ':
        print(f'{title}\n')
    else:
        print(f'{char * 69 }\n{title}\n{char * 69}\n')


def format_score(value):
    '''
    format precision of score value if available
    '''
    return 'nicht definiert' if value is None or pd.isna(value) else f'{value:.3f}'


def print_dataset_summary(path, rows, receipts, labelled):
    '''
    print summary of processed csv data set
    '''
    print(f'Datei eingelesen: {path}')
    print(f'Datenumfang: {len(rows)} Produktkäufe in {len(receipts)} Bons.')
    if labelled:
        classes = receipts['buyer'].value_counts().to_dict()
        print('Klassenverteilung: ' +
              ', '.join(f'{k}={v}' for k, v in classes.items()) + '\n')


def print_supervised_results(result):
    '''
    print supervised evaluation results
    '''
    if not result['metrics']:
        print(f'Bewertung nicht möglich: {result["note"]}')
        return

    print('Die Konfusionsmatrix der Supervised-Analyse zeigt:')
    print('                         Prognose MEN   Prognose WOMAN')

    for i, label in enumerate(LABELS):
        print(f'  Tatsächlich {label:<7} {result["confusion"][i, 0]:>12} '
              f'{result["confusion"][i, 1]:>16}')

    correct = int(result["confusion"].trace())
    print(
        f'Damit wurden {correct} von {result["n_test"]} Test-Bons korrekt klassifiziert.\n')


def print_evaluation(result, address):
    '''
    print evaluation metrics
    '''
    variant = 'mit Filialadresse' if address else 'ohne Filialadresse'

    print_section(
        f'Modellierung und Evaluation: {result["method"].upper()} - {variant} ...', ' ')

    print(f'Modell: {result["model"]}')
    print(f'Aufteilung: {result["split"]}')
    print(
        f'Trainings-/Cluster-Bons: {result["n_train"]}  |  Test-Bons: {result["n_test"]}')

    if result['method'] == 'Supervised':
        print_supervised_results(result)


def plot_confusion_matrix(axis, matrix):
    '''
    plot confusion matrix
    '''
    axis.imshow(matrix, cmap='Oranges', vmin=0)
    axis.set_title('Konfusionsmatrix (Einkäufe gelabelled)', fontsize=13)
    axis.set_xticks([0, 1], LABELS)
    axis.set_yticks([0, 1], LABELS)
    axis.set_xlabel('Vorhergesagte Klasse')
    axis.set_ylabel('Tatsächliche Klasse')

    for (row, col), value in np.ndenumerate(matrix):
        axis.text(col, row, str(value), ha='center', va='center', fontsize=17,
                  color='white' if value > matrix.max() / 2 and value else 'black')


def plot_cluster_sizes(axis, counts):
    '''
    cluster plots
    '''
    x = np.arange(len(counts))
    # cluster colors depend on size of clusters
    axis.bar(x, counts, color=[CLUSTER_COLORS[i %
             len(CLUSTER_COLORS)] for i in x])
    axis.set_xticks(x, [f'Cluster {i}' for i in x],
                    rotation=45 if len(x) > 5 else 0,
                    ha='right' if len(x) > 5 else 'center')
    axis.set_ylabel('Anzahl Bons')
    axis.set_title('Verteilung der Bons auf Cluster', fontsize=13)
    axis.set_ylim(0, max(counts) * 1.2 + 1)

    for i, count in enumerate(counts):
        axis.text(i, count, str(count), ha='center', va='bottom')


def create_dashboard(result, csv_path, product_count, receipt_count, with_address):
    '''
    creates A4 landscape dashboard per model (with or w/ address and supervised or unsupervised
    '''

    fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
    address = 'mit Filialadresse' if with_address else 'ohne Filialadresse'
    fig.text(.06, .92, f'{result["method"]} - {address}', fontsize=21,
             color=COLOR, weight='bold')
    fig.text(.06, .87, f'Quelle: {Path(csv_path).name}',
             fontsize=10, color=COLOR)
    fig.text(.06, .81, f'Produktkäufe: {product_count}   |   Bons: {receipt_count}   |   '
             f'Modell: {result["model"]}', fontsize=12, weight="bold")
    split_details = f'Training: {result["n_train"]}   |   Test: {result["n_test"]}'
    fig.text(.06, .765,
             f'Datenaufteilung: {result["split"]}   |   {split_details}', fontsize=10.5)
    fig.add_artist(plt.Line2D([.06, .94], [.73, .73], transform=fig.transFigure,
                              color=COLOR, linewidth=1.5))

    axis_metrics = fig.add_axes([.06, .24, .45, .43])
    axis_metrics.axis('off')
    axis_metrics.text(0, 1.06, 'Evaluationsmetriken', fontsize=15,
                      weight='bold', color=COLOR)

    metric_labels = (
        [('Accuracy', 'accuracy'), ('Balanced Accuracy', 'balanced_accuracy'),
         ('Precision (Macro)', 'precision'), ('Recall (Macro)', 'recall'),
         ('F-Score (Macro)', 'f1'), ('ROC-AUC', 'roc_auc'), ('MCC', 'mcc')]
        if result['method'] == 'Supervised' else
        [('Silhouette Score', 'silhouette'), ('Davies-Bouldin-Index', 'davies_bouldin'),
         ('Calinski-Harabasz-Index', 'calinski_harabasz')]
    )

    for index, (label, key) in enumerate(metric_labels):
        y = .92 - index * .125
        axis_metrics.text(0, y, label, fontsize=11.5,
                          transform=axis_metrics.transAxes)
        axis_metrics.text(.80, y, format_score(result['metrics'].get(key)),
                          fontsize=11.5, weight='bold', transform=axis_metrics.transAxes)

    axis_chart = fig.add_axes([.59, .29, .34, .37])

    if result['confusion'] is not None:
        plot_confusion_matrix(axis_chart, result['confusion'])
    elif result['counts'] is not None:
        plot_cluster_sizes(axis_chart, result['counts'])
    else:
        axis_chart.axis('off')
        axis_chart.text(.5, .5, 'Keine Konfusionsmatrix:\nzu wenig Test-Bons',
                        fontsize=13, ha='center', va='center')

    if result.get('cv_mean') is not None:
        fig.text(.06, .19, f'Cross-Validation Macro-F-Score: {result["cv_mean"]:.3f} '
                 f'(+/- {result["cv_std"]:.3f})', fontsize=10, color=COLOR)
    elif result.get('silhouette_candidates'):
        fig.text(.06, .19, 'Silhouette-Optimierung: k='
                 f'{result["selected_clusters"]} (Suche 2 bis '
                 f'{max(result["silhouette_candidates"])})', fontsize=10, color=COLOR)

    return fig


def save_predictions(assignments, output_dir, key):
    '''
    export predictions as CSV file
    '''
    if not assignments.empty:
        assignments.to_csv(output_dir / f'bons_{key}.csv', index=False)


def save_plot(figure, output_dir, key):
    '''
    export predictions as PNG files
    '''
    figure.savefig(output_dir / f'{key}.png', dpi=400)


def save_pdf(figures, output_dir):
    '''
    export pdf result report
    '''
    with PdfPages(output_dir / OUTPUT_PDF) as pdf:
        for figure in figures:
            pdf.savefig(figure)
            plt.close(figure)


def save_summary(results, output_dir):
    '''
    exports all results as a simple CSV file for further processing
    '''
    pd.DataFrame(results).to_csv(output_dir / OUTPUT_SUMMARY, index=False)


def print_comparison(overview):
    '''
    summary in the console
    '''
    print('\nZusammenfassung aller Auswertungen:')
    print(f'  {"Verfahren":<15} {"Adresse":<12} {"Hauptmetrik":<20} {"Wert":>10}')

    for entry in overview:
        supervised = entry['verfahren'] == 'Supervised'
        metric = 'F-Score (Macro)' if supervised else 'Silhouette Score'
        score = entry.get('f1') if supervised else entry.get('silhouette')
        print(
            f'{entry["verfahren"]:<15} {entry["adresse"]:<12} {metric:<20} {format_score(score):>10}')


def summary_record(result, address, product_count, receipt_count):
    '''
    collects metrics data for the final summary in console and CSV output
    '''
    return {
        'verfahren': result['method'],
        'adresse': 'ja' if address else 'nein',
        'produktkaeufe': product_count, 'bons': receipt_count,
        'modell': result['model'], 'training': result['n_train'],
        'test': result['n_test'],
        'cv_f1_mean': result.get('cv_mean'), 'cv_f1_std': result.get('cv_std'),
        'selected_clusters': result.get('selected_clusters'),
        **result['metrics'],
    }


def process_dataset(method, path, address_variants, args, output_dir, figures, overview):
    '''
    processing data, aggregate receiptss and perform models
    '''
    print_section(
        f'Daten einlesen und Vorverarbeitung: {method.upper()} ...', char=' ')
    labelled = method == 'supervised'
    product_rows = read_product_data(path, labelled=labelled)
    receipts = aggregate_receipts(product_rows, labelled=labelled)
    print_dataset_summary(path, product_rows, receipts, labelled=labelled)

    for with_address in address_variants:
        process_model_variant(method, receipts, product_rows,
                              path, with_address, args, output_dir, figures, overview)


def process_model_variant(method, receipts, product_rows, path, with_address, args, output_dir, figures, overview):
    '''
    execute model, evaluate and visualize result
    '''
    print_section('Feature Engineering und Modellberechnung ...', char=' ')
    if method == 'supervised':
        result, assignments = run_supervised(
            receipts, with_address, args.supervised_model)
    else:
        result, assignments = run_unsupervised(
            receipts, with_address, args.clusters,
            args.unsupervised_model, args.fixed_clusters)
    print_evaluation(result, with_address)

    suffix = 'mit_adresse' if with_address else 'ohne_adresse'
    key = f'{method}_{suffix}'
    save_predictions(assignments, output_dir, key)
    figure = create_dashboard(result, path, len(
        product_rows), len(receipts), with_address)
    save_plot(figure, output_dir, key)
    figures.append(figure)
    overview.append(summary_record(result, with_address,
                    len(product_rows), len(receipts)))


def export_analysis_results(figures, overview, output_dir):
    '''
    save result export
    '''
    save_pdf(figures, output_dir)
    save_summary(overview, output_dir)
    print_comparison(overview)


def main():
    '''
    main method of the script
    '''

    args = parse_arguments()
    output_dir = OUTPUT_BASE_DIR / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    address_variants = [False, True] if args.address == 'both' else [
        args.address == 'yes']
    figures, overview = [], []

    print_section('Data Science Use Case - Start der Analyse')

    for method, path in [('supervised', args.labelled), ('unsupervised', args.unlabelled)]:
        if path is not None:
            process_dataset(method, path, address_variants, args, output_dir,
                            figures, overview)

    export_analysis_results(figures, overview, output_dir)


# call main function
if __name__ == '__main__':
    main()
