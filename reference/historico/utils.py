import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import matplotlib.patches as mpatches
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import math
import shap
import xgboost as xgb
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay

def print_class_distribution(y, label):
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    total = sum(counts)
    print(f'Distribuição de classes {label}:')
    for k, v in distribution.items():
        print(f'Classe {k}: {v/total:.2%}')
    print('--------------------')



def calculate_binary_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calcula métricas de classificação binária a partir das probabilidades preditas.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    sensitivity = recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    g_mean = np.sqrt(sensitivity * specificity)

    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        auc = np.nan

    try:
        prauc = average_precision_score(y_true, y_pred_proba)
    except Exception:
        prauc = np.nan

    try:
        logloss = log_loss(y_true, y_pred_proba, labels=[0, 1])
    except Exception:
        logloss = np.nan

    try:
        brier = brier_score_loss(y_true, y_pred_proba)
    except Exception:
        brier = np.nan

    gini = 2 * auc - 1 if pd.notnull(auc) else np.nan
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "prauc": prauc,
        "log_loss": logloss,
        "balanced_accuracy": balanced_acc,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "g_mean": g_mean,
        "brier_score": brier,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "gini": gini,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP
    }




def evaluate_model_robustness_cv(
    model,
    X: pd.DataFrame,
    y,
    n_splits: int = 5,
    threshold: float = 0.5,
    shuffle: bool = True,
    random_state: int = 42,
    return_fold_models: bool = False
):
    """
    Avalia a robustez do modelo com cross-validation estratificado.

    Parâmetros
    ----------
    model : estimador sklearn-like
        Ex.: XGBClassifier, RandomForestClassifier, LogisticRegression etc.
    X : pd.DataFrame
        Features.
    y : array-like
        Target binária.
    n_splits : int
        Número de folds.
    threshold : float
        Threshold para converter probabilidade em classe.
    shuffle : bool
        Se True, embaralha antes de dividir os folds.
    random_state : int
        Semente.
    return_fold_models : bool
        Se True, retorna também os modelos treinados em cada fold.

    Retorna
    -------
    results : dict com
        - df_folds
        - df_summary
        - fold_models (opcional)
    """
    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    fold_results = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()

        model_fold = clone(model)
        model_fold.fit(X_train_fold, y_train_fold)

        y_val_proba = model_fold.predict_proba(X_val_fold)[:, 1]

        metrics = calculate_binary_classification_metrics(
            y_true=y_val_fold,
            y_pred_proba=y_val_proba,
            threshold=threshold
        )

        metrics["fold"] = fold
        metrics["n_train"] = len(train_idx)
        metrics["n_valid"] = len(val_idx)
        metrics["event_rate_valid"] = y_val_fold.mean()

        fold_results.append(metrics)

        if return_fold_models:
            fold_models.append(model_fold)

    df_folds = pd.DataFrame(fold_results)

    metric_cols = [
        "accuracy", "precision", "recall", "f1_score",
        "auc", "prauc", "log_loss", "balanced_accuracy",
        "specificity", "fpr", "fnr", "g_mean",
        "brier_score", "mcc", "cohen_kappa", "gini"
    ]

    df_summary = (
        df_folds[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    results = {
        "df_folds": df_folds,
        "df_summary": df_summary
    }

    if return_fold_models:
        results["fold_models"] = fold_models

    return results

def plot_cv_metrics_heatmap(
    df_folds: pd.DataFrame,
    metrics: list = None,
    figsize=(12, 6),
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    title="Robustez do Modelo - Heatmap por Fold"
):
    """
    Plota heatmap com folds nas linhas e métricas nas colunas.
    """
    if metrics is None:
        metrics = ["auc", "prauc", "recall", "precision", "f1_score"]

    heatmap_data = df_folds.set_index("fold")[metrics]

    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, annot=annot, fmt=fmt, cmap=cmap)
    plt.title(title, fontsize=14)
    plt.xlabel("Métrica")
    plt.ylabel("Fold")
    plt.tight_layout()
    plt.show()    

def get_cv_summary_table(
    df_folds: pd.DataFrame,
    metrics: list = None,
    round_digits: int = 4
):
    """
    Gera tabela-resumo das métricas nos folds.
    """
    if metrics is None:
        metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "auc", "prauc", "log_loss", "balanced_accuracy",
            "specificity", "fpr", "fnr", "g_mean",
            "brier_score", "mcc", "cohen_kappa", "gini"
        ]

    df_summary = (
        df_folds[metrics]
        .agg(["mean", "std", "min", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    return df_summary.round(round_digits)

def plot_cv_metrics_boxplot(
    df_folds: pd.DataFrame,
    metrics: list = None,
    figsize=(12, 6),
    title="Robustez do Modelo - Distribuição das Métricas nos Folds"
):
    """
    Plota boxplot das métricas ao longo dos folds.
    """
    if metrics is None:
        metrics = ["auc", "prauc", "recall", "precision", "f1_score"]

    df_long = df_folds.melt(
        id_vars=["fold"],
        value_vars=metrics,
        var_name="metric",
        value_name="value"
    )

    plt.figure(figsize=figsize)
    sns.boxplot(data=df_long, x="metric", y="value")
    sns.stripplot(data=df_long, x="metric", y="value", color="black", alpha=0.6, size=5)
    plt.title(title, fontsize=14)
    plt.xlabel("Métrica")
    plt.ylabel("Valor")
    plt.xticks(rotation=25)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_full_cv_robustness_analysis(
    model,
    X: pd.DataFrame,
    y,
    n_splits: int = 5,
    threshold: float = 0.5,
    random_state: int = 42,
    metrics_to_plot: list = None,
    heatmap_figsize=(12, 6),
    boxplot_figsize=(12, 6)
):
    """
    Executa a análise completa de robustez com CV:
    - calcula métricas por fold
    - gera tabela resumo
    - plota heatmap
    - plota boxplot
    """
    results = evaluate_model_robustness_cv(
        model=model,
        X=X,
        y=y,
        n_splits=n_splits,
        threshold=threshold,
        shuffle=True,
        random_state=random_state,
        return_fold_models=False
    )

    df_folds = results["df_folds"]
    df_summary = results["df_summary"]

    if metrics_to_plot is None:
        metrics_to_plot = ["auc", "prauc", "recall", "precision", "f1_score"]

    print("Resultados por fold:")
    print(df_folds.round(4))

    print("\nTabela resumo:")
    print(
        df_summary[
            ["metric", "mean", "std", "min", "max"]
        ].round(4)
    )

    plot_cv_metrics_heatmap(
        df_folds=df_folds,
        metrics=metrics_to_plot,
        figsize=heatmap_figsize
    )

    plot_cv_metrics_boxplot(
        df_folds=df_folds,
        metrics=metrics_to_plot,
        figsize=boxplot_figsize
    )

    return {
        "df_folds": df_folds,
        "df_summary": df_summary
    }

def get_pred_proba(model, X, model_type="sklearn"):
    """
    Retorna probabilidade da classe positiva.

    model_type:
    - 'sklearn': usa predict_proba(X)[:, 1]
    - 'xgb_booster': usa model.predict(X), onde X deve ser DMatrix
    """
    if model_type == "sklearn":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            return 1 / (1 + np.exp(-scores))
        else:
            raise ValueError("Modelo sem predict_proba e sem decision_function.")
    
    elif model_type == "xgb_booster":
        return model.predict(X)
    
    else:
        raise ValueError("model_type deve ser 'sklearn' ou 'xgb_booster'.")

def get_pred_class(y_pred_proba, threshold=0.5):
    return (np.array(y_pred_proba) >= threshold).astype(int)

def plot_score_distribution(
    model,
    X,
    y,
    threshold=0.5,
    model_type="sklearn",
    dataset_name="Dataset",
    bins=40,
    kde=True,
    figsize=(10, 6)
):
    """
    Plota distribuição dos scores preditos por classe real.
    """
    y_true = np.array(y)
    y_pred_proba = get_pred_proba(model, X, model_type=model_type)

    data = pd.DataFrame({
        "True Label": y_true,
        "Predicted Probability": y_pred_proba
    })

    plt.figure(figsize=figsize)
    sns.histplot(
        data=data,
        x="Predicted Probability",
        hue="True Label",
        kde=kde,
        bins=bins,
        alpha=0.5,
        stat="count",
        common_norm=False
    )
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.3f}")
    plt.title(f"Distribuição dos Scores Preditos por Classe - {dataset_name}", fontsize=14)
    plt.xlabel("Probabilidade Predita da Classe Positiva", fontsize=12)
    plt.ylabel("Frequência", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_xgb_feature_importance(
    model,
    max_num_features=20,
    importance_type="weight",
    figsize=(13, 9),
    title=None
):
    """
    Plota importância de variáveis do XGBoost.
    
    importance_type:
    - 'weight'
    - 'gain'
    - 'cover'
    - 'total_gain'
    - 'total_cover'
    """
    plt.figure(figsize=figsize)

    booster = model.get_booster() if hasattr(model, "get_booster") else model

    xgb.plot_importance(
        booster,
        max_num_features=max_num_features,
        importance_type=importance_type
    )

    if title is None:
        title = f"Importância das Variáveis - {importance_type}"
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_threshold(
    model,
    X,
    y,
    threshold=0.5,
    model_type="sklearn",
    dataset_name="Dataset",
    figsize=(6, 5),
    cmap="Blues"
):
    """
    Plota matriz de confusão com threshold escolhido.
    """
    y_true = np.array(y)
    y_pred_proba = get_pred_proba(model, X, model_type=model_type)
    y_pred = get_pred_class(y_pred_proba, threshold=threshold)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False)
    plt.title(f"Matriz de Confusão - {dataset_name} | threshold = {threshold:.3f}")
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")
    plt.tight_layout()
    plt.show()

    return cm

def plot_roc_curve_dataset(
    model,
    X,
    y,
    model_type="sklearn",
    dataset_name="Dataset",
    figsize=(8, 6)
):
    """
    Plota curva ROC.
    """
    y_true = np.array(y)
    y_pred_proba = get_pred_proba(model, X, model_type=model_type)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_value = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.title(f"Curva ROC - {dataset_name}", fontsize=14)
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"auc": auc_value}

def plot_pr_curve_dataset(
    model,
    X,
    y,
    model_type="sklearn",
    dataset_name="Dataset",
    figsize=(8, 6)
):
    """
    Plota curva Precision-Recall.
    """
    y_true = np.array(y)
    y_pred_proba = get_pred_proba(model, X, model_type=model_type)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    prauc = average_precision_score(y_true, y_pred_proba)
    baseline = np.mean(y_true)

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, linewidth=2, label=f"PR AUC = {prauc:.4f}")
    plt.axhline(baseline, linestyle="--", linewidth=1, label=f"Baseline = {baseline:.4f}")
    plt.title(f"Curva Precision-Recall - {dataset_name}", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"prauc": prauc}


def calculate_ece_mce(y_true, y_pred_proba, bins=10):
    """
    Calcula ECE e MCE.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
        else:
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])

        if np.sum(mask) > 0:
            avg_conf = np.mean(y_pred_proba[mask])
            frac_pos = np.mean(y_true[mask])
            error = abs(frac_pos - avg_conf)

            ece += error * np.sum(mask)
            mce = max(mce, error)

    ece /= len(y_true)
    return ece, mce

def calibration_metrics(y_true, y_pred_proba, bins=10):
    """
    Calcula métricas de calibração.
    """
    return {
        "log_loss": log_loss(y_true, y_pred_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
        "ece": calculate_ece_mce(y_true, y_pred_proba, bins=bins)[0],
        "mce": calculate_ece_mce(y_true, y_pred_proba, bins=bins)[1]
    }

def plot_calibration_multiple_datasets(
    model,
    datasets: dict,
    model_type="sklearn",
    n_bins=10,
    strategy="uniform",
    figsize=(10, 7)
):
    """
    Plota gráfico de calibração para múltiplas bases e retorna métricas.
    
    datasets:
    {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "oot": (X_oot, y_oot)
    }
    """
    plt.figure(figsize=figsize)

    calibration_results = []

    for dataset_name, (X_data, y_data) in datasets.items():
        y_true = np.array(y_data)
        y_pred_proba = get_pred_proba(model, X_data, model_type=model_type)

        prob_true, prob_pred = calibration_curve(
            y_true,
            y_pred_proba,
            n_bins=n_bins,
            strategy=strategy
        )

        metrics = calibration_metrics(y_true, y_pred_proba, bins=n_bins)
        metrics["dataset"] = dataset_name
        calibration_results.append(metrics)

        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label=f"{dataset_name} | Brier={metrics['brier_score']:.4f} | ECE={metrics['ece']:.4f}"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfeitamente calibrado")
    plt.xlabel("Probabilidade prevista")
    plt.ylabel("Proporção observada")
    plt.title("Gráfico de Calibração por Dataset")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(calibration_results)        

def plot_shap_beeswarm(
    model,
    X,
    max_display=20,
    figsize=(10, 8)
):
    """
    Plota SHAP beeswarm para modelos baseados em árvore.
    """
    plt.figure(figsize=figsize)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
        plot_type="dot",
        max_display=max_display,
        show=False
    )

    plt.title("SHAP Beeswarm")
    plt.tight_layout()
    plt.show()

def plot_shap_importance_bar(
    model,
    X,
    max_display=20,
    figsize=(10, 8)
):
    """
    Plota SHAP bar plot de importância global.
    """
    plt.figure(figsize=figsize)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=max_display,
        show=False
    )

    plt.title("SHAP Importance")
    plt.tight_layout()
    plt.show()



def get_pred_proba(model, X, model_type="sklearn"):
    """
    Retorna probabilidade da classe positiva.
    """
    if model_type == "sklearn":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            return 1 / (1 + np.exp(-scores))
        else:
            raise ValueError("Modelo sem predict_proba e sem decision_function.")
    elif model_type == "xgb_booster":
        return model.predict(X)
    else:
        raise ValueError("model_type deve ser 'sklearn' ou 'xgb_booster'.")


def calculate_temporal_stability_metrics(
    df_base: pd.DataFrame,
    X: pd.DataFrame,
    y,
    model,
    time_col: str,
    threshold: float = 0.5,
    model_type: str = "sklearn",
    min_obs: int = 20,
    positive_label: int = 1
):
    """
    Calcula métricas por período temporal.

    Parâmetros
    ----------
    df_base : pd.DataFrame
        DataFrame base que contém a coluna temporal.
        O índice deve estar alinhado com X e y.

    X : pd.DataFrame
        Features usadas para previsão.

    y : array-like
        Target binária.

    model : modelo treinado

    time_col : str
        Coluna temporal para agrupamento, ex.: 'mes', 'data_dia'

    threshold : float
        Threshold para converter probabilidade em classe

    model_type : str
        'sklearn' ou 'xgb_booster'

    min_obs : int
        Quantidade mínima de observações por período para calcular métricas

    Retorna
    -------
    pd.DataFrame
    """
    y_true = np.array(y)
    y_proba = get_pred_proba(model, X, model_type=model_type)
    y_pred = (y_proba >= threshold).astype(int)

    df_eval = df_base.copy().loc[X.index].copy()
    df_eval["y_true"] = y_true
    df_eval["y_proba"] = y_proba
    df_eval["y_pred"] = y_pred

    results = []

    for period, grp in df_eval.groupby(time_col):
        n_obs = len(grp)
        event_rate = grp["y_true"].mean()

        row = {
            time_col: period,
            "n_obs": n_obs,
            "event_rate": event_rate,
            "auc": np.nan,
            "prauc": np.nan,
            "recall": np.nan,
            "precision": np.nan,
            "f1_score": np.nan
        }

        if n_obs >= min_obs and grp["y_true"].nunique() > 1:
            try:
                row["auc"] = roc_auc_score(grp["y_true"], grp["y_proba"])
            except:
                row["auc"] = np.nan

            try:
                row["prauc"] = average_precision_score(grp["y_true"], grp["y_proba"])
            except:
                row["prauc"] = np.nan

        if n_obs >= min_obs:
            row["recall"] = recall_score(grp["y_true"], grp["y_pred"], zero_division=0, pos_label=positive_label)
            row["precision"] = precision_score(grp["y_true"], grp["y_pred"], zero_division=0, pos_label=positive_label)
            row["f1_score"] = f1_score(grp["y_true"], grp["y_pred"], zero_division=0, pos_label=positive_label)

        results.append(row)

    df_stability = pd.DataFrame(results).sort_values(time_col).reset_index(drop=True)
    return df_stability

def plot_temporal_metric(
    df_stability: pd.DataFrame,
    time_col: str,
    metric_col: str,
    title: str = None,
    figsize=(10, 5),
    marker="o",
    show_labels=False
):
    """
    Plota estabilidade temporal de uma métrica.
    """
    df_plot = df_stability[[time_col, metric_col, "n_obs", "event_rate"]].copy()
    df_plot = df_plot.dropna(subset=[metric_col])

    plt.figure(figsize=figsize)
    plt.plot(df_plot[time_col], df_plot[metric_col], marker=marker, linewidth=2)

    if show_labels:
        for _, row in df_plot.iterrows():
            plt.text(row[time_col], row[metric_col], f"{row[metric_col]:.3f}", fontsize=9)

    if title is None:
        title = f"Estabilidade Temporal - {metric_col.upper()}"

    plt.title(title, fontsize=14)
    plt.xlabel(time_col, fontsize=12)
    plt.ylabel(metric_col, fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()    


def plot_all_temporal_stability_metrics(
    df_stability: pd.DataFrame,
    time_col: str,
    figsize=(10, 5),
    show_labels=False
):
    """
    Plota AUC, PRAUC, Recall, Precision e F1-score em gráficos separados.
    """
    metrics_map = {
        "auc": "Estabilidade Temporal - AUC",
        "prauc": "Estabilidade Temporal - PRAUC",
        "recall": "Estabilidade Temporal - Recall",
        "precision": "Estabilidade Temporal - Precision",
        "f1_score": "Estabilidade Temporal - F1-Score"
    }

    for metric, title in metrics_map.items():
        plot_temporal_metric(
            df_stability=df_stability,
            time_col=time_col,
            metric_col=metric,
            title=title,
            figsize=figsize,
            show_labels=show_labels
        )




def get_pred_proba(model, X, model_type="sklearn"):
    """
    Retorna probabilidade da classe positiva.
    """
    if model_type == "sklearn":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            return 1 / (1 + np.exp(-scores))
        else:
            raise ValueError("Modelo sem predict_proba e sem decision_function.")
    elif model_type == "xgb_booster":
        return model.predict(X)
    else:
        raise ValueError("model_type deve ser 'sklearn' ou 'xgb_booster'.")


def calculate_temporal_metrics_for_dataset(
    df_base: pd.DataFrame,
    X: pd.DataFrame,
    y,
    model,
    time_col: str,
    dataset_name: str,
    threshold: float = 0.5,
    model_type: str = "sklearn",
    min_obs: int = 20,
    positive_label: int = 1
):
    """
    Calcula métricas temporais para um dataset específico.
    """
    y_true = np.array(y)
    y_proba = get_pred_proba(model, X, model_type=model_type)
    y_pred = (y_proba >= threshold).astype(int)

    df_eval = df_base.loc[X.index].copy()
    df_eval["y_true"] = y_true
    df_eval["y_proba"] = y_proba
    df_eval["y_pred"] = y_pred
    df_eval["dataset"] = dataset_name

    results = []

    for period, grp in df_eval.groupby(time_col):
        row = {
            "dataset": dataset_name,
            time_col: period,
            "n_obs": len(grp),
            "event_rate": grp["y_true"].mean(),
            "auc": np.nan,
            "prauc": np.nan,
            "recall": np.nan,
            "precision": np.nan,
            "f1_score": np.nan
        }

        # Métricas baseadas em probabilidade exigem as duas classes no período
        if len(grp) >= min_obs and grp["y_true"].nunique() > 1:
            try:
                row["auc"] = roc_auc_score(grp["y_true"], grp["y_proba"])
            except:
                row["auc"] = np.nan

            try:
                row["prauc"] = average_precision_score(grp["y_true"], grp["y_proba"])
            except:
                row["prauc"] = np.nan

        # Métricas baseadas em threshold
        if len(grp) >= min_obs:
            row["recall"] = recall_score(
                grp["y_true"], grp["y_pred"],
                zero_division=0,
                pos_label=positive_label
            )
            row["precision"] = precision_score(
                grp["y_true"], grp["y_pred"],
                zero_division=0,
                pos_label=positive_label
            )
            row["f1_score"] = f1_score(
                grp["y_true"], grp["y_pred"],
                zero_division=0,
                pos_label=positive_label
            )

        results.append(row)

    return pd.DataFrame(results).sort_values(time_col).reset_index(drop=True)

def build_temporal_stability_dataframe(
    model,
    datasets_dict: dict,
    time_col: str,
    threshold: float = 0.5,
    model_type: str = "sklearn",
    min_obs: int = 20,
    positive_label: int = 1
):
    """
    datasets_dict no formato:
    {
        "train": {"df": df_train_ref, "X": X_train, "y": y_train},
        "test":  {"df": df_test_ref,  "X": X_test,  "y": y_test},
        "oot":   {"df": df_oot_ref,   "X": X_oot,   "y": y_oot}
    }
    """
    all_results = []

    for dataset_name, data in datasets_dict.items():
        df_temp = calculate_temporal_metrics_for_dataset(
            df_base=data["df"],
            X=data["X"],
            y=data["y"],
            model=model,
            time_col=time_col,
            dataset_name=dataset_name,
            threshold=threshold,
            model_type=model_type,
            min_obs=min_obs,
            positive_label=positive_label
        )
        all_results.append(df_temp)

    df_all = pd.concat(all_results, axis=0, ignore_index=True)
    return df_all

def plot_temporal_metric_comparative(
    df_stability: pd.DataFrame,
    time_col: str,
    metric_col: str,
    dataset_col: str = "dataset",
    figsize=(11, 6),
    marker="o",
    show_labels=False,
    ylim=(0, 1),
    title=None
):
    """
    Plota uma métrica temporal com uma linha para cada dataset.
    """
    plt.figure(figsize=figsize)

    df_plot = df_stability.dropna(subset=[metric_col]).copy()

    for dataset_name, grp in df_plot.groupby(dataset_col):
        grp = grp.sort_values(time_col)

        plt.plot(
            grp[time_col],
            grp[metric_col],
            marker=marker,
            linewidth=2,
            label=dataset_name
        )

        if show_labels:
            for _, row in grp.iterrows():
                plt.text(
                    row[time_col],
                    row[metric_col],
                    f"{row[metric_col]:.3f}",
                    fontsize=8
                )

    if title is None:
        title = f"Estabilidade Temporal Comparativa - {metric_col.upper()}"

    plt.title(title, fontsize=14)
    plt.xlabel(time_col, fontsize=12)
    plt.ylabel(metric_col, fontsize=12)

    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.show()

def plot_all_temporal_metrics_comparative(
    df_stability: pd.DataFrame,
    time_col: str,
    dataset_col: str = "dataset",
    figsize=(11, 6),
    show_labels=False
):
    """
    Gera um gráfico separado para cada métrica,
    com uma linha para cada dataset.
    """
    metrics_titles = {
        "auc": "Estabilidade Temporal Comparativa - AUC",
        "prauc": "Estabilidade Temporal Comparativa - PRAUC",
        "recall": "Estabilidade Temporal Comparativa - Recall",
        "precision": "Estabilidade Temporal Comparativa - Precision",
        "f1_score": "Estabilidade Temporal Comparativa - F1-Score"
    }

    for metric, title in metrics_titles.items():
        plot_temporal_metric_comparative(
            df_stability=df_stability,
            time_col=time_col,
            metric_col=metric,
            dataset_col=dataset_col,
            figsize=figsize,
            show_labels=show_labels,
            ylim=(0, 1),
            title=title
        )

def calc_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Calcula métricas de classificação binária.
    y_prob: probabilidade da classe positiva
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(y_true, y_prob),
        "aucpr": average_precision_score(y_true, y_prob),
        "logloss": log_loss(y_true, y_prob, labels=[0, 1]),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
    }
    return metrics


def summarize_fold_metrics(metrics_list):
    """
    Recebe lista de dicionários e devolve média por métrica.
    """
    return pd.DataFrame(metrics_list).mean().to_dict()


def evaluate_model_on_dataset(model, X_data, y_data, threshold=0.5):
    """
    Avaliação única em qualquer base.
    """
    y_prob = model.predict_proba(X_data)[:, 1]
    return calc_classification_metrics(y_data, y_prob, threshold=threshold)

#  HIPERTUNING COM 5-FOLD CV

global_results = []

def hipertunnig(space, X_train, y_train):
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "max_depth": int(space["max_depth"]),
        "gamma": float(space["gamma"]),
        "reg_alpha": float(space["reg_alpha"]),
        "reg_lambda": float(space["reg_lambda"]),
        "min_child_weight": int(space["min_child_weight"]),
        "colsample_bytree": float(space["colsample_bytree"]),
        "colsample_bylevel": float(space["colsample_bylevel"]),
        "colsample_bynode": float(space["colsample_bynode"]),
        "n_estimators": int(space["n_estimators"]),
        "learning_rate": float(space["learning_rate"]),
        "max_delta_step": float(space["max_delta_step"]),
        "subsample": float(space["subsample"]),

        # Fixos recomendados
        "eval_metric": ["auc", "aucpr", "logloss"],
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "use_label_encoder": False
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_fold_metrics = []
    valid_fold_metrics = []
    fold_results = []

    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_reset, y_train_reset), start=1):
        X_tr = X_train_reset.iloc[tr_idx]
        X_val = X_train_reset.iloc[val_idx]
        y_tr = y_train_reset.iloc[tr_idx]
        y_val = y_train_reset.iloc[val_idx]

        model = XGBClassifier(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=False
        )

        # Probabilidades
        y_tr_prob = model.predict_proba(X_tr)[:, 1]
        y_val_prob = model.predict_proba(X_val)[:, 1]

        # Métricas treino e validação
        tr_metrics = calc_classification_metrics(y_tr, y_tr_prob, threshold=0.5)
        val_metrics = calc_classification_metrics(y_val, y_val_prob, threshold=0.5)

        train_fold_metrics.append(tr_metrics)
        valid_fold_metrics.append(val_metrics)

        fold_results.append({
            "fold": fold,
            "train_auc": tr_metrics["auc"],
            "valid_auc": val_metrics["auc"],
            "train_aucpr": tr_metrics["aucpr"],
            "valid_aucpr": val_metrics["aucpr"],
            "train_logloss": tr_metrics["logloss"],
            "valid_logloss": val_metrics["logloss"],
            "train_f1": tr_metrics["f1"],
            "valid_f1": val_metrics["f1"],
            "train_precision": tr_metrics["precision"],
            "valid_precision": val_metrics["precision"],
            "train_recall": tr_metrics["recall"],
            "valid_recall": val_metrics["recall"],
            "train_balanced_accuracy": tr_metrics["balanced_accuracy"],
            "valid_balanced_accuracy": val_metrics["balanced_accuracy"]
        })

    # Médias
    mean_train = summarize_fold_metrics(train_fold_metrics)
    mean_valid = summarize_fold_metrics(valid_fold_metrics)

    # Gaps absolutos treino vs validação
    gaps = {
        metric: abs(mean_train[metric] - mean_valid[metric])
        for metric in mean_train.keys()
    }

    # Regra de negócio: diferença não pode ser > 0.05
    overfit_limit = 0.05
    gap_violations = {
        metric: max(0.0, gap - overfit_limit)
        for metric, gap in gaps.items()
    }

    total_gap_penalty = sum(gap_violations.values())

    # Objetivo principal:
    # maximizar AUC-PR, com penalização forte se houver excesso de gap
    # Hyperopt minimiza => loss negativo da métrica boa + penalidade
    loss = -mean_valid["aucpr"] + (20.0 * total_gap_penalty)

    result_row = {
        "params": params,
        "loss": loss,
        "status": STATUS_OK,
        "mean_train_auc": mean_train["auc"],
        "mean_valid_auc": mean_valid["auc"],
        "gap_auc": gaps["auc"],

        "mean_train_aucpr": mean_train["aucpr"],
        "mean_valid_aucpr": mean_valid["aucpr"],
        "gap_aucpr": gaps["aucpr"],

        "mean_train_logloss": mean_train["logloss"],
        "mean_valid_logloss": mean_valid["logloss"],
        "gap_logloss": gaps["logloss"],

        "mean_train_f1": mean_train["f1"],
        "mean_valid_f1": mean_valid["f1"],
        "gap_f1": gaps["f1"],

        "mean_train_precision": mean_train["precision"],
        "mean_valid_precision": mean_valid["precision"],
        "gap_precision": gaps["precision"],

        "mean_train_recall": mean_train["recall"],
        "mean_valid_recall": mean_valid["recall"],
        "gap_recall": gaps["recall"],

        "mean_train_balanced_accuracy": mean_train["balanced_accuracy"],
        "mean_valid_balanced_accuracy": mean_valid["balanced_accuracy"],
        "gap_balanced_accuracy": gaps["balanced_accuracy"],

        "total_gap_penalty": total_gap_penalty,
        "fold_results": fold_results
    }

    global_results.append(result_row)

    return {
        "loss": loss,
        "status": STATUS_OK
    }



def evaluate_binary_classifier(
    model,
    datasets: dict,
    threshold: float = 0.5,
    positive_class: int = 1,
    model_type: str = "sklearn"
):
    """
    Avalia um classificador binário em múltiplos datasets.

    Parâmetros
    ----------
    model : modelo treinado
        Ex.: XGBClassifier, LogisticRegression, RandomForest, Booster etc.

    datasets : dict
        Dicionário no formato:
        {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
            "oot": (X_oot, y_oot)
        }

    threshold : float, default=0.5
        Limiar para transformar probabilidade em classe.

    positive_class : int, default=1
        Classe positiva.

    model_type : str, default="sklearn"
        - "sklearn": usa predict_proba(X)[:, 1]
        - "xgb_booster": usa model.predict(DMatrix)

    Retorna
    -------
    pd.DataFrame
        DataFrame com uma linha por dataset e colunas com as métricas.
    """

    def get_pred_proba(model, X, model_type="sklearn"):
        if model_type == "sklearn":
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                # transforma score em pseudo-probabilidade pela logística
                return 1 / (1 + np.exp(-scores))
            else:
                raise ValueError("O modelo não possui predict_proba nem decision_function.")
        elif model_type == "xgb_booster":
            return model.predict(X)
        else:
            raise ValueError("model_type deve ser 'sklearn' ou 'xgb_booster'.")

    results = []

    for dataset_name, (X_data, y_data) in datasets.items():
        y_true = np.array(y_data)
        y_pred_proba = get_pred_proba(model, X_data, model_type=model_type)
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
        else:
            TN, FP, FN, TP = 0, 0, 0, 0

        # Métricas de classificação
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        sensitivity = recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        g_mean = np.sqrt(sensitivity * specificity) if sensitivity >= 0 and specificity >= 0 else 0.0

        # Métricas probabilísticas
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = np.nan

        try:
            prauc = average_precision_score(y_true, y_pred_proba)
        except:
            prauc = np.nan

        try:
            logloss = log_loss(y_true, y_pred_proba, labels=[0, 1])
        except:
            logloss = np.nan

        try:
            brier = brier_score_loss(y_true, y_pred_proba)
        except:
            brier = np.nan

        gini = 2 * auc - 1 if pd.notnull(auc) else np.nan
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        result = {
            "dataset": dataset_name,
            "threshold": threshold,
            "n_obs": len(y_true),
            "event_rate": np.mean(y_true),

            "TN": TN,
            "FP": FP,
            "FN": FN,
            "TP": TP,

            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
            "balanced_accuracy": balanced_acc,
            "fpr": fpr,
            "fnr": fnr,
            "g_mean": g_mean,

            "auc": auc,
            "prauc": prauc,
            "gini": gini,
            "log_loss": logloss,
            "brier_score": brier,
            "mcc": mcc,
            "cohen_kappa": kappa
        }

        results.append(result)

    df_results = pd.DataFrame(results)
    return df_results

def compare_model_gaps(
    df_metrics: pd.DataFrame,
    base_ref: str = "train",
    base_compare: str = "test",
    metrics_to_compare: list = None,
    max_gap: float = 0.05
):
    """
    Compara gaps absolutos entre duas bases.

    Parâmetros
    ----------
    df_metrics : pd.DataFrame
        Saída da função evaluate_binary_classifier.

    base_ref : str
        Base de referência, ex.: 'train'

    base_compare : str
        Base a comparar, ex.: 'test' ou 'oot'

    metrics_to_compare : list
        Lista de métricas numéricas.

    max_gap : float
        Gap máximo permitido.

    Retorna
    -------
    pd.DataFrame
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "balanced_accuracy",
            "auc",
            "prauc",
            "log_loss",
            "specificity",
            "g_mean",
            "mcc",
            "cohen_kappa"
        ]

    row_ref = df_metrics[df_metrics["dataset"] == base_ref]
    row_cmp = df_metrics[df_metrics["dataset"] == base_compare]

    if row_ref.empty or row_cmp.empty:
        raise ValueError(f"Não encontrei as bases '{base_ref}' e/ou '{base_compare}' no DataFrame.")

    row_ref = row_ref.iloc[0]
    row_cmp = row_cmp.iloc[0]

    comparisons = []
    for metric in metrics_to_compare:
        val_ref = row_ref[metric]
        val_cmp = row_cmp[metric]

        gap = abs(val_ref - val_cmp) if pd.notnull(val_ref) and pd.notnull(val_cmp) else np.nan

        comparisons.append({
            "metric": metric,
            f"{base_ref}": val_ref,
            f"{base_compare}": val_cmp,
            "abs_gap": gap,
            "within_limit": gap <= max_gap if pd.notnull(gap) else np.nan
        })

    return pd.DataFrame(comparisons)


"""
Módulo: Análise Comparativa do PSI (Population Stability Index)

Este módulo contém a função `comparative_psi_analysis` que realiza a análise do 
Population Stability Index (PSI) entre diferentes datasets para variáveis 
selecionadas (model_features). A função compara os datasets de forma par a par e 
gera um heatmap para visualização dos resultados.
"""



def comparative_psi_analysis(datasets, model_features, dataset_names=None):
    """
    Realiza a análise comparativa do PSI para variáveis especificadas entre 
    diferentes datasets e gera um heatmap para visualização das diferenças.

    Parameters:
    -----------
    datasets : list of pandas.DataFrame
        Lista de dataframes contendo os dados que serão comparados.
        
    model_features : list of str
        Lista com os nomes das colunas (variáveis) que serão analisadas.
        
    dataset_names : list of str, optional
        Lista de nomes para os datasets. Se None, os nomes serão gerados 
        automaticamente como 'Dataset_1', 'Dataset_2', ... com base na quantidade de datasets.

    Returns:
    --------
    psi_df : pandas.DataFrame
        DataFrame contendo os valores do PSI para cada variável, com cada 
        coluna representando a comparação entre dois datasets.

    Raises:
    -------
    ValueError:
        Se o número de datasets for diferente do número de nomes passados em 
        `dataset_names`.
    """

    def psi(data_ref, data_new, num_bins=10, mode='fixed', eps=1e-5):
        """
        Calcula o Population Stability Index (PSI) entre duas distribuições.

        A função trata tanto variáveis numéricas quanto categóricas. Para variáveis 
        numéricas, o cálculo é feito a partir de discretizações em bins definidos por 
        intervalo fixo ou quantil. Para variáveis categóricas, o cálculo é feito a partir 
        das frequências relativas das categorias.

        Parameters:
        -----------
        data_ref : array-like
            Dados de referência (base) para a comparação.
            
        data_new : array-like
            Novos dados para comparar com a referência.
            
        num_bins : int, default=10
            Número de bins a serem usados para variáveis numéricas no modo 'fixed'.
            
        mode : {'fixed', 'quantile'}, default='fixed'
            Modo de criação dos bins para variáveis numéricas. 'fixed' usa intervalo fixo e 
            'quantile' usa quantis da distribuição de referência.
            
        eps : float, default=1e-5
            Valor pequeno usado para substituir zeros e evitar divisões por zero.

        Returns:
        --------
        psi_value : float
            Valor calculado do PSI para as distribuições comparadas.
        """
        # Verifica se ambas as variáveis são numéricas
        if pd.api.types.is_numeric_dtype(data_ref) and pd.api.types.is_numeric_dtype(data_new):
            data_ref = np.array(data_ref)
            data_new = np.array(data_new)
            
            # Define os limites mínimo e máximo para criar os bins
            min_val = min(np.min(data_ref), np.min(data_new))
            max_val = max(np.max(data_ref), np.max(data_new))
            
            # Se os valores mínimo e máximo forem iguais, não há variação, logo PSI é zero
            if np.isclose(min_val, max_val):
                return 0
            
            # Definir os bins com base no modo especificado
            if mode == 'fixed':
                bins = np.linspace(min_val, max_val, num_bins + 1)
            elif mode == 'quantile':
                bins = np.unique(np.quantile(data_ref, np.linspace(0, 1, num_bins + 1)))
            else:
                raise ValueError('Modo inválido. Utilize "fixed" ou "quantile".')

            # Atribuir cada valor a um bin
            bins_ref = pd.cut(data_ref, bins=bins, labels=False, include_lowest=True)
            bins_new = pd.cut(data_new, bins=bins, labels=False, include_lowest=True)
            
            # Calcula as frequências normalizadas para cada bin
            freq_ref = pd.Series(bins_ref).value_counts(normalize=True, sort=False).fillna(eps)
            freq_new = pd.Series(bins_new).value_counts(normalize=True, sort=False).fillna(eps)
            
            # Garante que todos os bins estejam presentes
            freq_ref = freq_ref.reindex(range(len(bins) - 1), fill_value=eps)
            freq_new = freq_new.reindex(range(len(bins) - 1), fill_value=eps)
            
            # Substitui eventuais zeros pelo valor de eps
            freq_ref = freq_ref.replace(0, eps)
            freq_new = freq_new.replace(0, eps)
            
            # Calcula o PSI para cada bin e retorna a soma dos valores
            psi_values = (freq_new - freq_ref) * np.log(freq_new / freq_ref)
            return psi_values.sum()
        else:
            # Tratamento para variáveis categóricas
            # Obter todas as categorias presentes em ambos os conjuntos
            cat_values = np.union1d(np.unique(data_ref), np.unique(data_new))
            freq_ref = pd.Series(data_ref).value_counts(normalize=True).reindex(cat_values, fill_value=eps)
            freq_new = pd.Series(data_new).value_counts(normalize=True).reindex(cat_values, fill_value=eps)
            
            # Substituir zeros por eps para evitar divisão por zero
            freq_ref = freq_ref.replace(0, eps)
            freq_new = freq_new.replace(0, eps)
            
            # Calcula e retorna o valor do PSI
            psi_values = (freq_new - freq_ref) * np.log(freq_new / freq_ref)
            return psi_values.sum()

    # Se não forem informados nomes para os datasets, gera nomes automáticos
    if dataset_names is None:
        dataset_names = [f'Dataset_{i + 1}' for i in range(len(datasets))]
    
    # Validação: o número de datasets deve ser igual ao número de nomes
    if len(datasets) != len(dataset_names):
        raise ValueError("Number of datasets must match number of dataset names")
    
    psi_results = []  # Lista para armazenar os resultados do PSI por variável
    
    # Realiza comparações par a par entre os datasets para cada variável em model_features
    for var in model_features:
        var_results = {'var': var}  # Dicionário que armazenará os resultados para a variável atual
        for (i, df1), (j, df2) in combinations(enumerate(datasets), 2):
            comparison_name = f'{dataset_names[i]} vs {dataset_names[j]}'
            try:
                # Calcula o PSI para a variável 'var' entre os dois datasets
                psi_value = psi(df1[var], df2[var])
                var_results[comparison_name] = psi_value
            except Exception as e:
                # Em caso de erro, imprime uma mensagem e armazena NaN para a comparação
                print(f"Erro ao calcular PSI para a variável {var} na comparação {comparison_name}: {str(e)}")
                var_results[comparison_name] = np.nan
        psi_results.append(var_results)
    
    # Converte os resultados para um DataFrame do pandas
    psi_df = pd.DataFrame(psi_results)

    # Prepara os dados para gerar um heatmap (valores de PSI por comparação)
    comparison_columns = [col for col in psi_df.columns if ' vs ' in col]
    psi_melted = psi_df.melt(id_vars='var', value_vars=comparison_columns, 
                             var_name='comparison', value_name='psi')

    # Define os intervalos para categorização dos valores do PSI:
    # - Valores menores que 0.1 indicam "Little Shift"
    # - Valores entre 0.1 e 0.2 indicam "Moderate Shift"
    # - Valores maiores ou iguais a 0.2 indicam "Significant Shift"
    shift_bins = [-np.inf, 0.1, 0.2, np.inf]
    
    # Mapeamento para as cores a serem usadas no heatmap
    color_map = {
    0: '#D3D3D3',  # Pouca mudança
    1: '#C0A080',  # Mudança moderada
    2: '#FF6347'   # Mudança significativa
    }
    
    # Reestrutura os dados para criar uma matriz (pivot) para o heatmap, onde as linhas são as variáveis e 
    # as colunas as comparações
    heatmap_data = psi_melted.pivot(index='var', columns='comparison', values='psi')
    
    # Converte os valores de PSI para categorias (0, 1 ou 2) de acordo com os intervalos definidos
    heatmap_data_num = heatmap_data.applymap(lambda x: 0 if x < 0.1 else (1 if x < 0.2 else 2))
    
    # Configurações de estilo para o plot do heatmap
    plt.style.context({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 14})
    plt.figure(figsize=(14, 8))
    
    # Cria o heatmap utilizando o seaborn, utilizando os valores categorizados para definir as cores
    ax = sns.heatmap(
        heatmap_data_num,
        annot=heatmap_data.round(5).astype(str),  # Anotação com valores do PSI formatados
        cmap=list(color_map.values()),
        fmt='',
        linewidths=0.5,
        cbar=False
    )

    # Ajusta os limites do colorbar para que correspondam às categorias
    ax.collections[0].set_clim(-0.5, 2.5)
    
    # Configura título e rótulos dos eixos
    plt.title('Population Stability Index')
    plt.ylabel('Features')
    plt.xlabel(None)
    plt.xticks(rotation=0, ha='center')

    # Cria patches para a legenda com base no mapeamento de cores definido
    patches = [
        mpatches.Patch(color=color_map[0], label='Little Shift'),
        mpatches.Patch(color=color_map[1], label='Moderate Shift'),
        mpatches.Patch(color=color_map[2], label='Significant Shift'),
    ]
    plt.legend(handles=patches, title='PSI', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    # Exibe o heatmap
    plt.show()

    # Imprime os valores do PSI para cada variável no console
    print("\nValores de PSI para cada variável:")
    print(psi_df.to_string(index=False, float_format='%.5f'))
    
    return psi_df

def plot_pdp_all_features(
    model,
    X: pd.DataFrame,
    features: list = None,
    kind: str = "average",
    grid_resolution: int = 30,
    percentiles: tuple = (0.01, 0.99),
    n_cols: int = 2,
    figsize_per_plot: tuple = (7, 4.8),
    target: int = 1,
    categorical_features: list = None,
    rotate_xticks: int = 30,
    suptitle: str = "Partial Dependence Plots (PDP)"
):
    """
    Gera PDP de todas as variáveis, com layout ajustado para melhor visualização.
    Adaptado para classificação binária no seu contexto.

    Parâmetros
    ----------
    model : modelo treinado
    X : pd.DataFrame
        Base no mesmo formato usado no treino.
    features : list
        Lista de features. Se None, usa todas as colunas.
    kind : str
        'average', 'individual' ou 'both'
    grid_resolution : int
        Resolução do grid para PDP
    percentiles : tuple
        Limites percentuais para variáveis contínuas
    n_cols : int
        Número de colunas da grade
    figsize_per_plot : tuple
        Tamanho de cada subplot
    target : int
        Classe positiva
    categorical_features : list
        Features discretas/categóricas
    rotate_xticks : int
        Rotação dos rótulos do eixo x
    suptitle : str
        Título geral da figura
    """

    if features is None:
        features = X.columns.tolist()

    if categorical_features is None:
        categorical_features = []

    n_features = len(features)
    n_rows = math.ceil(n_features / n_cols)

    fig_width = figsize_per_plot[0] * n_cols
    fig_height = figsize_per_plot[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.array(axes).reshape(-1)

    for i, feature in enumerate(features):
        ax = axes[i]
        is_cat = feature in categorical_features

        # Para evitar o erro do sklearn, sempre passamos grid_resolution inteiro
        # Para features categóricas/discretas, usamos percentiles=(0,1)
        PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X,
            features=[feature],
            kind=kind,
            grid_resolution=grid_resolution,
            percentiles=(0, 1) if is_cat else percentiles,
            target=target,
            categorical_features=[feature] if is_cat else None,
            ax=ax
        )

        ax.set_title(f"PDP - {feature}", fontsize=11, pad=10)
        ax.set_xlabel(feature, fontsize=10, labelpad=8)
        ax.set_ylabel("Dependência Parcial", fontsize=10, labelpad=8)

        # Melhorar leitura do eixo x
        ax.tick_params(axis="x", labelrotation=rotate_xticks, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

        # Para variáveis discretas, forçar ticks inteiros se fizer sentido
        if feature in ["hora", "dia_da_semana", "final_de_semana"]:
            unique_vals = sorted(pd.Series(X[feature]).dropna().unique())
            if len(unique_vals) <= 24:
                ax.set_xticks(unique_vals)

        # Grade suave
        ax.grid(True, linestyle="--", alpha=0.25)

    # Remover eixos vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(suptitle, fontsize=16, y=1.02)

    # Espaçamento bem mais folgado para não sobrepor título/eixo
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.94,
        wspace=0.35,
        hspace=0.60
    )

    plt.show()

def plot_ale_multiple_features(
    model,
    X: pd.DataFrame,
    features: list = None,
    bins: int = 20,
    n_cols: int = 2,
    figsize_per_plot: tuple = (7, 4.5),
    center: bool = True,
    include_rug: bool = False,
    suptitle: str = "ALE - Multiple Features"
):
    """
    Gera ALE 1D para várias features em uma grade de subplots.
    """
    if features is None:
        features = X.columns.tolist()

    n_features = len(features)
    n_rows = math.ceil(n_features / n_cols)

    fig_width = figsize_per_plot[0] * n_cols
    fig_height = figsize_per_plot[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.array(axes).reshape(-1)

    for k, feature in enumerate(features):
        ax = axes[k]

        x_values = X[feature].values.astype(float)
        bin_edges = np.unique(np.quantile(x_values, np.linspace(0, 1, bins + 1)))

        if len(bin_edges) < 3:
            ax.set_title(f"{feature}\nSem variabilidade suficiente", fontsize=11)
            ax.axis("off")
            continue

        bins_real = len(bin_edges) - 1
        local_effects = np.zeros(bins_real)
        counts = np.zeros(bins_real, dtype=int)

        for i in range(bins_real):
            left = bin_edges[i]
            right = bin_edges[i + 1]

            if i < bins_real - 1:
                mask = (x_values >= left) & (x_values < right)
            else:
                mask = (x_values >= left) & (x_values <= right)

            idx = np.where(mask)[0]
            counts[i] = len(idx)

            if len(idx) == 0:
                local_effects[i] = 0.0
                continue

            X_low = X.iloc[idx].copy()
            X_high = X.iloc[idx].copy()

            X_low[feature] = left
            X_high[feature] = right

            pred_low = model.predict_proba(X_low)[:, 1]
            pred_high = model.predict_proba(X_high)[:, 1]

            local_effects[i] = np.mean(pred_high - pred_low)

        ale_values = np.cumsum(local_effects)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if center:
            weights = counts / counts.sum()
            ale_values = ale_values - np.sum(ale_values * weights)

        ax.plot(bin_centers, ale_values, marker="o", linewidth=2)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(f"ALE - {feature}", fontsize=11, pad=10)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("ALE", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.25)

        if include_rug:
            y_min, y_max = ax.get_ylim()
            rug_y = np.full_like(x_values, y_min + (y_max - y_min) * 0.02, dtype=float)
            ax.plot(x_values, rug_y, "|", alpha=0.15)

    for j in range(k + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(suptitle, fontsize=15, y=1.02)
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.06,
        top=0.94,
        wspace=0.32,
        hspace=0.55
    )
    plt.show()