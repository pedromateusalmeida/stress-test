import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.8f}'.format)

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
# Utilitários e Estatística
from scipy.stats import chi2_contingency, ks_2samp, shapiro
from sklearn.feature_selection import (mutual_info_classif,
                                       mutual_info_regression)
# Machine Learning - Modelos e Pré-processamento
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_distribution_by_target(
    dataframe: pd.DataFrame,
    target_column: str,
    figsize: tuple = (15, 6),
    bins: int = 30
):
    """
    Gera histogramas de distribuição para todas as variáveis numéricas,
    comparando as distribuições entre classes de uma variável target.

    Parâmetros
    ----------
    dataframe : pd.DataFrame
        Dataset pandas.
    target_column : str
        Nome da coluna target.
    figsize : tuple, default=(15, 6)
        Tamanho da figura.
    bins : int, default=30
        Número de bins dos histogramas.

    Funcionalidades
    ---------------
    - Valida se a coluna target existe.
    - Emite aviso se target tiver mais de 2 classes.
    - Identifica automaticamente apenas colunas numéricas.
    - Exclui a própria coluna target das variáveis a plotar.
    - Remove NaN antes de plotar.
    - Cria 2 subplots lado a lado (um para cada classe).
    - Adiciona percentual acima de cada barra.
    - Adiciona linhas de média e mediana.
    - Mostra n no título de cada subplot.
    - Usa escala linear nos eixos x e y.
    - Adiciona grade no eixo y.
    - Usa bordas pretas finas nas barras.
    - Trata erros por coluna com try/except.
    """

    sns.set_style("whitegrid")

    # -------------------------
    # Validações iniciais
    # -------------------------
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("O parâmetro 'dataframe' deve ser um pandas DataFrame.")

    if target_column not in dataframe.columns:
        raise ValueError(f"A coluna target '{target_column}' não existe no DataFrame.")

    df = dataframe.copy()

    target_classes = df[target_column].dropna().unique().tolist()

    if len(target_classes) == 0:
        raise ValueError("A coluna target não possui valores válidos.")

    try:
        target_classes = sorted(target_classes)
    except Exception:
        pass

    if len(target_classes) > 2:
        print("⚠️ Aviso: a função foi otimizada para classificação binária, mas o target possui mais de 2 classes.")
        print(f"📌 Classes encontradas: {target_classes}")

    # -------------------------
    # Seleção de colunas numéricas
    # -------------------------
    numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_column]

    print(f"📊 Total de colunas numéricas identificadas: {len(numeric_cols)}")

    if not numeric_cols:
        print("✗ Nenhuma coluna numérica encontrada para processamento.")
        return

    colors = ["skyblue", "lightcoral", "lightgreen", "gold"]

    processed = 0
    errors = 0

    # -------------------------
    # Loop pelas variáveis numéricas
    # -------------------------
    for col in numeric_cols:
        try:
            print("\n" + "=" * 90)
            print(f"📌 Processando coluna: {col}")

            n_classes = len(target_classes)
            fig, axes = plt.subplots(1, 2, figsize=figsize)

            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]

            fig.suptitle(f"Distribuição de '{col}' por classe de '{target_column}'", fontsize=14, y=1.03)

            for i in range(2):
                ax = axes[i]

                if i < n_classes:
                    cls = target_classes[i]

                    values = df.loc[df[target_column] == cls, col].dropna()

                    n_valid = len(values)

                    if n_valid == 0:
                        ax.text(
                            0.5, 0.5,
                            "Sem valores válidos",
                            ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=12
                        )
                        ax.set_title(f"{target_column} = {cls} | n = 0")
                        ax.set_xlabel(col)
                        ax.set_ylabel("Frequência")
                        ax.set_xscale("linear")
                        ax.set_yscale("linear")
                        continue

                    counts, bin_edges, patches = ax.hist(
                        values,
                        bins=bins,
                        color=colors[i % len(colors)],
                        edgecolor="black",
                        linewidth=0.7
                    )

                    # labels com percentual acima das barras
                    total = counts.sum()
                    ymax = counts.max() if len(counts) > 0 else 0

                    for count, left, right in zip(counts, bin_edges[:-1], bin_edges[1:]):
                        if count > 0:
                            x = (left + right) / 2
                            pct = (count / total) * 100 if total > 0 else 0
                            ax.text(
                                x,
                                count,
                                f"{pct:.1f}%",
                                ha="center",
                                va="bottom",
                                rotation=45,
                                fontsize=8
                            )

                    # média e mediana
                    mean_val = values.mean()
                    median_val = values.median()

                    ax.axvline(
                        mean_val,
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                        label=f"Média = {mean_val:.2f}"
                    )
                    ax.axvline(
                        median_val,
                        color="orange",
                        linestyle="--",
                        linewidth=1.5,
                        label=f"Mediana = {median_val:.2f}"
                    )

                    ax.set_title(f"{target_column} = {cls} | n = {n_valid}")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequência")

                    ax.set_xscale("linear")
                    ax.set_yscale("linear")

                    ax.grid(axis="y", alpha=0.3)
                    ax.legend()

                    print(f"✓ Classe '{cls}' processada com {n_valid} registros válidos")

                else:
                    ax.axis("off")
                    ax.set_visible(False)

            # se houver apenas uma classe, oculta segundo subplot
            if len(target_classes) == 1:
                axes[1].axis("off")
                axes[1].set_visible(False)

            plt.tight_layout()
            plt.show()

            print(f"✓ Gráfico gerado com sucesso para: {col}")
            processed += 1

        except TypeError as e:
            print(f"✗ TypeError ao processar coluna '{col}': {e}")
            errors += 1
        except Exception as e:
            print(f"✗ Erro ao processar coluna '{col}': {e}")
            errors += 1

    # -------------------------
    # Resumo final
    # -------------------------
    print("\n" + "=" * 90)
    print("✅ Processamento concluído.")
    print(f"📊 Colunas processadas com sucesso: {processed}")
    print(f"✗ Colunas com erro: {errors}")


def analisar_variaveis_por_target(
    dataframe: pd.DataFrame,
    target_column: str,
    figsize: tuple = (15, 6),
    max_categories: int = 15,
    min_frequency: int = 5
):
    """
    Analisa variáveis categóricas e numéricas de um DataFrame em relação a uma variável target.

    Funcionalidades:
    ---------------
    1. Valida se a coluna target existe.
    2. Identifica automaticamente:
       - colunas categóricas: tipo object, category ou numéricas com <= 20 valores únicos
       - colunas numéricas: numéricas com > 20 valores únicos
    3. Para colunas categóricas:
       - cria gráficos de barras comparando distribuições entre classes do target
       - agrupa categorias raras em 'Outros'
       - limita o número de categorias exibidas
       - ordena barras por frequência decrescente
    4. Para colunas numéricas:
       - imprime estatísticas detalhadas no console por classe
       - gera boxplot por variável numérica
       - inclui caixa de texto com n, média (μ) e desvio padrão (σ) por classe
    5. Usa tratamento de erros com try/except para ignorar colunas problemáticas.

    Parâmetros:
    -----------
    dataframe : pd.DataFrame
        Dataset pandas.
    target_column : str
        Nome da coluna target.
    figsize : tuple, default=(15, 6)
        Tamanho das figuras.
    max_categories : int, default=15
        Número máximo de categorias exibidas separadamente.
    min_frequency : int, default=5
        Frequência mínima para uma categoria ser exibida separadamente.
    """

    # ----------------------------
    # Validação inicial
    # ----------------------------
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("O parâmetro 'dataframe' deve ser um pandas DataFrame.")

    if target_column not in dataframe.columns:
        raise ValueError(f"A coluna target '{target_column}' não existe no DataFrame.")

    sns.set_style("whitegrid")

    df = dataframe.copy()

    print("\n📊 Iniciando análise das variáveis...\n")

    # Classes do target
    target_classes = df[target_column].dropna().unique()
    target_classes = sorted(target_classes)

    if len(target_classes) < 2:
        print("✗ A variável target precisa ter pelo menos 2 classes para comparação.")
        return

    # Paleta solicitada
    colors = ["skyblue", "lightcoral", "lightgreen", "gold"]

    # ----------------------------
    # Identificação automática
    # ----------------------------
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        if col == target_column:
            continue

        try:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                nunique = df[col].nunique(dropna=True)
                if nunique <= 20:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
        except TypeError:
            print(f"✗ Coluna ignorada por erro de tipo: {col}")

    # ----------------------------
    # Função auxiliar categórica
    # ----------------------------
    def agrupar_categorias_rare(series, max_categories=15, min_frequency=5):
        s = series.astype("object").fillna("NaN").copy()

        freq = s.value_counts(dropna=False)

        categorias_validas = freq[freq >= min_frequency].index.tolist()

        if len(categorias_validas) > max_categories:
            categorias_validas = freq.head(max_categories).index.tolist()

        s = s.apply(lambda x: x if x in categorias_validas else "Outros")

        ordem_final = s.value_counts().sort_values(ascending=False).index.tolist()
        return s, ordem_final

    # ----------------------------
    # 1) ANÁLISE DAS CATEGÓRICAS
    # ----------------------------
    print(f"📊 {len(categorical_cols)} coluna(s) categórica(s) identificada(s).")

    for col in categorical_cols:
        try:
            serie_tratada, ordem = agrupar_categorias_rare(
                df[col],
                max_categories=max_categories,
                min_frequency=min_frequency
            )

            df_plot = df[[target_column]].copy()
            df_plot[col] = serie_tratada

            n_classes = len(target_classes)
            fig, axes = plt.subplots(
                1, n_classes,
                figsize=(max(figsize[0], 6 * n_classes), figsize[1]),
                squeeze=False
            )
            axes = axes.flatten()

            fig.suptitle(f"Distribuição de '{col}' por classe de '{target_column}'", fontsize=14, y=1.02)

            for i, cls in enumerate(target_classes):
                ax = axes[i]
                subset = df_plot[df_plot[target_column] == cls][col]

                counts = subset.value_counts().reindex(ordem, fill_value=0)

                sns.barplot(
                    x=counts.values,
                    y=counts.index,
                    ax=ax,
                    color=colors[i % len(colors)]
                )

                ax.set_title(f"{target_column} = {cls}")
                ax.set_xlabel("Frequência")
                ax.set_ylabel(col)

                for j, v in enumerate(counts.values):
                    ax.text(v + max(counts.values) * 0.01 if max(counts.values) > 0 else 0.1,
                            j,
                            str(v),
                            va="center",
                            fontsize=9)

            plt.tight_layout()
            plt.show()

            print(f"✓ Gráfico categórico gerado para: {col}")

        except TypeError:
            print(f"✗ Coluna categórica ignorada por TypeError: {col}")
        except Exception as e:
            print(f"✗ Erro ao processar coluna categórica '{col}': {e}")

    # ----------------------------
    # 2) ANÁLISE DAS NUMÉRICAS
    # ----------------------------
    print(f"\n📊 {len(numeric_cols)} coluna(s) numérica(s) identificada(s).\n")

    for col in numeric_cols:
        try:
            print(f"\n{'=' * 80}")
            print(f"📊 Estatísticas detalhadas para a variável numérica: {col}")
            print(f"{'=' * 80}")

            resumo_texto = []

            for i, cls in enumerate(target_classes):
                serie = df.loc[df[target_column] == cls, col].dropna()

                if len(serie) == 0:
                    print(f"✗ Classe '{cls}' sem valores válidos em '{col}'.")
                    resumo_texto.append(f"Classe {cls}\nn=0\nμ=NA\nσ=NA")
                    continue

                q1 = serie.quantile(0.25)
                q3 = serie.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = ((serie < lower) | (serie > upper)).sum()

                n = len(serie)
                media = serie.mean()
                mediana = serie.median()
                desvio = serie.std()

                print(f"\nClasse: {cls}")
                print(f"✓ n = {n}")
                print(f"✓ média = {media:.4f}")
                print(f"✓ mediana = {mediana:.4f}")
                print(f"✓ Q1 = {q1:.4f}")
                print(f"✓ Q3 = {q3:.4f}")
                print(f"✓ IQR = {iqr:.4f}")
                print(f"✓ outliers = {outliers}")

                resumo_texto.append(
                    f"Classe {cls}\n"
                    f"n={n}\n"
                    f"μ={media:.2f}\n"
                    f"σ={desvio:.2f}"
                )

            # Gráfico para variável numérica
            plt.figure(figsize=figsize)
            ax = sns.boxplot(
                data=df,
                x=target_column,
                y=col,
                palette=colors[:len(target_classes)]
            )

            ax.set_title(f"Distribuição de '{col}' por classe de '{target_column}'")
            ax.set_xlabel(target_column)
            ax.set_ylabel(col)

            texto_final = "\n\n".join(resumo_texto)
            ax.text(
                1.02, 0.98,
                texto_final,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            plt.tight_layout()
            plt.show()

            print(f"✓ Gráfico numérico gerado para: {col}")

        except Exception as e:
            print(f"✗ Erro ao processar coluna numérica '{col}': {e}")

    print("\n✓ Análise concluída com sucesso.")

def plot_categorical_distributions_by_target(
    dataframe: pd.DataFrame,
    target_column: str,
    figsize: tuple = (15, 6),
    max_categories: int = 15,
    min_frequency: int = 5
):
    """
    Gera gráficos de barras para todas as variáveis categóricas,
    comparando suas distribuições entre classes de uma variável target.

    Parâmetros
    ----------
    dataframe : pd.DataFrame
        Dataset pandas.
    target_column : str
        Nome da coluna target.
    figsize : tuple, default=(15, 6)
        Tamanho da figura.
    max_categories : int, default=15
        Número máximo de categorias exibidas. As demais são agrupadas em 'Outros'.
    min_frequency : int, default=5
        Frequência mínima para uma categoria ser exibida separadamente.

    Funcionalidades
    ---------------
    - Valida se a coluna target existe.
    - Identifica automaticamente colunas categóricas:
      object, category ou numéricas com <= 20 valores únicos.
    - Ignora colunas problemáticas com try/except.
    - Para cada variável categórica:
        * cria 2 subplots lado a lado (um para cada classe do target)
        * usa mesmas categorias em ambos os subplots
        * agrupa categorias raras em 'Outros'
        * ordena barras por frequência decrescente
        * adiciona labels com percentual e contagem
        * trunca nomes longos (>15 caracteres)
        * rotaciona labels em 45 graus
        * adiciona caixa de texto com a categoria mais frequente
        * mostra n no título de cada subplot
        * usa escala linear
    - Se houver apenas uma classe, oculta o segundo subplot.
    - Imprime no console resumo por coluna.
    """

    sns.set_style("whitegrid")

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("O parâmetro 'dataframe' deve ser um pandas DataFrame.")

    if target_column not in dataframe.columns:
        raise ValueError(f"A coluna target '{target_column}' não existe no DataFrame.")

    df = dataframe.copy()

    # Identifica classes do target
    target_classes = df[target_column].dropna().unique().tolist()

    if len(target_classes) == 0:
        raise ValueError("A coluna target não possui valores válidos.")

    # Mantém ordem estável
    try:
        target_classes = sorted(target_classes)
    except Exception:
        pass

    # Identificar colunas categóricas
    categorical_cols = []

    for col in df.columns:
        if col == target_column:
            continue

        try:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) <= 20:
                categorical_cols.append(col)
        except TypeError:
            print(f"✗ Coluna ignorada por TypeError: {col}")

    if not categorical_cols:
        print("✗ Nenhuma coluna categórica encontrada.")
        return

    print(f"📊 Total de colunas categóricas identificadas: {len(categorical_cols)}")

    colors = ["skyblue", "lightcoral", "lightgreen", "gold"]

    def truncate_label(label, max_len=15):
        label = str(label)
        return label if len(label) <= max_len else label[:max_len] + "..."

    def prepare_categories(series, max_categories=15, min_frequency=5):
        """
        Agrupa categorias raras em 'Outros' considerando:
        - frequência mínima
        - limite máximo de categorias
        """
        s = series.copy()

        # tenta evitar problemas com unhashable
        s = s.apply(lambda x: tuple(x) if isinstance(x, list) else x)
        s = s.astype("object").fillna("NaN")

        counts = s.value_counts(dropna=False)

        # categorias com frequência mínima
        valid_categories = counts[counts >= min_frequency].index.tolist()

        # limita número máximo
        if len(valid_categories) > max_categories:
            valid_categories = counts.loc[valid_categories].head(max_categories).index.tolist()

        s_grouped = s.apply(lambda x: x if x in valid_categories else "Outros")

        final_order = s_grouped.value_counts().sort_values(ascending=False).index.tolist()
        return s_grouped, final_order

    for col in categorical_cols:
        try:
            print("\n" + "=" * 90)
            print(f"📌 Processando coluna: {col}")

            col_series, categories_order = prepare_categories(
                df[col],
                max_categories=max_categories,
                min_frequency=min_frequency
            )

            plot_df = df[[target_column]].copy()
            plot_df[col] = col_series

            # Garantir mesmas categorias em ambos os subplots
            all_categories = categories_order

            n_classes = len(target_classes)
            n_subplots = 2 if n_classes >= 2 else 2  # cria 2 e oculta o segundo se necessário

            fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
            if n_subplots == 1:
                axes = [axes]

            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]

            fig.suptitle(
                f"Distribuição da variável categórica '{col}' por classe de '{target_column}'",
                fontsize=14,
                y=1.03
            )

            # resumo no console
            unique_categories = plot_df[col].nunique(dropna=False)
            print(f"📎 Número de categorias únicas após agrupamento: {unique_categories}")

            for i in range(2):
                ax = axes[i]

                if i < n_classes:
                    cls = target_classes[i]
                    subset = plot_df[plot_df[target_column] == cls][col]
                    counts = subset.value_counts().reindex(all_categories, fill_value=0)
                    total_n = counts.sum()

                    bars = ax.bar(
                        x=np.arange(len(counts)),
                        height=counts.values,
                        color=colors[i % len(colors)]
                    )

                    ax.set_title(f"{target_column} = {cls} | n = {total_n}")
                    ax.set_ylabel("Frequência")
                    ax.set_xlabel("Categoria")
                    ax.set_yscale("linear")

                    truncated_labels = [truncate_label(cat) for cat in counts.index]
                    ax.set_xticks(np.arange(len(counts)))
                    ax.set_xticklabels(truncated_labels, rotation=45, ha="right")

                    # labels nas barras: percentual e contagem
                    for bar, count in zip(bars, counts.values):
                        pct = (count / total_n * 100) if total_n > 0 else 0
                        label = f"{count}\n({pct:.1f}%)"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            label,
                            ha="center",
                            va="bottom",
                            fontsize=8
                        )

                    # categoria mais frequente
                    if total_n > 0:
                        top_category = counts.idxmax()
                        top_count = counts.max()
                        top_pct = (top_count / total_n) * 100

                        textbox = (
                            f"Mais frequente:\n"
                            f"{truncate_label(top_category, 20)}\n"
                            f"{top_pct:.1f}%"
                        )

                        ax.text(
                            0.98,
                            0.98,
                            textbox,
                            transform=ax.transAxes,
                            ha="right",
                            va="top",
                            fontsize=9,
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
                        )

                    print(f"✓ Classe '{cls}': {total_n} registros")

                else:
                    ax.axis("off")
                    ax.set_visible(False)

            plt.tight_layout()
            plt.show()

            print(f"✓ Gráfico gerado com sucesso para: {col}")

        except TypeError as e:
            print(f"✗ TypeError ao processar coluna '{col}': {e}")
        except Exception as e:
            print(f"✗ Erro ao processar coluna '{col}': {e}")

    print("\n✅ Processamento concluído.")

import numpy as np
import pandas as pd


def calculate_vif(dataframe, features):
    """
    Calcula o VIF para um conjunto de features.
    
    Parameters:
        dataframe (pd.DataFrame): dataset
        features (list): lista de colunas numéricas
        
    Returns:
        pd.DataFrame: VIF de cada variável
    """
    
    X = dataframe[features].copy()
    
    # Remove valores nulos
    X = X.dropna()
    
    # Adiciona constante (importante para regressão)
    X["intercept"] = 1
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    
    # Remove intercepto
    vif_data = vif_data[vif_data["feature"] != "intercept"]
    
    return vif_data.sort_values(by="VIF", ascending=False)



def plot_vif(vif_df, figsize=(10, 6)):
    """
    Plota o VIF com linhas críticas (5 e 10)
    """
    
    plt.figure(figsize=figsize)
    
    sns.barplot(
        data=vif_df,
        x="VIF",
        y="feature"
    )
    
    # Linhas críticas
    plt.axvline(x=5, color="orange", linestyle="--", label="VIF = 5 (atenção)")
    plt.axvline(x=10, color="red", linestyle="--", label="VIF = 10 (crítico)")
    
    plt.title("Variance Inflation Factor (VIF)")
    plt.xlabel("VIF")
    plt.ylabel("Variáveis")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def variaveis_normalidade_shapiro(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Aplica o teste de Shapiro-Wilk para verificar a normalidade em cada variável numérica do DataFrame.
    Retorna um DataFrame contendo as variáveis testadas com os valores do teste (statistic), os p-valores
    e uma indicação de normalidade ("Normal" se o p-valor for >= alpha, "Não Normal" caso contrário).
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame que contém os dados.
    alpha : float, opcional
        Nível de significância (padrão 0.05). Variáveis com p-valor inferior a alpha serão consideradas
        significativamente diferentes de uma distribuição normal.
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame com as colunas:
          - 'variável': nome da variável numérica.
          - 'statistic': valor do teste de Shapiro-Wilk.
          - 'p_value': p‑valor obtido.
          - 'normalidade': "Normal" se p_value >= alpha, ou "Não Normal" se p_value < alpha.
        O DataFrame é ordenado em ordem crescente de p‑valor.
        
    Exceções:
    -----------
    ValueError:
        Se não houver variáveis numéricas no DataFrame para o teste.
    """
    # Seleciona apenas as colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        raise ValueError("Não há variáveis numéricas no DataFrame para realizar o teste de Shapiro-Wilk.")
    
    resultados = []
    for variavel in numeric_cols.columns:
        try:
            # Remove valores ausentes
            dados = numeric_cols[variavel].dropna()
            # O teste de Shapiro-Wilk requer pelo menos 3 observações
            if len(dados) < 3:
                print(f"Variável '{variavel}' possui menos de 3 observações; teste não aplicado.")
                continue
            stat, p = shapiro(dados)
            # Define a normalidade com base no p-valor comparado ao nível de significância
            normalidade = "Normal" if p >= alpha else "Não Normal"
            resultados.append({'variável': variavel, 'statistic': stat, 'p_value': p, 'normalidade': normalidade})
        except Exception as e:
            print(f"Erro ao processar '{variavel}': {e}")

    return pd.DataFrame(resultados).sort_values(by='p_value').reset_index(drop=True)

def variaveis_significativas_chi2(df: pd.DataFrame, y: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Realiza o teste de qui-quadrado para verificar a associação entre variáveis categóricas 
    e a variável dependente. Retorna um DataFrame contendo apenas as variáveis que são 
    significativamente associadas à variável dependente (p‑valor < alpha), junto com seus p‑valores.

    Parâmetros:
    -----------
    df : pd.DataFrame
        O DataFrame que contém os dados.
    y : str
        O nome da variável dependente (target) no DataFrame.
    alpha : float, opcional
        O nível de significância (padrão é 0.05). Variáveis com p‑valor menor que alpha serão consideradas significativas.

    Retorna:
    --------
    pd.DataFrame
        Um DataFrame com duas colunas:
        — variável: nome da variável categórica significativa  
        — p_value: p‑valor do teste qui-quadrado  
        Ordenado em ordem crescente de p‑valor. Se não houver variáveis significativas, retorna um DataFrame vazio.

    Exceções:
    ---------
    ValueError:
        — Se a variável dependente não estiver presente no DataFrame.  
        — Se não houver variáveis categóricas para testar no DataFrame.
    """
    # Validação de inputs
    if y not in df.columns:
        raise ValueError(f"A variável dependente '{y}' não foi encontrada no DataFrame.")

    # Selecionar apenas colunas categóricas, excluindo a variável dependente
    colunas_categ = df.select_dtypes(include=['object']).drop(columns=[y], errors='ignore')
    if colunas_categ.empty:
        raise ValueError("Não há variáveis categóricas no DataFrame para realizar o teste de qui-quadrado.")

    # Inicializar lista de resultados
    resultados = []

    # Loop pelas colunas categóricas e executar o teste qui-quadrado
    for coluna in colunas_categ.columns:
        try:
            tabela = pd.crosstab(df[coluna], df[y])
            _, p_value, _, _ = chi2_contingency(tabela)

            # Adicionar somente se o p-valor for menor que o nível de significância
            if p_value < alpha:
                resultados.append({'variável': coluna, 'p_value': p_value})
        except Exception as e:
            print(f"Erro ao processar '{coluna}': {e}")

    # Retornar DataFrame ordenado por p-valor
    return pd.DataFrame(resultados).sort_values(by='p_value').reset_index(drop=True)



def teste_ks_variaveis(
    df: pd.DataFrame,
    y: str,
    grupo1_nome,
    grupo2_nome,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Realiza o teste Kolmogorov‑Smirnov (KS) para verificar diferenças significativas na distribuição
    de variáveis numéricas entre dois grupos categóricos. Retorna um DataFrame contendo apenas as variáveis 
    numéricas que apresentam diferenças significativas (p‑valor < alpha), junto com seus estatísticos e p‑valores.

    Parâmetros:
    -----------
    df : pd.DataFrame
        O DataFrame que contém os dados.
    y : str
        O nome da variável categórica que contém os grupos (ex.: 'status_do_voo').
    grupo1_nome : str
        O valor da variável categórica correspondente ao primeiro grupo (ex.: 'Pontual').
    grupo2_nome : str
        O valor da variável categórica correspondente ao segundo grupo (ex.: 'Atrasado').
    alpha : float, opcional
        O nível de significância (padrão é 0.05). Variáveis com p‑valor menor que alpha serão consideradas significativas.

    Retorna:
    --------
    pd.DataFrame
        Um DataFrame com três colunas:
        — variável: nome da variável numérica significativa  
        — statistic: valor do teste KS  
        — p_value: p‑valor do teste  
        Ordenado em ordem crescente de p‑valor. Se não houver variáveis significativas, retorna um DataFrame vazio.

    Exceções:
    ---------
    ValueError:
        — Se a variável categórica y não estiver presente no DataFrame.  
        — Se não houver colunas numéricas no DataFrame.
    """
    if y not in df.columns:
        raise ValueError(f"A variável '{y}' não foi encontrada no DataFrame.")

    df_ = df.copy()

    colunas_numericas = df_.select_dtypes(include=["number"]).columns.tolist()
    if y in colunas_numericas:
        colunas_numericas.remove(y)

    if not colunas_numericas:
        raise ValueError("Não há colunas numéricas disponíveis para o teste KS.")

    for col in colunas_numericas:
        df_[col] = df_[col].fillna(df_[col].median())

    grupo1_df = df_[df_[y] == grupo1_nome]
    grupo2_df = df_[df_[y] == grupo2_nome]

    if grupo1_df.empty or grupo2_df.empty:
        raise ValueError("Um dos grupos está vazio. Verifique os valores informados.")

    resultados = []

    for coluna in colunas_numericas:
        x1 = grupo1_df[coluna].dropna()
        x2 = grupo2_df[coluna].dropna()

        if len(x1) >= 2 and len(x2) >= 2:
            stat, p_value = ks_2samp(x1, x2)
            resultados.append({
                "variavel": coluna,
                "statistic": stat,
                "p_value": p_value,
                "significativa": p_value < alpha
            })

    return (
        pd.DataFrame(resultados)
        .sort_values(by="p_value", ascending=True)
        .reset_index(drop=True)
    )


def variaveis_significativas_mi(
    df: pd.DataFrame,
    y: str,
    problem_type: str = 'classification',
    n_features: int = None
) -> pd.DataFrame:
    """
    Calcula a Informação Mútua entre variáveis preditoras e a variável alvo,
    retornando um DataFrame com as variáveis ordenadas pelo score de MI.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    y : str
        Nome da variável dependente (target) no DataFrame.
    problem_type : str, opcional
        Tipo de problema: 'classification' ou 'regression'. Padrão é 'classification'.
    n_features : int, opcional
        Número de features mais importantes a retornar. Se None, retorna todas.

    Retorna:
    --------
    pd.DataFrame
        DataFrame com duas colunas:
        — variável: nome da variável preditora  
        — score_mi: score de Informação Mútua  
        Ordenado do maior para o menor score. Se n_features for especificado,
        retorna apenas as top n_features.

    Exceções:
    ---------
    ValueError:
        — Se a variável alvo não existir no DataFrame.  
        — Se problem_type não for 'classification' nem 'regression'.
    """
    # Validação de inputs
    if y not in df.columns:
        raise ValueError(f"A variável dependente '{y}' não foi encontrada no DataFrame.")

    # Separar preditoras (X) e alvo (y)
    X = df.drop(columns=[y])
    y_data = df[y]

    # Codificar variáveis categóricas em X
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object', 'category']):
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))

    # Codificar variável alvo e calcular MI conforme tipo de problema
    if problem_type == 'classification':
        y_encoded = LabelEncoder().fit_transform(y_data.astype(str)) \
            if y_data.dtype == 'object' or y_data.dtype.name == 'category' else y_data
        mi_scores = mutual_info_classif(X_encoded, y_encoded, discrete_features='auto')
    elif problem_type == 'regression':
        y_encoded = y_data
        mi_scores = mutual_info_regression(X_encoded, y_encoded, discrete_features='auto')
    else:
        raise ValueError("O parâmetro 'problem_type' deve ser 'classification' ou 'regression'.")

    # Construir DataFrame de resultados ordenado pelo score
    mi_df = pd.DataFrame({
        'variável': X.columns,
        'score_mi': mi_scores
    }).sort_values(by='score_mi', ascending=False).reset_index(drop=True)

    # Retornar somente as top n_features, se especificado
    return mi_df.head(n_features) if n_features else mi_df

def variaveis_significativas_information_value(
    df: pd.DataFrame, 
    y: str, 
    bins: int = 10, 
    show_woe: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Calcula o Information Value (IV) e opcionalmente o Weight of Evidence (WoE) para cada variável preditora.
    """
    # -----------------------
    # Validação rigorosa do Input
    # -----------------------
    if y not in df.columns:
        raise ValueError(f"A variável dependente '{y}' não foi encontrada no DataFrame.")
    
    df_copy = df.copy()
    
    # Remover linhas onde o próprio target é nulo (não servem para treinar/avaliar)
    df_copy = df_copy.dropna(subset=[y])
    
    unique_targets = df_copy[y].unique()
    if not set(unique_targets).issubset({0, 1}):
        raise ValueError("A variável alvo deve conter estritamente os valores numéricos 0 e 1. Converta-a antes de usar a função para garantir controle sobre o evento.")

    # -----------------------
    # Totais Globais
    # -----------------------
    total_eventos = df_copy[y].sum()
    total_nao_eventos = len(df_copy) - total_eventos

    iv_list = []
    woe_dict = {}

    # -----------------------
    # Loop de Variáveis
    # -----------------------
    for coluna in df_copy.drop(columns=[y]).columns:
        x = df_copy[coluna]

        # Lidar com nulos explicitamente criando uma máscara
        is_null = x.isnull()

        # Discretizar numéricas vs categóricas
        if pd.api.types.is_numeric_dtype(x):
            # Cria a série com o mesmo índice, inicializando com strings vazias
            x_binned = pd.Series(index=x.index, dtype=str)
            try:
                x_binned[~is_null] = pd.qcut(x[~is_null], q=bins, duplicates='drop').astype(str)
            except ValueError:
                x_binned[~is_null] = pd.cut(x[~is_null], bins=bins).astype(str)
        else:
            x_binned = x.astype(str)

        # Preencher os nulos com a categoria "Missing"
        x_binned[is_null] = "Missing"

        # Construir tabela agregada (dropna=False garante que tudo entre na conta)
        agg = pd.DataFrame({
            'total': x_binned.value_counts(dropna=False),
            'eventos': df_copy.groupby(x_binned, dropna=False)[y].sum()
        }).fillna(0)
        
        agg['nao_eventos'] = agg['total'] - agg['eventos']

        # -----------------------
        # Tratamento de Zeros (Additive Smoothing)
        # -----------------------
        # Adiciona 0.5 em bins que possuem 0 eventos ou 0 não eventos
        zero_mask = (agg['eventos'] == 0) | (agg['nao_eventos'] == 0)
        if zero_mask.any():
            agg.loc[zero_mask, 'eventos'] += 0.5
            agg.loc[zero_mask, 'nao_eventos'] += 0.5

        # -----------------------
        # Proporções, WoE e IV
        # -----------------------
        agg['prop_eventos'] = agg['eventos'] / total_eventos
        agg['prop_nao_eventos'] = agg['nao_eventos'] / total_nao_eventos

        agg['woe'] = np.log(agg['prop_eventos'] / agg['prop_nao_eventos'])
        agg['iv'] = (agg['prop_eventos'] - agg['prop_nao_eventos']) * agg['woe']
        
        iv_value = agg['iv'].sum()

        # Armazenar resultados
        iv_list.append({'variável': coluna, 'information_value': iv_value})
        
        if show_woe:
            woe_dict[coluna] = agg.reset_index().rename(columns={'index': 'bin'})

    # Construir DataFrame final
    iv_df = pd.DataFrame(iv_list).sort_values(by='information_value', ascending=False).reset_index(drop=True)

    return (iv_df, woe_dict) if show_woe else iv_df

def outlier_analysis_all_features(
    df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    bins: int = 30,
    figsize=(14, 5),
    plot: bool = True,
    max_plots: int = None  # limitar plots se dataset for grande
):
    """
    Análise univariada de outliers para todas as variáveis numéricas.

    Outputs:
    - DataFrame consolidado com métricas
    - Gráficos opcionais por variável
    """

    results = []

    # Apenas numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns

    for i, col in enumerate(numeric_cols):

        data = df[col].dropna()

        if len(data) == 0:
            continue

        # Percentis
        p_low = data.quantile(lower_q)
        p_high = data.quantile(upper_q)

        # Outliers
        out_low = data[data < p_low]
        out_high = data[data > p_high]

        total = len(data)

        # Métricas
        result = {
            "feature": col,
            "p1": p_low,
            "p99": p_high,
            "mean": data.mean(),
            "median": data.median(),
            "std": data.std(),
            "n_outliers_low": len(out_low),
            "n_outliers_high": len(out_high),
            "pct_outliers_low": len(out_low) / total,
            "pct_outliers_high": len(out_high) / total,
            "pct_outliers_total": (len(out_low) + len(out_high)) / total,
            "n_unique": data.nunique()
        }

        results.append(result)

        # -------------------------
        # PLOTS
        # -------------------------
        if plot:

            if max_plots and i >= max_plots:
                continue

            fig, axes = plt.subplots(1, 2, figsize=figsize)

            # Histograma
            sns.histplot(data, bins=bins, ax=axes[0])

            axes[0].axvline(p_low, linestyle='--', label=f"P{int(lower_q*100)}")
            axes[0].axvline(p_high, linestyle='--', label=f"P{int(upper_q*100)}")

            axes[0].axvline(result["mean"], linestyle='-', label="Mean")
            axes[0].axvline(result["median"], linestyle='-', label="Median")

            axes[0].set_title(f"{col} - Distribuição")
            axes[0].legend()

            # Boxplot
            sns.boxplot(x=data, ax=axes[1])

            axes[1].axvline(p_low, linestyle='--')
            axes[1].axvline(p_high, linestyle='--')

            axes[1].set_title(f"{col} - Boxplot")

            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(results)

    return results_df.sort_values(by="pct_outliers_total", ascending=False)

def outlier_analysis_dataset(
    df: pd.DataFrame,
    z_thresh: float = 3.0,
    iqr_factor: float = 1.5,
    lower_q: float = 0.01,
    upper_q: float = 0.99
):
    """
    Aplica comparação de métodos de outlier para todas as features numéricas
    """

    results = []

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        data = df[col].dropna()

        if len(data) == 0:
            continue

        # Percentil
        p_low = data.quantile(lower_q)
        p_high = data.quantile(upper_q)
        out_pct = data[(data < p_low) | (data > p_high)]

        # Z-score
        mean = data.mean()
        std = data.std()

        if std == 0:
            continue  # evita divisão por zero

        z = (data - mean) / std
        out_z = data[np.abs(z) > z_thresh]

        # IQR
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        out_iqr = data[(data < lower) | (data > upper)]

        results.append({
            "feature": col,

            "pct_outliers_percentile": len(out_pct) / len(data),
            "pct_outliers_zscore": len(out_z) / len(data),
            "pct_outliers_iqr": len(out_iqr) / len(data),

            "n_outliers_percentile": len(out_pct),
            "n_outliers_zscore": len(out_z),
            "n_outliers_iqr": len(out_iqr),

            "mean": mean,
            "std": std,
            "q1": q1,
            "q3": q3
        })

    results_df = pd.DataFrame(results)

    return results_df.sort_values(
        by="pct_outliers_percentile",
        ascending=False
    )

def detect_outliers_gaussian_visual(
    df: pd.DataFrame,
    features: list = None,
    threshold: float = 3,
    bins: int = 30,
    figsize=(14, 5),
    title_fontsize=14,
    tick_fontsize=10,
    show_boxplot: bool = True,
    max_plots: int = None
):
    """
    Detecta outliers via Gaussian (mean ± k*std) com visualizações ricas
    """

    results = []

    if features is None:
        features = df.select_dtypes(include=np.number).columns.tolist()

    for i, feature in enumerate(features):

        data = df[feature].dropna()

        if len(data) == 0:
            continue

        mean = data.mean()
        std = data.std()

        if std == 0:
            continue

        lower = mean - threshold * std
        upper = mean + threshold * std

        outliers = data[(data < lower) | (data > upper)]

        pct_outliers = len(outliers) / len(data)

        # Estatísticas adicionais
        median = data.median()
        skew = data.skew()
        kurt = data.kurtosis()

        results.append({
            "feature": feature,
            "mean": mean,
            "median": median,
            "std": std,
            "skew": skew,
            "kurtosis": kurt,
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outliers": len(outliers),
            "pct_outliers": pct_outliers
        })

        # -------------------------
        # PLOTS
        # -------------------------
        if max_plots and i >= max_plots:
            continue

        fig, axes = plt.subplots(1, 2 if show_boxplot else 1, figsize=figsize)

        if not show_boxplot:
            axes = [axes]

        # HISTOGRAMA
        sns.histplot(data, bins=bins, ax=axes[0])

        axes[0].axvline(mean, linestyle='-', label=f"Mean: {mean:.2f}")
        axes[0].axvline(median, linestyle='-', label=f"Median: {median:.2f}")

        axes[0].axvline(lower, linestyle='--', label=f"-{threshold}σ")
        axes[0].axvline(upper, linestyle='--', label=f"+{threshold}σ")

        axes[0].set_title(f"{feature} - Gaussian Outliers", fontsize=title_fontsize)
        axes[0].tick_params(axis='both', labelsize=tick_fontsize)
        axes[0].legend()

        # -------------------------
        # BLOCO DE ESTATÍSTICAS
        # -------------------------
        stats_text = (
            f"N: {len(data)}\n"
            f"Outliers: {len(outliers)} ({pct_outliers:.2%})\n"
            f"Mean: {mean:.2f}\n"
            f"Median: {median:.2f}\n"
            f"Std: {std:.2f}\n"
            f"Skew: {skew:.2f}\n"
            f"Kurtosis: {kurt:.2f}"
        )

        axes[0].text(
            0.97, 0.95,
            stats_text,
            transform=axes[0].transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        # -------------------------
        # BOXPLOT
        # -------------------------
        if show_boxplot:
            sns.boxplot(x=data, ax=axes[1])

            axes[1].axvline(lower, linestyle='--')
            axes[1].axvline(upper, linestyle='--')

            axes[1].set_title(f"{feature} - Boxplot", fontsize=title_fontsize)
            axes[1].tick_params(axis='both', labelsize=tick_fontsize)

        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results).sort_values(
        by="pct_outliers",
        ascending=False
    )

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca_analysis(
    df: pd.DataFrame,
    target_col: str = None,
    n_components: int = None,
    scale: bool = True,
    drop_first_dummies: bool = True,
    random_state: int = 42
):
    """
    Executa PCA com boas práticas e gera:
    - Scatter PC1 vs PC2 (colorido pelo target)
    - Barplot variância explicada
    - Lineplot variância acumulada
    """

    df = df.copy()

    # Separar target
    y = None
    if target_col:
        y = df[target_col]
        df = df.drop(columns=[target_col])

    # One-hot encoding para categóricas
    df = pd.get_dummies(df, drop_first=drop_first_dummies)

    # Escalonamento (CRÍTICO para PCA)
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
    else:
        X_scaled = df.values

    # PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Criar DataFrame com componentes
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
    )

    if y is not None:
        pca_df[target_col] = y.values

    # ---------------------------
    # 1. Scatter PC1 vs PC2
    # ---------------------------
    plt.figure(figsize=(8, 6))

    if y is not None:
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue=target_col,
            palette="Set2",
            alpha=0.7
        )
    else:
        plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7)

    plt.title("PCA - PC1 vs PC2")
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # 2. Variância explicada
    # ---------------------------
    explained_var = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=[f"PC{i+1}" for i in range(len(explained_var))],
        y=explained_var
    )
    plt.title("Variância Explicada por Componente")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # 3. Variância acumulada
    # ---------------------------
    cumulative_var = np.cumsum(explained_var)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_var)+1), cumulative_var, marker='o')
    plt.axhline(y=0.9, linestyle='--')
    plt.xlabel("Número de Componentes")
    plt.ylabel("Variância Acumulada")
    plt.title("Variância Explicada Acumulada")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return pca, df.columns, pca_df


def get_pca_loadings(pca, feature_names, n_components=5):
    """
    Retorna os loadings das primeiras n componentes
    """

    loadings = pd.DataFrame(
        pca.components_[:n_components].T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=feature_names
    )

    return loadings
    
def plot_loadings_heatmap(
    loadings: pd.DataFrame,
    top_n: int = None,
    sort_by: str = "PC1",
    center: float = 0,
    cmap: str = "coolwarm",
    figsize=(10, 8)
):
    """
    Plota um heatmap dos loadings do PCA.

    Parâmetros:
    - loadings: DataFrame retornado por get_pca_loadings
    - top_n: seleciona as top features mais importantes (opcional)
    - sort_by: componente para ordenar (ex: "PC1")
    - center: centralização da escala (0 é ideal)
    - cmap: colormap
    """

    df = loadings.copy()

    # Selecionar top features com base na magnitude global
    if top_n is not None:
        importance = df.abs().sum(axis=1)
        top_features = importance.sort_values(ascending=False).head(top_n).index
        df = df.loc[top_features]

    # Ordenar
    if sort_by in df.columns:
        df = df.reindex(df[sort_by].abs().sort_values(ascending=False).index)

    plt.figure(figsize=figsize)

    sns.heatmap(
        df,
        cmap=cmap,
        center=center,
        annot=True,       # opcional: valores numéricos
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("PCA Loadings Heatmap")
    plt.xlabel("Componentes Principais")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

def feature_importance_from_pca(pca, feature_names, n_components=5):
    """
    Calcula importância global das features baseada nos loadings ponderados
    pela variância explicada
    """

    components = pca.components_[:n_components]
    explained_var = pca.explained_variance_ratio_[:n_components]

    # Peso por variância explicada
    weighted_loadings = np.abs(components.T) * explained_var

    importance = weighted_loadings.sum(axis=1)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    return importance_df


def plot_feature_importance(importance_df, top_n=15):
    plt.figure(figsize=(8, 6))

    top_features = importance_df.head(top_n)

    sns.barplot(
        x="importance",
        y="feature",
        data=top_features
    )

    plt.title(f"Top {top_n} Features (PCA Importance)")
    plt.tight_layout()
    plt.show()