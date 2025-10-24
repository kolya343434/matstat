from pathlib import Path
import re
import math
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from IPython.display import display, Markdown  # для вывода формул в консоль

# ---------- утилиты вывода ----------
def mprint(*strings):
    """Вывод строк в формате Markdown (в Jupyter красиво, в консоли просто текст)."""
    for s in strings:
        display(Markdown(s))

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Преобразует DataFrame в Markdown-таблицу (устойчиво к типам столбцов)."""
    if df.index.name is None:
        first_cell = "Index"
    elif df.columns.name is None:
        first_cell = df.index.name
    else:
        first_cell = df.index.name + "\\" + df.columns.name

    headers = " | ".join(map(str, df.columns))
    markdown_table = f"| {first_cell} | {headers} |\n"
    markdown_table += "|" + "---|" * (len(df.columns) + 1) + "\n"

    for idx, row in df.iterrows():
        row_values = [
            (f"{val:.2f}" if np.issubdtype(type(val), np.number) else str(val))
            for val in row
        ]
        markdown_table += f"| {idx} | " + " | ".join(row_values) + " |\n"
    return markdown_table

# ---------- параметры ----------
variant = 14
path_to_vars = Path(r"C:\Users\79165\Downloads\gfggf\var_matstat_K5.xls")
path_to_data = Path(r"C:\Users\79165\Downloads\gfggf\data_matstat_K5.xls")
XLS_ENGINE = "xlrd"  

# ---------- чтение таблицы вариантов ----------
datas = pd.read_excel(path_to_vars, index_col=0, engine=XLS_ENGINE).loc[variant]

variant_dict = {}
for tasks in datas.index:  # например: "2.1, 3.1"
    for task in str(tasks).split(","):
        task = task.strip()
        cols_norm = (
            str(datas[tasks])
            .replace("А", "A").replace("В", "B").replace("С", "C")
        )
        variant_dict[task] = cols_norm.split()

# ---------- чтение книги данных ----------
dfs = pd.read_excel(path_to_data, sheet_name=None, engine=XLS_ENGINE)

# A/B/C/D -> имя листа
sheet_dict = {}
for name in dfs.keys():
    low = str(name).lower()
    if "description" in low:
        continue
    match_obj = re.match(r"\s*([ABCD])\b", str(name), flags=re.IGNORECASE)
    if match_obj:
        sheet_dict[match_obj.group(1).upper()] = name

# нормализуем заголовки столбцов
for nm, df in dfs.items():
    if str(nm).lower().startswith("description"):
        continue
    df.columns = [str(c).strip() for c in df.columns]

def _resolve_key(df: pd.DataFrame, letter: str, col: str):
    """Подбирает реальный ключ столбца ('16', 16, 'B16', 'B 16')."""
    candidates = [col]
    if col.isdigit():
        candidates.append(int(col))
    candidates += [f"{letter}{col}", f"{letter} {col}"]

    for key in candidates:
        if key in df.columns:
            return key

    norm_cols = {str(c).strip().lower(): c for c in df.columns}
    for key in candidates:
        k = str(key).strip().lower()
        if k in norm_cols:
            return norm_cols[k]
    raise KeyError(f"Не найден столбец среди {list(df.columns)[:8]} по ключам {candidates}")

def data(task: str):
    """Возвращает список Series по заданию (например, '1.1')."""
    if task not in variant_dict:
        raise KeyError(f"Задание {task} отсутствует в variant_dict")
    X = []
    for token in variant_dict[task]:
        letter, col = token[0].upper(), token[1:]
        sheet_name = sheet_dict[letter]
        df = dfs[sheet_name]
        key = _resolve_key(df, letter, col)
        X.append(df[key])
    return X

def verdict(alpha: float, p_value: float) -> str:
    return "H0 отвергается" if alpha > p_value else "H0 не отвергается"

# ---------- безопасные настройки Matplotlib ----------
# Стиль (с запасным вариантом)
try:
    plt.style.use("seaborn-v0_8")
except Exception:
    try:
        plt.style.use("seaborn-whitegrid")
    except Exception:
        pass

# LaTeX: включаем только если latex установлен, иначе отключаем.
if shutil.which("latex"):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage[english,russian]{babel}"
else:
    plt.rcParams["text.usetex"] = False

# ---------- Задание 1.1: описательная статистика ----------
columns = data("1.1")
for s in columns:
    print(s.head())

for i, Xs in enumerate(columns, start=1):
    Xs = pd.to_numeric(Xs, errors="coerce").dropna()
    mean_ = Xs.mean()
    D = Xs.var(ddof=0)
    d = Xs.std(ddof=0)
    g = stats.skew(Xs, bias=True, nan_policy="omit")
    e = stats.kurtosis(Xs, bias=True, nan_policy="omit")
    output = (
        fr"$\mathbf{{X_{i}}}:$ "
        fr"$\overline x_{{{i}}} = {mean_:.2f};$ "
        fr"$D^*_{{X_{i}}} = {D:.2f};$ "
        fr"$\sigma^*_{{X_{i}}} = {d:.2f};$ "
        fr"$\gamma^*_{{X_{i}}} = {g:.2f};$ "
        fr"$\varepsilon^*_{{X_{i}}} = {e:.2f}$"
    )
    print(output)

# ---------- Задание 1.2: группировка и графики ----------
X_ser = pd.to_numeric(data("1.2")[0], errors="coerce").dropna()
print(X_ser)
n = len(X_ser)
print(n)

X = pd.DataFrame({"value": X_ser.values})

maximum = X["value"].max()
minimum = X["value"].min()
k = round(1 + 1.3 * math.log(n))  # правило Стерджесса с поправкой
print(fr"$\mathbf{{X}}: \max = {maximum}, \min = {minimum}, n = {n}, k = {k}$")

X["group"] = pd.cut(X["value"], bins=k)

delta_min = np.min([iv.right - iv.left for iv in X["group"].cat.categories])
delta_max = np.max([iv.right - iv.left for iv in X["group"].cat.categories])
print(f"Ширина интервалов: от {delta_min:.2f} до {delta_max:.2f}")

groups = pd.DataFrame(index=pd.Index(range(1, k + 1), name="Номер интервала"))
groups["low edge"] = [iv.left for iv in X["group"].cat.categories]
groups["high edge"] = [iv.right for iv in X["group"].cat.categories]
groups["frequency"] = X["group"].value_counts().sort_index().values
groups["relative frequency"] = groups["frequency"] / n
groups["cumulative frequency"] = groups["frequency"].cumsum()
groups["cumulative relative frequency"] = groups["cumulative frequency"] / n

print(dataframe_to_markdown(groups))

bin_centers = (groups["low edge"] + groups["high edge"]) / 2
width = groups["high edge"] - groups["low edge"]

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

axes[0, 0].bar(bin_centers, groups["frequency"], width=width, edgecolor="black")
axes[0, 0].plot(bin_centers, groups["frequency"], marker="o")
axes[0, 0].set_xlabel("Значения"); axes[0, 0].set_ylabel("Частота")
axes[0, 0].set_title("Гистограмма и полигон частот")

axes[0, 1].bar(bin_centers, groups["relative frequency"], width=width, edgecolor="black")
axes[0, 1].plot(bin_centers, groups["relative frequency"], marker="o")
axes[0, 1].set_xlabel("Значения"); axes[0, 1].set_ylabel("Относительная частота")
axes[0, 1].set_title("Гистограмма и полигон относительных частот")

axes[1, 0].bar(bin_centers, groups["cumulative frequency"], width=width, edgecolor="black")
axes[1, 0].plot(bin_centers, groups["cumulative frequency"], marker="o")
axes[1, 0].set_xlabel("Значения"); axes[1, 0].set_ylabel("Накопленная частота")
axes[1, 0].set_title("Гистограмма и полигон накопленных частот")

axes[1, 1].bar(bin_centers, groups["cumulative relative frequency"], width=width, edgecolor="black")
axes[1, 1].plot(bin_centers, groups["cumulative relative frequency"], marker="o")
axes[1, 1].set_xlabel("Значения"); axes[1, 1].set_ylabel("Относительная накопленная частота")
axes[1, 1].set_title("Гистограмма и полигон относительных накопленных частот")

plt.tight_layout()
plt.show()

# Эмпирическая функция распределения
X_sorted = np.sort(X["value"])
X_sorted = np.concatenate(([-1e10], X_sorted, [1e10]))
F = np.arange(1, n + 1) / n
F = np.concatenate(([0], F, [1]))
plt.step(X_sorted, F, where="post", label=r"$F_X^*(x)$", linewidth=2)
plt.xlabel("Значения"); plt.ylabel(r"$F^*_X(x)$")
plt.title("Эмпирическая функция распределения")
plt.legend()
plt.show()
