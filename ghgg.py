
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, jarque_bera, binomtest, chi2_contingency

ALPHA = 0.10 # уровень значимости 
TARGET_COLS = ("B16", "B17", "B18")     

# ---------- утилиты чтения ----------
def _norm_name(s) -> str:
    return "".join(str(s).split()).upper()   # убираем пробелы, приводим к UPPER

def load_excel_cols(file_mask="data_matstat_k5", targets=TARGET_COLS, sheet=None):
    # 1) находим файл независимо от регистра и расширения
    p = Path.cwd()
    cand = next((f for f in p.iterdir()
                 if f.is_file() and f.name.lower().startswith(file_mask.lower() + ".")
                 and f.suffix.lower() in (".xls", ".xlsx")), None)
    if cand is None:
        raise FileNotFoundError(f"Не найден файл {file_mask}.xls/.xlsx в: {p}")

    # 2) выбираем engine по расширению
    engine = "xlrd" if cand.suffix.lower()==".xls" else "openpyxl"
    xls = pd.ExcelFile(cand, engine=engine)
    sheets = [sheet] if sheet is not None else xls.sheet_names

    # 3) пытаемся определить строку заголовков (0..9) и взять нужные колонки
    for sh in sheets:
        for hdr in list(range(0,10))+[None]:
            df = pd.read_excel(cand, sheet_name=sh, header=hdr, engine=engine)
            colmap = { _norm_name(c): c for c in df.columns }
            if all(_norm_name(t) in colmap for t in targets):
                data = { t: pd.to_numeric(df[colmap[_norm_name(t)]], errors="coerce").to_numpy()
                         for t in targets }
                return data, cand.name, sh, hdr
    

# ---------- критерии ----------


def chi2_normality(X, alpha=ALPHA): # проверяем гипотезу Н0 о нормальности распределеня выборки
    X = np.asarray(X, float); X = X[~np.isnan(X)]
    n, mu, sd = len(X), X.mean(), X.std(ddof=1)
    k = min(12, max(6, n//20))  # число бинов
    edges = norm.ppf(np.linspace(0,1,k+1), loc=mu, scale=sd) # границы столбиков
    edges[0]= -np.inf # всё, что меньше первой границы, попадёт в первый интервал
    edges[-1] =  np.inf # всё, что больше последней границы, попадёт в последний интервал
    obs, _ = np.histogram(X, bins=edges) # реальны е наблюдения которые попали в интервали в гистограмме
    exp = np.full(k, n/k) # кол-во наблюдений в столбце  если H0 верна
    stat = np.sum((obs-exp)**2/exp) # вычисление критерия пирсона
    dfree = k-2-1                            # оценивали μ и σ
    pval = 1 - chi2.cdf(stat, dfree) # критерий проверки , если 
    dec = "Отвергаем H0" if pval < alpha else "Нет оснований отвергать H0"
    concl = "F(B18) ∉ N" if pval < alpha else "Нет оснований отклонять нормальность F(B18)"
    return stat, dfree, pval, dec, concl

def jb_row(X, alpha=ALPHA):# проверка ассиметрии и экцессы
    X = np.asarray(X, float); X = X[~np.isnan(X)] 
    stat, pval = jarque_bera(X) 
    dec = "Отвергаем H0" if pval < alpha else "Нет оснований отвергать H0"
    concl = "F(B18) ∉ N" if pval < alpha else "Нет оснований отклонять нормальность F(B18)"
    return stat, 2, pval, dec, concl

def sign_test(A, B, alpha=ALPHA):
    A = pd.Series(A); B = pd.Series(B)
    P = pd.DataFrame({"A":A,"B":B}).dropna().to_numpy()
    if P.size == 0: 
        return np.nan, "Недостаточно данных", "—", 0, 0
    d = P[:,0]-P[:,1]; d = d[d!=0]
    n = len(d); 
    if n==0:
        return np.nan, "Все разности нули", "—", 0, 0
    s_plus = np.count_nonzero(d>0)
    pval = binomtest(s_plus, n, p=0.5, alternative="two-sided").pvalue
    dec = "Отвергаем H0" if pval < alpha else "Нет оснований отвергать H0"
    concl = "F1(B16) ≠ F2(B18)" if pval < alpha else "F1(B16) = F2(B18)"
    return pval, dec, concl, n, s_plus

def chi2_homog(A, B, alpha=ALPHA):
    A = np.asarray(A, float); B = np.asarray(B, float)
    A = A[~np.isnan(A)]; B = B[~np.isnan(B)]
    Z = np.concatenate([A,B])
    k = min(10, max(6, len(Z)//30))
    edges = np.quantile(Z, np.linspace(0,1,k+1)) 
    edges[0], edges[-1] = -np.inf, np.inf
    o1,_ = np.histogram(A, bins=edges); o2,_ = np.histogram(B, bins=edges)
    stat, pval, dfree, _ = chi2_contingency(np.vstack([o1,o2]), correction=False)
    dec = "Отвергаем H0" if pval < alpha else "Нет оснований отвергать H0"
    concl = "F1(B16) ≠ F2(B18)" if pval < alpha else "F1(B16) = F2(B18)"
    return stat, dfree, pval, dec, concl

# ---------- загрузка и расчёт ----------
data, fname, sheet_used, header_row = load_excel_cols("data_matstat_k5", TARGET_COLS)
x16, x17, x18 = data["B16"], data["B17"], data["B18"]

r41 = chi2_normality(x18)
r42 = jb_row(x18)
p51, d51, c51, n_pairs, s_plus = sign_test(x16, x18)
r52 = chi2_homog(x16, x18)

table = pd.DataFrame([
    ["4.1", "H₀: F(B18) ~ N",      "Хи-квадрат", f"stat={r41[0]:.3f}, df={r41[1]}, p={r41[2]:.4f} → {r41[3]}", r41[4]],
    ["4.2", "H₀: F(x) ~ N",        "Жарке–Бера", f"stat={r42[0]:.3f}, df={r42[1]}, p={r42[2]:.4f} → {r42[3]}", r42[4]],
    ["5.1", "H₀: F₁(B16)=F₂(B18)", "знаков",     f"pairs={n_pairs}, s+={s_plus}, p={p51:.4f} → {d51}",           c51],
    ["5.2", "H₀: F₁(B16)=F₂(B18)", "Хи-квадрат", f"stat={r52[0]:.3f}, df={r52[1]}, p={r52[2]:.4f} → {r52[3]}",   r52[4]],
], columns=["№ задания","Проверяемая гипотеза H₀","Критерий","Статистическое решение (α = 0.1)","Вывод"])

print(f"Файл: {fname} | Лист: {sheet_used} | Строка заголовков: {header_row}")
print(table.to_string(index=False))
