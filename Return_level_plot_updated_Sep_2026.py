import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme
import warnings

# Suppress warnings for clean execution
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
FILE_PATH = r"C:all_sheet_2.xlsx"
LOCATION_TO_FORECAST = 'Klang'

def plot_return_levels():
    print(f"Loading data from sheet '{LOCATION_TO_FORECAST}' in: {FILE_PATH}...")
    
    # 1. Load the dataset
    try:
        df = pd.read_excel(FILE_PATH, sheet_name=LOCATION_TO_FORECAST, engine='openpyxl')
    except Exception as e:
        print(f"\n[ERROR] Could not load data: {e}")
        return
        
    df.columns = [str(col).strip().lower() for col in df.columns]
    if 'date' not in df.columns: 
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    sea_cols = [c for c in df.columns if 'sea' in c or 'level' in c]
    sea_col = sea_cols[0]
    
    # 2. Filter (1994-2024) and Extract Annual Maxima
    df['extracted_year'] = df['date'].dt.year
    df = df[(df['extracted_year'] >= 1994) & (df['extracted_year'] <= 2024)]
    annual_maxima = df.groupby('extracted_year')[sea_col].max().dropna().values
    
    # 3. Fit the GEV Distribution
    # NOTE on SciPy Convention: SciPy uses 'c' where c = -xi. 
    c, loc, scale = genextreme.fit(annual_maxima)
    xi_standard = -c # Converting to standard convention (Coles, 2001)
    
    # 4. Calculate Empirical Return Periods
    N = len(annual_maxima)
    sorted_maxima = np.sort(annual_maxima)
    ranks = np.arange(1, N + 1)
    empirical_return_periods = (N + 1) / (N + 1 - ranks)
    
    # 5. Generate Continuous Theoretical Curve
    T_theoretical = np.logspace(np.log10(1.05), np.log10(1000), 200)
    prob_non_exceedance = 1 - (1 / T_theoretical)
    theoretical_levels = genextreme.ppf(prob_non_exceedance, c, loc=loc, scale=scale)
    
    # 6. Calculate Confidence Intervals via Parametric Bootstrap
    print("Calculating 95% Confidence Intervals (Bootstrapping)...")
    B = 500 
    bootstrap_levels = np.zeros((B, len(T_theoretical)))
    
    # Arrays to hold bootstrapped parameters for Standard Errors
    boot_c = np.zeros(B)
    boot_loc = np.zeros(B)
    boot_scale = np.zeros(B)
    
    for i in range(B):
        synth_sample = genextreme.rvs(c, loc=loc, scale=scale, size=N)
        try:
            c_b, loc_b, scale_b = genextreme.fit(synth_sample)
            bootstrap_levels[i, :] = genextreme.ppf(prob_non_exceedance, c_b, loc=loc_b, scale=scale_b)
            boot_c[i] = c_b
            boot_loc[i] = loc_b
            boot_scale[i] = scale_b
        except:
            bootstrap_levels[i, :] = np.nan
            boot_c[i], boot_loc[i], boot_scale[i] = np.nan, np.nan, np.nan
            
    # Calculate Standard Errors (Standard deviation of bootstrapped parameters)
    se_c = np.nanstd(boot_c)
    se_loc = np.nanstd(boot_loc)
    se_scale = np.nanstd(boot_scale)
    
    # Print Parameters and Standard Errors for the manuscript
    print("\n--- PARAMETER ESTIMATES & STANDARD ERRORS ---")
    print(f"Location (\u03BC): {loc:.4f} \u00B1 {se_loc:.4f}")
    print(f"Scale (\u03C3):    {scale:.4f} \u00B1 {se_scale:.4f}")
    print(f"Shape (SciPy 'c'): {c:.4f} \u00B1 {se_c:.4f}")
    print(f"Shape (\u03BE standard): {xi_standard:.4f} \u00B1 {se_c:.4f}")
    print("---------------------------------------------\n")
            
    # Extract percentiles for the return level plot CIs
    ci_lower = np.nanpercentile(bootstrap_levels, 2.5, axis=0)
    ci_upper = np.nanpercentile(bootstrap_levels, 97.5, axis=0)
    
    # 7. Plotting the Figure
    plt.figure(figsize=(9, 6))
    plt.plot(T_theoretical, ci_upper, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, label='95% Confidence Interval')
    plt.plot(T_theoretical, ci_lower, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
    plt.plot(T_theoretical, theoretical_levels, color='black', linewidth=1.2, label='GEV Model Fit')
    plt.scatter(empirical_return_periods, sorted_maxima, facecolors='none', edgecolors='black', 
                s=30, label=f'Empirical Data ({LOCATION_TO_FORECAST})')
    
    plt.xscale('log')
    x_ticks = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    plt.xticks(x_ticks, [str(t) for t in x_ticks])
    
    plt.xlabel('Return Period (years)', fontsize=12)
    plt.ylabel('Return Level (m)', fontsize=12)
    plt.title(f'Return Level Plot: {LOCATION_TO_FORECAST} Station (1994-2024)', fontsize=14)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f'Return_Level_Plot_{LOCATION_TO_FORECAST}.png', dpi=300)
    plt.show()
    print("Plot saved successfully!")

if __name__ == "__main__":
    plot_return_levels()
