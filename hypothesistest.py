#!/usr/bin/env python
# coding: utf-8

# ##### ONE SAMPLE Z TEST -POPULATION MEAN & STANDARD DEVIATION KNOWN

# In[1]:


import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def main():
    st.set_page_config(page_title="Mean Test Analyzer", layout="wide")
    st.title("One-Sample Mean Test Analyzer")
    st.write("Perform Z-tests or t-tests based on known/unknown population parameters.")

    # --- 1. SIDEBAR: DATA INPUT ---
    st.sidebar.header("1. Data Source")
    data_source = st.sidebar.radio("Select input method:", 
                                 ["Upload Sample Data", "Enter Statistics Manually"])
    
    sample_mean = sample_std = sample_size = None
    
    if data_source == "Upload Sample Data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                numeric_cols = df_input.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.sidebar.selectbox("Select column", numeric_cols)
                    sample_data = df_input[selected_col].dropna()
                    sample_size = len(sample_data)
                    sample_mean = np.mean(sample_data)
                    sample_std = np.std(sample_data, ddof=1) if sample_size > 1 else 0.0
                    
                    st.subheader("Data Summary")
                    st.write(f"**n:** {sample_size} | **x̄:** {sample_mean:.4f} | **s:** {sample_std:.4f}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        sample_size = st.sidebar.number_input("Sample size (n)", min_value=1, value=30)
        sample_mean = st.sidebar.number_input("Sample mean (x̄)", value=0.0)
        sample_std = st.sidebar.number_input("Sample std dev (s)", min_value=0.0, value=1.0)
    
    # --- 2. SIDEBAR: POPULATION PARAMETERS ---
    st.sidebar.header("2. Population Parameters")
    
    pop_mean_known = st.sidebar.checkbox("Population Mean (μ₀) is known", value=True)
    pop_mean = st.sidebar.number_input("Enter μ₀", value=0.0) if pop_mean_known else 0.0
    
    sigma_known = st.sidebar.checkbox("Population std dev (σ) is known")
    pop_std = st.sidebar.number_input("Enter σ", min_value=0.01, value=1.0) if sigma_known else None

    # --- 3. SIDEBAR: TEST CONFIG ---
    st.sidebar.header("3. Test Configuration")
    alpha = st.sidebar.slider("Significance level (α)", 0.001, 0.20, 0.05, step=0.001)
    test_type = st.sidebar.selectbox("Alternative Hypothesis", 
                                   ["Two-tailed (μ ≠ μ₀)", "Left-tailed (μ < μ₀)", "Right-tailed (μ > μ₀)"])
    
    # --- 4. CALCULATION LOGIC ---
    if sample_mean is not None and sample_size is not None:
        # Case 1: Z-test — σ known and sample size ≥ 30
        if sigma_known and pop_std and sample_size >= 30:
            std_error = pop_std / np.sqrt(sample_size)
            test_stat = (sample_mean - pop_mean) / std_error
            dist_name = "Standard Normal (Z)"
            critical_lower = stats.norm.ppf(alpha/2) if test_type == "Two-tailed (μ ≠ μ₀)" else stats.norm.ppf(alpha)
            critical_upper = stats.norm.ppf(1 - alpha/2) if test_type == "Two-tailed (μ ≠ μ₀)" else stats.norm.ppf(1 - alpha)
            p_value = (2 * (1 - stats.norm.cdf(abs(test_stat)))) if test_type == "Two-tailed (μ ≠ μ₀)" else (
                stats.norm.cdf(test_stat) if test_type == "Left-tailed (μ < μ₀)" else 1 - stats.norm.cdf(test_stat))

        # Case 2: t-test — σ known and sample size < 30
        elif sigma_known and pop_std and sample_size < 30:
            std_error = pop_std / np.sqrt(sample_size)
            test_stat = (sample_mean - pop_mean) / std_error
            df = sample_size - 1
            dist_name = f"t-distribution (df={df})"
            critical_lower = stats.t.ppf(alpha/2, df) if test_type == "Two-tailed (μ ≠ μ₀)" else stats.t.ppf(alpha, df)
            critical_upper = stats.t.ppf(1 - alpha/2, df) if test_type == "Two-tailed (μ ≠ μ₀)" else stats.t.ppf(1 - alpha, df)
            p_value = (2 * (1 - stats.t.cdf(abs(test_stat), df))) if test_type == "Two-tailed (μ ≠ μ₀)" else (
                stats.t.cdf(test_stat, df) if test_type == "Left-tailed (μ < μ₀)" else 1 - stats.t.cdf(test_stat, df))

        # Case 3: t-test — σ unknown and sample size ≥ 2
        elif not sigma_known and sample_std and sample_size >= 2:
            std_error = sample_std / np.sqrt(sample_size)
            test_stat = (sample_mean - pop_mean) / std_error
            df = sample_size - 1
            dist_name = f"t-distribution (df={df})"
            critical_lower = stats.t.ppf(alpha/2, df) if test_type == "Two-tailed (μ ≠ μ₀)" else stats.t.ppf(alpha, df)
            critical_upper = stats.t.ppf(1 - alpha/2, df) if test_type == "Two-tailed (μ ≠ μ₀)" else stats.t.ppf(1 - alpha, df)
            p_value = (2 * (1 - stats.t.cdf(abs(test_stat), df))) if test_type == "Two-tailed (μ ≠ μ₀)" else (
                stats.t.cdf(test_stat, df) if test_type == "Left-tailed (μ < μ₀)" else 1 - stats.t.cdf(test_stat, df))

        else:
            st.error("Insufficient data or unsupported configuration for hypothesis test.")
            return

        # Decision rule
        reject = (test_stat < critical_lower) if test_type == "Left-tailed (μ < μ₀)" else (
                (test_stat > critical_upper) if test_type == "Right-tailed (μ > μ₀)" else
                (test_stat < critical_lower) or (test_stat > critical_upper))
    
        # Display results
        st.header("Test Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test statistic", f"{test_stat:.4f}")
            st.metric("Standard Error", f"{std_error:.4f}")
            test_label = "Z-test" if sigma_known and sample_size >= 30 else f"t-test (df={df})"
            st.metric("Test type", test_label)
        with col2:
            st.metric("p-value", f"{p_value:.4f}")
            st.metric("α level", f"{alpha:.3f}")
            st.metric("Conclusion", "Reject H₀" if reject else "Fail to reject H₀")
        
        # Visualization
        st.header("Distribution Visualization")
        x_min = min(-4, test_stat - 1, critical_lower - 1 if 'critical_lower' in locals() else -4)
        x_max = max(4, test_stat + 1, critical_upper + 1 if 'critical_upper' in locals() else 4)
        x = np.linspace(x_min, x_max, 500)
        y = stats.norm.pdf(x) if sigma_known and sample_size >= 30 else stats.t.pdf(x, df)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, label=dist_name)
        
        # Rejection regions
        if test_type == "Two-tailed (μ ≠ μ₀)":
            ax.fill_between(x, y, where=(x <= critical_lower), color='red', alpha=0.3, label='Rejection region')
            ax.fill_between(x, y, where=(x >= critical_upper), color='red', alpha=0.3)
        elif test_type == "Left-tailed (μ < μ₀)":
            ax.fill_between(x, y, where=(x <= critical_lower), color='red', alpha=0.3, label='Rejection region')
        else:
            ax.fill_between(x, y, where=(x >= critical_upper), color='red', alpha=0.3, label='Rejection region')
        
        ax.axvline(test_stat, color='green', linestyle='--', label=f'Test statistic = {test_stat:.2f}')
        if test_type == "Two-tailed (μ ≠ μ₀)":
            ax.axvline(critical_lower, color='black', linestyle=':', label=f'Critical values = ±{abs(critical_lower):.2f}')
            ax.axvline(critical_upper, color='black', linestyle=':')
        else:
            crit_val = critical_lower if test_type == "Left-tailed (μ < μ₀)" else critical_upper
            ax.axvline(crit_val, color='black', linestyle=':', label=f'Critical value = {crit_val:.2f}')
        
        dist_type = "Z" if sigma_known and sample_size >= 30 else "t"
        ax.set_title(f"{dist_type}-distribution with {test_type.split(' ')[0]} test")

        ax.legend()
        st.pyplot(fig)
        
        # Interpretation
        st.header("Interpretation")
        st.write(f"**Null Hypothesis (H₀):** μ = {pop_mean:.2f}")
        st.write(f"**Alternative (H₁):** μ {'≠' if test_type == 'Two-tailed (μ ≠ μ₀)' else '<' if test_type == 'Left-tailed (μ < μ₀)' else '>'} {pop_mean:.2f}")
        st.write(f"**Test:** {'Z-test (σ known)' if sigma_known else f't-test (df={df}, σ unknown)'}")
        st.write(f"At α = {alpha:.3f}, we {'reject' if reject else 'fail to reject'} H₀ (p = {p_value:.4f}).")
        
        # In the Interpretation section, replace the last if-else block with:

        # Colorful conclusion
        st.header("Conclusion")
        if reject:
            st.success(f"""
            **Statistically Significant Result**  
            🎯 We reject the null hypothesis (p = {p_value:.4f} < α = {alpha:.3f})  
            ✅ There is significant evidence that μ {'≠' if test_type == 'Two-tailed (μ ≠ μ₀)' else '<' if test_type == 'Left-tailed (μ < μ₀)' else '>'} {pop_mean:.2f}
            """)
        else:
            st.info(f"""
            **No Significant Evidence Found**  
            🎯 We fail to reject the null hypothesis (p = {p_value:.4f} ≥ α = {alpha:.3f})  
            🔍 The data does not provide sufficient evidence to conclude μ {'≠' if test_type == 'Two-tailed (μ ≠ μ₀)' else '<' if test_type == 'Left-tailed (μ < μ₀)' else '>'} {pop_mean:.2f}
            """)

if __name__ == "__main__":
    main()

