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
    # --- 2. SIDEBAR: POPULATION PARAMETERS ---
    st.sidebar.header("2. Population Parameters")
    
    # FIX: Remove the "pop_mean_known" checkbox entirely. 
    # A 1-sample mean test mathematically REQUIRES a target baseline to compare against.
    pop_mean = st.sidebar.number_input("Hypothesized Population Mean (μ₀)", value=0.0)
    
    sigma_known = st.sidebar.checkbox("Population std dev (σ) is known")
    pop_std = st.sidebar.number_input("Enter σ", min_value=0.01, value=1.0) if sigma_known else None

    # --- 3. SIDEBAR: TEST CONFIG ---
    st.sidebar.header("3. Test Configuration")
    alpha = st.sidebar.slider("Significance level (α)", 0.001, 0.20, 0.05, step=0.001)
    test_type = st.sidebar.selectbox("Alternative Hypothesis", 
                                   ["Two-tailed (μ ≠ μ₀)", "Left-tailed (μ < μ₀)", "Right-tailed (μ > μ₀)"])

    # --- 4. CALCULATION LOGIC ---
    if sample_size is not None:
        df_val = None # Initialize df
        
        if sigma_known:
            test_name = "Z-test"
            dist = stats.norm
            dist_args = ()
            std_err = pop_std / np.sqrt(sample_size)
            legend_dist = "Standard Normal (Z)"
        else:
            if sample_size < 2:
                st.error("❌ Need at least n=2 to perform a t-test when σ is unknown.")
                return
            test_name = "t-test"
            df_val = sample_size - 1
            dist = stats.t
            dist_args = (df_val,)
            std_err = sample_std / np.sqrt(sample_size)
            legend_dist = f"t-distribution (df={df_val})"

        test_stat = (sample_mean - pop_mean) / std_err

        # P-value and Critical Value calculation
        if test_type == "Two-tailed (μ ≠ μ₀)":
            p_value = 2 * (1 - dist.cdf(abs(test_stat), *dist_args))
            crit_l = dist.ppf(alpha/2, *dist_args)
            crit_u = dist.ppf(1 - alpha/2, *dist_args)
            display_crit = f"±{abs(crit_u):.4f}"
        elif test_type == "Left-tailed (μ < μ₀)":
            p_value = dist.cdf(test_stat, *dist_args)
            crit_l = dist.ppf(alpha, *dist_args)
            crit_u = np.inf
            display_crit = f"{crit_l:.4f}"
        else: # Right-tailed
            p_value = 1 - dist.cdf(test_stat, *dist_args)
            crit_l = -np.inf
            crit_u = dist.ppf(1 - alpha, *dist_args)
            display_crit = f"{crit_u:.4f}"

        reject = p_value < alpha

        # --- 5. RESULTS DISPLAY (KPIs) ---
        # st.header(f"Results: {test_name}")
        # kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        # kpi1.metric("Test Statistic", f"{test_stat:.4f}")
        # kpi2.metric("P-Value", f"{p_value:.4f}")
        # kpi3.metric("Critical Value", display_crit)
        
        # # Display DF only if it's a T-test
        # if df_val is not None:
        #     kpi4.metric("Deg. of Freedom (df)", df_val)
        # else:
        #     kpi4.metric("Distribution", "Z (Normal)")
            
        # kpi5.metric("Decision", "Reject H₀" if reject else "Fail to Reject")

        st.header("Test Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test statistic", f"{test_stat:.4f}")
            st.metric("Standard Error", f"{std_err:.4f}")
            test_label = "Z-test" if sigma_known and sample_size >= 30 else f"t-test (df={df_val})"
            st.metric("Test type", test_label)
        with col2:
            st.metric("p-value", f"{p_value:.4f}")
            st.metric("α level", f"{alpha:.3f}")
            st.metric("Conclusion", "Reject H₀" if reject else "Fail to reject H₀")

        # --- 6. VISUALIZATION ---
        st.header("Visualizing the Rejection Region")
        x_limit = max(4.5, abs(test_stat) + 1)
        x = np.linspace(-x_limit, x_limit, 1000)
        y = dist.pdf(x, *dist_args)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, color='blue', lw=2, label=f"PDF: {legend_dist}")
        
        # Shade rejection regions
        if test_type == "Two-tailed (μ ≠ μ₀)":
            ax.fill_between(x, 0, y, where=(x <= crit_l) | (x >= crit_u), color='red', alpha=0.3, label="Rejection Region")
            ax.axvline(crit_l, color='black', linestyle=':', alpha=0.6)
            ax.axvline(crit_u, color='black', linestyle=':', alpha=0.6)
        elif test_type == "Left-tailed (μ < μ₀)":
            ax.fill_between(x, 0, y, where=(x <= crit_l), color='red', alpha=0.3, label="Rejection Region")
            ax.axvline(crit_l, color='black', linestyle=':', alpha=0.6)
        else:
            ax.fill_between(x, 0, y, where=(x >= crit_u), color='red', alpha=0.3, label="Rejection Region")
            ax.axvline(crit_u, color='black', linestyle=':', alpha=0.6)

        ax.axvline(test_stat, color='green', linestyle='--', linewidth=2, label=f'Test Stat: {test_stat:.2f}')
        ax.set_title(f"{test_name} Analysis (H₁: {test_type})")
        ax.set_xlabel("Standard Deviations (Z/t units)")
        ax.legend()
        st.pyplot(fig)

        # Conclusion Box
        # if reject:
        #     st.success(f"**Significant Evidence:** We reject H₀ at α={alpha}. The sample mean is significantly different from {pop_mean}.")
        # else:
        #     st.info(f"**Inconclusive:** We fail to reject H₀ at α={alpha}. There is not enough evidence to say the mean differs from {pop_mean}.")
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