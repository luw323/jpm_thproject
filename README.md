# JPM_THProject

## Observations from the Correlation Heatmap:

### Multicollinearity Detection:
- Some features are highly correlated with each other:
  - **total_bc_limit** and **tot_hi_cred_lim** (correlation ≈ 1.00)
  - **percent_bc_gt_75** and **bc_util** (correlation ≈ 0.83)
  - **loan_amnt** and **internal_score** (correlation ≈ 1.00)
- These strong correlations suggest potential multicollinearity, which could negatively affect model performance.

### Target Correlation:
- The target variable **bad_flag** shows weak correlations with all numerical features.
- Features like **annual_inc**, **dti**, **term**, and **int_rate** show slightly higher correlation but are still low (< 0.2), indicating limited linear relationships.

## Recommendations:

1. **Remove or Combine Highly Correlated Features:**
   - For instance, reduce **total_bc_limit** and **tot_hi_cred_lim** to one feature.

2. **Feature Engineering:**
   - Create derived features that may better capture the relationships with **bad_flag**.

3. **Explore Non-linear Relationships:**
   - Given the weak correlations, consider using tree-based models or transformations to better capture patterns.
