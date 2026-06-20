### The problem
Currently, our model relies on the `GC.DOD.TOTL.GD.ZS` indicator (Central Government Debt, % of GDP) to calculate the Fiscal pillar score. However, this specific indicator suffers from incomplete global coverage, returning missing values for several major countries in our default panel. This forces the script to impute the panel average, which distorts the final fiscal ranking.

### The solution
To make the Fiscal pillar more accurate and reliable, we need to replace the current debt indicator. There are two potential approaches to implement:

1. **Calculate the ratio manually:** 
   Replace the current indicator with `GC.DOD.TOTL.CN` (Central Government Debt in local currency). Since this returns an absolute value, we will also need to fetch the country's GDP in local currency (e.g., using `NY.GDP.MKTP.CN`) and compute the Debt-to-GDP ratio ourselves within the script `(Debt / GDP) * 100`.

2. **Catalog Research:** 
   Investigate the World Bank Data Catalog to find if there is another pre-calculated Debt-to-GDP indicator with a significantly better global coverage rate (potentially relying on cross-referenced IMF data).

### Next steps
- [ ] Test the `GC.DOD.TOTL.CN` + GDP calculation approach in a separate notebook.
- [ ] Search the World Bank catalog for alternative pre-calculated metrics.
- [ ] Update the `WB_INDICATORS` dictionary in the main script.
