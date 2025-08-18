import requests
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
import numpy as np
import pandas as pd

class KEGGEnrichmentAnalyzer:
    def __init__(self):
        self.base_url = "https://rest.kegg.jp"

    def _get_kegg_links(self, source_ids, target_db):
        """
        Map a list of KOs to target KEGG database (e.g., pathway/module).
        Returns mapping: {KO: set([targets])}
        """
        mapping = defaultdict(set)
        batch_size = 50
    
        # Break source_ids into batches
        for i in range(0, len(source_ids), batch_size):
            batch_ids = source_ids[i:i+batch_size]
            
            url = f"{self.base_url}/link/{target_db}/{'+'.join(batch_ids)}"
            r = requests.get(url)
            
            if r.status_code == 200:
                for line in r.text.strip().split("\n"):
                    if not line:
                        continue
                    src, tgt = line.split("\t")
                    mapping[src.replace("ko:", "")].add(tgt.replace(f"{target_db}:", ""))
            else:
                print(f"Warning: Request failed for batch starting at {i}, status {r.status_code}")
        
        ungrouped = []
        for ko in source_ids:
            if ko not in mapping.keys():
                ungrouped.append(ko)

        print(f"KEGG mapping: {len(ungrouped)} out of {len(source_ids)} KOs had no {target_db} mapping.")

        return mapping

    def _build_reverse_mapping(self, mapping):
        """
        Invert KO->term mapping to term->set(KOs)
        """
        reverse = {}
        for ko, terms in mapping.items():
            for term in terms:
                reverse.setdefault(term, set()).add(ko)
        return reverse

    def perform_enrichment_analysis(self, ko_set1, ko_set2, level="pathway"):
        """
        Compare enrichment of KEGG pathways/modules between two KO sets.
        level = 'pathway' or 'module'

        Perform chi-squared test to identify strings enriched in ko_set1 compared to ko_set2.

        This function performs chi-squared tests on 2x2 contingency tables for each unique 
        string found in either dataset to identify which strings are significantly enriched 
        in ko_set1 relative to ko_set2. Multiple testing correction is applied using the 
        Benjamini-Hochberg FDR method.

        Contingency Table Structure for each string:
        ┌─────────────┬──────────────┬──────────────┐
        │             │ String       │ String       │
        │             │ Present      │ Absent       │
        ├─────────────┼──────────────┼──────────────┤
        │ Set 1       │ count_in_s1  │ absent_in_s1 │
        │ Set 2       │ count_in_s2  │ absent_in_s2 │
        └─────────────┴──────────────┴──────────────┘

        Parameters:
        -----------
        ko_set1 : list
            List of strings (e.g., KO identifiers) from first dataset
        ko_set2 : list
            List of strings (e.g., KO identifiers) from second dataset  
        alpha : float, default=0.05
            Significance level for multiple testing correction

        Returns:
        --------
        pandas.DataFrame
            Results sorted by corrected p-value, containing:
            - string: The identifier being tested
            - count_set1/count_set2: Number of occurrences in each set
            - total_set1/total_set2: Total size of each set
            - chi2_statistic: Chi-squared test statistic
            - p_value: Raw p-value from chi-squared test
            - enrichment_ratio: Odds ratio (>1 indicates enrichment in set1)
            - p_corrected: FDR-corrected p-value (Benjamini-Hochberg)
            - significant: Boolean indicating statistical significance after correction
            - enriched_in_set1: Boolean indicating enrichment in set1 (significant + ratio > 1)

        Example:
        --------
        >>> ko_set1 = ['K00001', 'K00001', 'K00002', 'K00003']
        >>> ko_set2 = ['K00002', 'K00004', 'K00004', 'K00005']  
        >>> results = chi_squared_enrichment_test(ko_set1, ko_set2)
        >>> enriched = results[results['enriched_in_set1']]
        """
        
        # Map KOs to pathways/modules
        mapping = self._get_kegg_links(ko_set1 + ko_set2, level)
        term_to_kos = self._build_reverse_mapping(mapping)
        
        ko_set1 = [mapping.get(ko, set()) for ko in ko_set1]
        ko_set1 = [item for sublist in ko_set1 for item in sublist]
        ko_set2 = [mapping.get(ko, set()) for ko in ko_set2]
        ko_set2 = [item for sublist in ko_set2 for item in sublist]

        # Get all unique strings across both sets
        all_strings = list(set(ko_set1 + ko_set2))

        # Count occurrences in each set
        ko_set1_counts = {s: ko_set1.count(s) for s in all_strings}
        ko_set2_counts = {s: ko_set2.count(s) for s in all_strings}

        # Calculate totals
        total_set1 = len(ko_set1)
        total_set2 = len(ko_set2)

        results = []

        for string in all_strings:
            # Create 2x2 contingency table for each string
            observed_set1 = ko_set1_counts[string]
            absent_set1 = total_set1 - observed_set1
            observed_set2 = ko_set2_counts[string]
            absent_set2 = total_set2 - observed_set2

            # Contingency table: rows=sets, columns=[present, absent]
            contingency_table = np.array([
                [observed_set1, absent_set1],
                [observed_set2, absent_set2]
            ])

            # Perform chi-squared test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

            # Calculate enrichment ratio (odds ratio)
            # OR = (a/b) / (c/d) = (a*d) / (b*c)
            # where a=obs_set1, b=absent_set1, c=obs_set2, d=absent_set2
            if observed_set2 > 0 and absent_set1 > 0:
                enrichment_ratio = (observed_set1 * absent_set2) / (absent_set1 * observed_set2)
            else:
                # Handle edge cases
                if observed_set1 > 0 and observed_set2 == 0:
                    enrichment_ratio = np.inf  # Infinitely enriched in set1
                elif observed_set1 == 0 and observed_set2 > 0:
                    enrichment_ratio = 0  # Not present in set1
                else:
                    enrichment_ratio = 1  # Equal proportions or both zero

            results.append({
                'string': string,
                'count_set1': observed_set1,
                'count_set2': observed_set2,
                'total_set1': total_set1,
                'total_set2': total_set2,
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'enrichment_ratio': enrichment_ratio
            })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Multiple testing correction using Benjamini-Hochberg (FDR)
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            results_df['p_value'], 
            alpha=0.05, 
            method='fdr_bh'
        )

        # Add corrected p-values and significance indicators
        results_df['p_corrected'] = p_corrected
        results_df['significant'] = rejected
        results_df['enriched_in_set1'] = rejected & (results_df['enrichment_ratio'] > 1)

        # Sort by corrected p-value for easy interpretation
        results_df = results_df.sort_values('p_corrected').reset_index(drop=True)

        return results_df


    def save_results(self, df, filename):
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")