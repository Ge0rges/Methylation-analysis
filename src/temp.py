import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def analyze_differential_expression_patterns(df_diff, treatment_metadata, output_file='feature_analysis_results.xlsx'):
    """
    Comprehensive analysis of differential expression patterns across treatment types

    Parameters:
    -----------
    df_diff : pandas.DataFrame
        DataFrame with columns: contig, strand, position, treatment_1, treatment_2, significant
        - contig, strand, position: together define a unique feature
        - treatment_1, treatment_2: the two treatments being compared
        - significant: boolean indicating if the feature differs between treatments

    treatment_metadata : pandas.DataFrame
        DataFrame with columns: treatment, control, salinity
        - treatment: treatment name (must match treatment_1/treatment_2 in df_diff)
        - control: boolean indicating if this is a control treatment
        - salinity: boolean indicating if this is a high salinity treatment

    output_file : str
        Name of the Excel file to save results to

    Returns:
    --------
    dict: Dictionary containing all analysis results
    """

    # Create feature identifier
    df_diff = df_diff.copy()
    df_diff['feature_id'] = (df_diff['contig'].astype(str) + '<>' + 
                            df_diff['strand'].astype(str) + '<>' + 
                            df_diff['position'].astype(str))

    # Get all unique features
    features = df_diff['feature_id'].unique()
    print(f"Analyzing {len(features)} unique features...")

    # Create treatment mappings
    control_treatments = treatment_metadata[treatment_metadata['control'] == True]['treatment'].tolist()
    experimental_treatments = treatment_metadata[treatment_metadata['control'] == False]['treatment'].tolist()
    high_salinity_treatments = treatment_metadata[treatment_metadata['salinity'] == True]['treatment'].tolist()
    low_salinity_treatments = treatment_metadata[treatment_metadata['salinity'] == False]['treatment'].tolist()

    print(f"Control treatments: {control_treatments}")
    print(f"Experimental treatments: {experimental_treatments}")
    print(f"High salinity treatments: {high_salinity_treatments}")
    print(f"Low salinity treatments: {low_salinity_treatments}")

    def feature_changes_within_group(feature, group_treatments):
        """Check if feature changes within a group of treatments"""
        if len(group_treatments) < 2:
            return False

        feature_data = df_diff[df_diff['feature_id'] == feature]
        for i, t1 in enumerate(group_treatments):
            for j, t2 in enumerate(group_treatments):
                if i < j:
                    comparison = feature_data[
                        ((feature_data['treatment_1'] == t1) & (feature_data['treatment_2'] == t2)) |
                        ((feature_data['treatment_1'] == t2) & (feature_data['treatment_2'] == t1))
                    ]
                    if not comparison.empty and comparison['significant'].any():
                        return True
        return False

    def feature_changes_between_groups(feature, group1_treatments, group2_treatments):
        """Check if feature changes between two groups of treatments"""
        feature_data = df_diff[df_diff['feature_id'] == feature]
        for t1 in group1_treatments:
            for t2 in group2_treatments:
                comparison = feature_data[
                    ((feature_data['treatment_1'] == t1) & (feature_data['treatment_2'] == t2)) |
                    ((feature_data['treatment_1'] == t2) & (feature_data['treatment_2'] == t1))
                ]
                if not comparison.empty and comparison['significant'].any():
                    return True
        return False

    def feature_changes_in_treatment_type(feature, control_val, salinity_val):
        """Check if a feature shows significant changes within treatments of a specific type"""
        matching_treatments = treatment_metadata[
            (treatment_metadata['control'] == control_val) & 
            (treatment_metadata['salinity'] == salinity_val)
        ]['treatment'].tolist()

        return feature_changes_within_group(feature, matching_treatments)

    results = {}

    print("\nAnalyzing patterns...")

    # 1. Features constant throughout experiment
    print("1. Finding features constant throughout experiment...")
    constant_features = []
    for feature in features:
        feature_data = df_diff[df_diff['feature_id'] == feature]
        if not feature_data['significant'].any():
            constant_features.append(feature)

    results['1_constant_throughout'] = pd.DataFrame({
        'feature_id': constant_features,
        'contig': [f.split('<>')[0] for f in constant_features],
        'strand': [f.split('<>')[1] for f in constant_features],
        'position': [int(f.split('<>')[2]) for f in constant_features],
        'description': ['Never significant in any comparison'] * len(constant_features)
    })

    # 2. Features constant in one treatment type but not others
    print("2. Finding features constant in one treatment type but not others...")
    treatment_types = [(True, True), (True, False), (False, True), (False, False)]
    type_specific_constant = []

    for feature in features:
        if feature in constant_features:
            continue

        for control_val, salinity_val in treatment_types:
            if not feature_changes_in_treatment_type(feature, control_val, salinity_val):
                # Check if changes in other types
                changes_elsewhere = False
                for other_control, other_salinity in treatment_types:
                    if (other_control, other_salinity) != (control_val, salinity_val):
                        if feature_changes_in_treatment_type(feature, other_control, other_salinity):
                            changes_elsewhere = True
                            break

                # Also check between-type changes
                if not changes_elsewhere:
                    current_treatments = treatment_metadata[
                        (treatment_metadata['control'] == control_val) & 
                        (treatment_metadata['salinity'] == salinity_val)
                    ]['treatment'].tolist()

                    other_treatments = treatment_metadata[
                        ~((treatment_metadata['control'] == control_val) & 
                          (treatment_metadata['salinity'] == salinity_val))
                    ]['treatment'].tolist()

                    if feature_changes_between_groups(feature, current_treatments, other_treatments):
                        changes_elsewhere = True

                if changes_elsewhere:
                    type_specific_constant.append({
                        'feature_id': feature,
                        'contig': feature.split('<>')[0],
                        'strand': feature.split('<>')[1],
                        'position': int(feature.split('<>')[2]),
                        'treatment_type': f"control_{control_val}_salinity_{salinity_val}",
                        'description': f"Constant in control={control_val}, salinity={salinity_val} but changes elsewhere"
                    })

    results['2_constant_in_one_type'] = pd.DataFrame(type_specific_constant)

    # 3. Features that change in one treatment type but not others
    print("3. Finding features that change in one treatment type but not others...")
    type_specific_changing = []

    for feature in features:
        if feature in constant_features:
            continue

        for control_val, salinity_val in treatment_types:
            if feature_changes_in_treatment_type(feature, control_val, salinity_val):
                # Check if constant in all other types
                constant_elsewhere = True
                for other_control, other_salinity in treatment_types:
                    if (other_control, other_salinity) != (control_val, salinity_val):
                        if feature_changes_in_treatment_type(feature, other_control, other_salinity):
                            constant_elsewhere = False
                            break

                # Also check between-type changes
                if constant_elsewhere:
                    current_treatments = treatment_metadata[
                        (treatment_metadata['control'] == control_val) & 
                        (treatment_metadata['salinity'] == salinity_val)
                    ]['treatment'].tolist()

                    other_treatments = treatment_metadata[
                        ~((treatment_metadata['control'] == control_val) & 
                          (treatment_metadata['salinity'] == salinity_val))
                    ]['treatment'].tolist()

                    if feature_changes_between_groups(feature, current_treatments, other_treatments):
                        constant_elsewhere = False

                if constant_elsewhere:
                    type_specific_changing.append({
                        'feature_id': feature,
                        'contig': feature.split('<>')[0],
                        'strand': feature.split('<>')[1],
                        'position': int(feature.split('<>')[2]),
                        'treatment_type': f"control_{control_val}_salinity_{salinity_val}",
                        'description': f"Changes in control={control_val}, salinity={salinity_val} but constant elsewhere"
                    })

    results['3_changing_in_one_type'] = pd.DataFrame(type_specific_changing)

    # 4. Features constant in controls but change in experimentals
    print("4. Finding features constant in controls but changing in experimentals...")
    control_constant_exp_change = []
    for feature in features:
        control_constant = not feature_changes_within_group(feature, control_treatments)
        exp_changes = feature_changes_within_group(feature, experimental_treatments)

        if control_constant and exp_changes:
            control_constant_exp_change.append(feature)

    results['4_constant_controls_change_experimental'] = pd.DataFrame({
        'feature_id': control_constant_exp_change,
        'contig': [f.split('<>')[0] for f in control_constant_exp_change],
        'strand': [f.split('<>')[1] for f in control_constant_exp_change],
        'position': [int(f.split('<>')[2]) for f in control_constant_exp_change],
        'description': ['Constant in controls, changes in experimental conditions'] * len(control_constant_exp_change)
    })

    # 5. Features change in controls but constant in experimentals
    print("5. Finding features changing in controls but constant in experimentals...")
    control_change_exp_constant = []
    for feature in features:
        control_changes = feature_changes_within_group(feature, control_treatments)
        exp_constant = not feature_changes_within_group(feature, experimental_treatments)

        if control_changes and exp_constant:
            control_change_exp_constant.append(feature)

    results['5_change_controls_constant_experimental'] = pd.DataFrame({
        'feature_id': control_change_exp_constant,
        'contig': [f.split('<>')[0] for f in control_change_exp_constant],
        'strand': [f.split('<>')[1] for f in control_change_exp_constant],
        'position': [int(f.split('<>')[2]) for f in control_change_exp_constant],
        'description': ['Changes in controls, constant in experimental conditions'] * len(control_change_exp_constant)
    })

    # 6. Features constant in high salinity but change in low salinity
    print("6. Finding features constant in high salinity but changing in low salinity...")
    high_sal_constant_low_sal_change = []
    for feature in features:
        high_sal_constant = not feature_changes_within_group(feature, high_salinity_treatments)
        low_sal_changes = feature_changes_within_group(feature, low_salinity_treatments)

        if high_sal_constant and low_sal_changes:
            high_sal_constant_low_sal_change.append(feature)

    results['6_constant_high_salinity_change_low_salinity'] = pd.DataFrame({
        'feature_id': high_sal_constant_low_sal_change,
        'contig': [f.split('<>')[0] for f in high_sal_constant_low_sal_change],
        'strand': [f.split('<>')[1] for f in high_sal_constant_low_sal_change],
        'position': [int(f.split('<>')[2]) for f in high_sal_constant_low_sal_change],
        'description': ['Constant in high salinity, changes in low salinity'] * len(high_sal_constant_low_sal_change)
    })

    # 7. Features change in high salinity but constant in low salinity  
    print("7. Finding features changing in high salinity but constant in low salinity...")
    high_sal_change_low_sal_constant = []
    for feature in features:
        high_sal_changes = feature_changes_within_group(feature, high_salinity_treatments)
        low_sal_constant = not feature_changes_within_group(feature, low_salinity_treatments)

        if high_sal_changes and low_sal_constant:
            high_sal_change_low_sal_constant.append(feature)

    results['7_change_high_salinity_constant_low_salinity'] = pd.DataFrame({
        'feature_id': high_sal_change_low_sal_constant,
        'contig': [f.split('<>')[0] for f in high_sal_change_low_sal_constant],
        'strand': [f.split('<>')[1] for f in high_sal_change_low_sal_constant],
        'position': [int(f.split('<>')[2]) for f in high_sal_change_low_sal_constant],
        'description': ['Changes in high salinity, constant in low salinity'] * len(high_sal_change_low_sal_constant)
    })
    
    # 8. Features different in control, salinity=False, step=1 vs other control steps
    print("Finding features different in control, salinity=False, step=1 vs other control steps...")

    control_step1_treatments = treatment_metadata[
        (treatment_metadata['control'] == True) & 
        (treatment_metadata['salinity'] == False) & 
        (treatment_metadata['step'] == 1)
    ]['treatment'].tolist()

    other_control_steps = treatment_metadata[
        (treatment_metadata['control'] == True) & 
        (treatment_metadata['salinity'] == False) & 
        (treatment_metadata['step'] != 1)
    ]['treatment'].tolist()

    control_step1_diffs = []
    for feature in features:
        if feature_changes_between_groups(feature, control_step1_treatments, other_control_steps):
            control_step1_diffs.append(feature)

    results['8_control_step1_vs_other_control'] = pd.DataFrame({
        'feature_id': control_step1_diffs,
        'contig': [f.split('<>')[0] for f in control_step1_diffs],
        'strand': [f.split('<>')[1] for f in control_step1_diffs],
        'position': [int(f.split('<>')[2]) for f in control_step1_diffs],
        'description': ['Different in control salinity=False step=1 vs other control steps'] * len(control_step1_diffs)
    })
    
    
    # 9. Features constant within salinity=False step=1, but different vs salinity=True step=1
    print("Finding features constant in salinity=False, step=1 but different vs salinity=True, step=1...")

    low_sal_step1 = treatment_metadata[
        (treatment_metadata['salinity'] == False) & 
        (treatment_metadata['step'] == 1)
    ]['treatment'].tolist()

    high_sal_step1 = treatment_metadata[
        (treatment_metadata['salinity'] == True) & 
        (treatment_metadata['step'] == 1)
    ]['treatment'].tolist()

    low_step1_constant_high_diff = []
    for feature in features:
        # Must be constant inside low salinity step 1 group
        constant_low = not feature_changes_within_group(feature, low_sal_step1)
        # Must differ between low- vs high-salinity step 1
        diff_high = feature_changes_between_groups(feature, low_sal_step1, high_sal_step1)

        if constant_low and diff_high:
            low_step1_constant_high_diff.append(feature)

    results['9_lowSal_step1_constant_diff_highSal_step1'] = pd.DataFrame({
        'feature_id': low_step1_constant_high_diff,
        'contig': [f.split('<>')[0] for f in low_step1_constant_high_diff],
        'strand': [f.split('<>')[1] for f in low_step1_constant_high_diff],
        'position': [int(f.split('<>')[2]) for f in low_step1_constant_high_diff],
        'description': ['Constant in salinity=False step=1, but different vs salinity=True step=1'] * len(low_step1_constant_high_diff)
    })

    
     # 10. Features constant in salinity=high, step=1 and salinity=high, control=True
    print("Finding features constant in salinity=high, step=1 and salinity=high, control=True...")

    high_sal_step1 = treatment_metadata[
        (treatment_metadata['salinity'] == True) & 
        (treatment_metadata['step'] == 1)
    ]['treatment'].tolist()

    high_sal_control = treatment_metadata[
        (treatment_metadata['salinity'] == True) & 
        (treatment_metadata['control'] == True)
    ]['treatment'].tolist()

    both_high_sal_sets = set(high_sal_step1) & set(high_sal_control)
    # If both sets overlap (treatments that are both), OR, check features constant in BOTH groups?
    # To consider features constant in both groups (not just intersection):
    both_constant_features = []

    for feature in features:
        constant_highstep1_and_highcontrols = not feature_changes_within_group(feature, both_high_sal_sets)
        if constant_highstep1_and_highcontrols:
            both_constant_features.append(feature)

    results['10_constant_high_sal_step1_and_high_sal_control'] = pd.DataFrame({
        'feature_id': both_constant_features,
        'contig': [f.split('<>')[0] for f in both_constant_features],
        'strand': [f.split('<>')[1] for f in both_constant_features],
        'position': [int(f.split('<>')[2]) for f in both_constant_features],
        'description': ['Constant in both salinity=high, step=1 and salinity=high, control=True'] * len(both_constant_features)
    })

    
    # 11. Features constant in control=False, step=[1,2] vs constant in control=False, step=[14,15], and different with any control
    print("Finding features constant in control=False step=[1,2] vs constant in control=False step=[14,15], then different with any control...")

    exp_step_early = treatment_metadata[
        (treatment_metadata['control'] == False) & 
        (treatment_metadata['step'].isin([1, 2]))
    ]['treatment'].tolist()

    exp_step_late = treatment_metadata[
        (treatment_metadata['control'] == False) & 
        (treatment_metadata['step'].isin([14, 15]))
    ]['treatment'].tolist()

    control_all = treatment_metadata[
        (treatment_metadata['control'] == True)
    ]['treatment'].tolist()

    # Find features constant in both early and late experimental step groups:
    both_exp_constant = []
    for feature in features:
        constant_early = not feature_changes_within_group(feature, exp_step_early)
        constant_late = not feature_changes_within_group(feature, exp_step_late)
        # Only consider if constant in BOTH early and late
        if constant_early and constant_late:
            # Now check if this feature is different in any control
            if feature_changes_between_groups(feature, exp_step_early + exp_step_late, control_all):
                both_exp_constant.append(feature)

    results['11_constant_exp_early_late_diff_control'] = pd.DataFrame({
        'feature_id': both_exp_constant,
        'contig': [f.split('<>')[0] for f in both_exp_constant],
        'strand': [f.split('<>')[1] for f in both_exp_constant],
        'position': [int(f.split('<>')[2]) for f in both_exp_constant],
        'description': ['Constant in control=False, steps [1,2,14,15]; different vs any control'] * len(both_exp_constant)
    })


    # Write results to Excel
    print(f"\nWriting results to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in results.items():
            # Join df with df_diff on contig,strand,position, keep only columns "gene_caller_id_start", "distance_to_start", "function", "source", "gene_caller_id_end", "distance_to_end", "function_end", "source_end"
            df = df.merge(df_diff[['contig', 'strand', 'position', 'gene_callers_id_start', 'distance_to_start', 'function', 'source', 'gene_callers_id_end', 'distance_to_end', 'function_end', 'source_end']],
                          on=['contig', 'strand', 'position'], how='left')
            
            if df.empty:
                df = pd.DataFrame(columns=["no data"])
            
            clean_sheet_name = sheet_name.replace('_', ' ').title()[:31]
            df.to_excel(writer, sheet_name=clean_sheet_name, index=False)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF FEATURE ANALYSIS")
    print("="*60)
    for key, df in results.items():
        print(f"{key.replace('_', ' ').title()}: {len(df)} features")
    print(f"\nTotal unique features analyzed: {len(features)}")

    return results


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


def analyze_temporal_patterns(df_diff, treatment_metadata, output_file='temporal_analysis_results.xlsx'):
    """
    Analyze temporal patterns in differential expression data
    
    Parameters:
    -----------
    df_diff : pd.DataFrame with beta_A, beta_B columns and treatment comparisons
    treatment_metadata : pd.DataFrame with step column for temporal information
    """
    
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*50)
    
    # Create feature identifier
    df_diff = df_diff.copy()
    if 'feature_id' not in df_diff.columns:
        df_diff['feature_id'] = (df_diff['contig'] + '<>' + 
                                str(df_diff['strand']) + '<>' + 
                                df_diff['position'].astype(str))
    
    features = df_diff['feature_id'].unique()
    treatment_map = treatment_metadata.set_index('treatment').to_dict('index')
    
    # Calculate effect size for each comparison
    df_diff['effect_size'] = df_diff['beta_A'] - df_diff['beta_B']
    df_diff['magnitude'] = abs(df_diff['effect_size'])
    
    temporal_results = {}
    
    # 1. Early vs Late Responders
    print("1. Identifying early vs late responders...")
    
    early_responders = []
    late_responders = []
    sustained_responders = []
    transient_responders = []
    
    for feature in features:
        feature_data = df_diff[df_diff['feature_id'] == feature]
        
        # Get temporal profile for each treatment type
        temporal_profiles = {}
        
        for treatment_type in ['control_low_salt', 'control_high_salt', 'exp_low_salt', 'exp_high_salt']:
            control_val = 'control' in treatment_type
            salinity_val = 'high_salt' in treatment_type
            
            # Get treatments of this type
            type_treatments = treatment_metadata[
                (treatment_metadata['control'] == control_val) & 
                (treatment_metadata['salinity'] == salinity_val)
            ]['treatment'].tolist()
            
            if len(type_treatments) >= 2:
                # Calculate average magnitude at each timepoint for this treatment type
                timepoint_magnitudes = {}
                
                for step in [1, 2, 3]:
                    step_treatments = [t for t in type_treatments if treatment_map[t]['step'] == step]
                    
                    if step_treatments:
                        # Get comparisons involving this step and treatment type
                        step_comparisons = feature_data[
                            (feature_data['treatment_1'].isin(step_treatments)) |
                            (feature_data['treatment_2'].isin(step_treatments))
                        ]
                        
                        if not step_comparisons.empty:
                            timepoint_magnitudes[step] = step_comparisons['magnitude'].mean()
                
                temporal_profiles[treatment_type] = timepoint_magnitudes
        
        # Analyze temporal patterns
        feature_patterns = []
        for treatment_type, profile in temporal_profiles.items():
            if len(profile) >= 3:  # Need at least 3 timepoints
                steps = sorted(profile.keys())
                magnitudes = [profile[s] for s in steps]
                
                # Check for early response (peak at step 1)
                if magnitudes[0] > max(magnitudes[1:]) * 1.2:
                    feature_patterns.append('early')
                
                # Check for late response (peak at step 3)
                elif magnitudes[-1] > max(magnitudes[:-1]) * 1.2:
                    feature_patterns.append('late')
                
                # Check for sustained response (consistent across time)
                elif max(magnitudes) / min(magnitudes) < 1.5:
                    feature_patterns.append('sustained')
                
                # Check for transient response (peak at step 2)
                elif len(magnitudes) >= 3 and magnitudes[1] > max(magnitudes, magnitudes[asset:1]) * 1.2:
                    feature_patterns.append('transient')
        
        # Classify feature based on dominant pattern
        if 'early' in feature_patterns:
            early_responders.append(feature)
        elif 'late' in feature_patterns:
            late_responders.append(feature)
        elif 'sustained' in feature_patterns:
            sustained_responders.append(feature)
        elif 'transient' in feature_patterns:
            transient_responders.append(feature)
    
    # Store temporal classification results
    temporal_results['early_responders'] = pd.DataFrame({
        'feature_id': early_responders,
        'pattern': ['Early response - peak effect at initial timepoint'] * len(early_responders)
    })
    
    temporal_results['late_responders'] = pd.DataFrame({
        'feature_id': late_responders,
        'pattern': ['Late response - peak effect at final timepoint'] * len(late_responders)
    })
    
    temporal_results['sustained_responders'] = pd.DataFrame({
        'feature_id': sustained_responders,
        'pattern': ['Sustained response - consistent effect across timepoints'] * len(sustained_responders)
    })
    
    temporal_results['transient_responders'] = pd.DataFrame({
        'feature_id': transient_responders,
        'pattern': ['Transient response - peak effect at middle timepoint'] * len(transient_responders)
    })
    
    # 2. Treatment-specific temporal dynamics
    print("2. Analyzing treatment-specific temporal dynamics...")
    
    treatment_specific_temporal = []
    
    for feature in features:
        feature_data = df_diff[df_diff['feature_id'] == feature]
        
        # Compare temporal patterns between control and experimental
        control_temporal = {}
        exp_temporal = {}
        
        # Get temporal profiles for control vs experimental
        for step in [1, 2, 3]:
            control_step_treatments = treatment_metadata[
                (treatment_metadata['control'] == True) & 
                (treatment_metadata['step'] == step)
            ]['treatment'].tolist()
            
            exp_step_treatments = treatment_metadata[
                (treatment_metadata['control'] == False) & 
                (treatment_metadata['step'] == step)
            ]['treatment'].tolist()
            
            # Get average magnitude for control at this step
            control_comparisons = feature_data[
                feature_data['treatment_1'].isin(control_step_treatments) |
                feature_data['treatment_2'].isin(control_step_treatments)
            ]
            if not control_comparisons.empty:
                control_temporal[step] = control_comparisons['magnitude'].mean()
            
            # Get average magnitude for experimental at this step
            exp_comparisons = feature_data[
                feature_data['treatment_1'].isin(exp_step_treatments) |
                feature_data['treatment_2'].isin(exp_step_treatments)
            ]
            if not exp_comparisons.empty:
                exp_temporal[step] = exp_comparisons['magnitude'].mean()
        
        # Compare temporal patterns
        if len(control_temporal) >= 2 and len(exp_temporal) >= 2:
            # Calculate temporal correlation
            common_steps = set(control_temporal.keys()) & set(exp_temporal.keys())
            if len(common_steps) >= 2:
                control_values = [control_temporal[s] for s in sorted(common_steps)]
                exp_values = [exp_temporal[s] for s in sorted(common_steps)]
                
                if len(control_values) >= 2:
                    correlation, p_value = pearsonr(control_values, exp_values)
                    
                    # If correlation is low, this indicates different temporal dynamics
                    if abs(correlation) < 0.5 or p_value > 0.05:
                        treatment_specific_temporal.append({
                            'feature_id': feature,
                            'control_pattern': str(control_temporal),
                            'experimental_pattern': str(exp_temporal),
                            'correlation': correlation,
                            'p_value': p_value,
                            'description': 'Different temporal dynamics between control and experimental'
                        })
    
    temporal_results['treatment_specific_temporal'] = pd.DataFrame(treatment_specific_temporal)
    
    # Print summary
    print(f"Early responders: {len(early_responders)}")
    print(f"Late responders: {len(late_responders)}")
    print(f"Sustained responders: {len(sustained_responders)}")
    print(f"Transient responders: {len(transient_responders)}")
    print(f"Treatment-specific temporal patterns: {len(treatment_specific_temporal)}")
    
    return temporal_results


def analyze_magnitude_patterns(df_diff, treatment_metadata, output_file='magnitude_analysis_results.xlsx'):
    """
    Analyze magnitude-based patterns in differential expression data
    
    Parameters:
    -----------
    df_diff : pd.DataFrame with beta_A, beta_B columns
    treatment_metadata : pd.DataFrame with treatment metadata
    """
    
    print("\nMAGNITUDE-BASED ANALYSIS")
    print("="*50)
    
    # Ensure we have the necessary columns
    df_diff = df_diff.copy()
    if 'effect_size' not in df_diff.columns:
        df_diff['effect_size'] = df_diff['beta_A'] - df_diff['beta_B']
    if 'magnitude' not in df_diff.columns:
        df_diff['magnitude'] = abs(df_diff['effect_size'])
    
    if 'feature_id' not in df_diff.columns:
        df_diff['feature_id'] = (df_diff['contig'] + '<>' + 
                                str(df_diff['strand']) + '<>' + 
                                df_diff['position'].astype(str))
    
    features = df_diff['feature_id'].unique()
    treatment_map = treatment_metadata.set_index('treatment').to_dict('index')
    magnitude_results = {}
    
    # 1. Features with large changes in one condition but small in others
    print("1. Finding features with condition-specific large effects...")
    
    condition_specific_large = []
    
    # Define treatment groups
    treatment_groups = {
        'control_low_salt': treatment_metadata[
            (treatment_metadata['control'] == True) & 
            (treatment_metadata['salinity'] == False)
        ]['treatment'].tolist(),
        'control_high_salt': treatment_metadata[
            (treatment_metadata['control'] == True) & 
            (treatment_metadata['salinity'] == True)
        ]['treatment'].tolist(),
        'exp_low_salt': treatment_metadata[
            (treatment_metadata['control'] == False) & 
            (treatment_metadata['salinity'] == False)
        ]['treatment'].tolist(),
        'exp_high_salt': treatment_metadata[
            (treatment_metadata['control'] == False) & 
            (treatment_metadata['salinity'] == True)
        ]['treatment'].tolist()
    }
    
    for feature in features:
        feature_data = df_diff[df_diff['feature_id'] == feature]
        
        # Calculate average magnitude for each treatment group
        group_magnitudes = {}
        for group_name, treatments in treatment_groups.items():
            # Get comparisons within this group
            within_group = feature_data[
                feature_data['treatment_1'].isin(treatments) & 
                feature_data['treatment_2'].isin(treatments)
            ]
            
            # Get comparisons involving this group
            involving_group = feature_data[
                (feature_data['treatment_1'].isin(treatments)) |
                (feature_data['treatment_2'].isin(treatments))
            ]
            
            if not involving_group.empty:
                group_magnitudes[group_name] = involving_group['magnitude'].mean()
        
        # Find groups with large effects vs small effects
        if len(group_magnitudes) >= 2:
            max_group = max(group_magnitudes, key=group_magnitudes.get)
            min_group = min(group_magnitudes, key=group_magnitudes.get)
            max_magnitude = group_magnitudes[max_group]
            min_magnitude = group_magnitudes[min_group]
            
            # If one group has much larger effects than others
            if max_magnitude > min_magnitude * 2 and max_magnitude > 1.0:
                condition_specific_large.append({
                    'feature_id': feature,
                    'high_effect_condition': max_group,
                    'low_effect_condition': min_group,
                    'high_magnitude': max_magnitude,
                    'low_magnitude': min_magnitude,
                    'fold_difference': max_magnitude / (min_magnitude + 0.001),  # Add small value to avoid division by zero
                    'description': f'Large effect in {max_group} (mag={max_magnitude:.3f}), small in {min_group} (mag={min_magnitude:.3f})'
                })
    
    magnitude_results['condition_specific_large_effects'] = pd.DataFrame(condition_specific_large)
    
    # 2. Features showing opposite directions of change
    print("2. Finding features with opposite directions of change...")
    
    opposite_direction = []
    
    for feature in features:
        feature_data = df_diff[df_diff['feature_id'] == feature]
        
        # Get effect sizes for different treatment comparisons
        effect_sizes = feature_data['effect_size'].tolist()
        
        # Check if we have both positive and negative effect sizes with substantial magnitude
        positive_effects = [e for e in effect_sizes if e > 0.5]
        negative_effects = [e for e in effect_sizes if e < -0.5]
        
        if len(positive_effects) > 0 and len(negative_effects) > 0:
            max_positive = max(positive_effects)
            min_negative = min(negative_effects)
            
            # Find the comparisons responsible for opposite effects
            pos_comparison = feature_data[feature_data['effect_size'] == max_positive].iloc[0]
            neg_comparison = feature_data[feature_data['effect_size'] == min_negative].iloc
            
            opposite_direction.append({
                'feature_id': feature,
                'positive_effect': max_positive,
                'negative_effect': min_negative,
                'positive_comparison': f"{pos_comparison['treatment_1']}_vs_{pos_comparison['treatment_2']}",
                'negative_comparison': f"{neg_comparison['treatment_1']}_vs_{neg_comparison['treatment_2']}",
                'effect_range': max_positive - min_negative,
                'description': f'Opposite effects: +{max_positive:.3f} in one comparison, {min_negative:.3f} in another'
            })
    
    magnitude_results['opposite_direction_effects'] = pd.DataFrame(opposite_direction)
    
    # 3. Treatment-specific magnitude thresholds
    print("3. Analyzing treatment-specific magnitude thresholds...")
    
    # Calculate magnitude distributions for different treatment types
    magnitude_distributions = {}
    
    for group_name, treatments in treatment_groups.items():
        group_comparisons = df_diff[
            df_diff['treatment_1'].isin(treatments) | 
            df_diff['treatment_2'].isin(treatments)
        ]
        
        if not group_comparisons.empty:
            magnitude_distributions[group_name] = {
                'mean': group_comparisons['magnitude'].mean(),
                'median': group_comparisons['magnitude'].median(),
                'std': group_comparisons['magnitude'].std(),
                'q75': group_comparisons['magnitude'].quantile(0.75),
                'q90': group_comparisons['magnitude'].quantile(0.90),
                'max': group_comparisons['magnitude'].max(),
                'count': len(group_comparisons)
            }
    
    magnitude_distributions_df = pd.DataFrame(magnitude_distributions).T
    magnitude_results['magnitude_distributions'] = magnitude_distributions_df
    
    # 4. High-magnitude features per treatment type
    print("4. Identifying high-magnitude features per treatment type...")
    
    high_magnitude_features = []
    
    for group_name, treatments in treatment_groups.items():
        group_comparisons = df_diff[
            df_diff['treatment_1'].isin(treatments) | 
            df_diff['treatment_2'].isin(treatments)
        ]
        
        if not group_comparisons.empty:
            # Use 90th percentile as threshold for high magnitude
            threshold = group_comparisons['magnitude'].quantile(0.90)
            
            high_mag_in_group = group_comparisons[
                group_comparisons['magnitude'] >= threshold
            ]['feature_id'].unique()
            
            for feature in high_mag_in_group:
                feature_max_mag = group_comparisons[
                    group_comparisons['feature_id'] == feature
                ]['magnitude'].max()
                
                high_magnitude_features.append({
                    'feature_id': feature,
                    'treatment_group': group_name,
                    'max_magnitude': feature_max_mag,
                    'threshold': threshold,
                    'description': f'High magnitude effect (>{threshold:.3f}) in {group_name}'
                })
    
    magnitude_results['high_magnitude_features'] = pd.DataFrame(high_magnitude_features)
    
    # Print summary
    print(f"Features with condition-specific large effects: {len(condition_specific_large)}")
    print(f"Features with opposite direction effects: {len(opposite_direction)}")
    print(f"Total high-magnitude feature instances: {len(high_magnitude_features)}")
    
    return magnitude_results


def analyze_clustering_patterns(df_diff, treatment_metadata, n_clusters=5, output_file='clustering_analysis_results.xlsx'):
    """
    Perform clustering-based pattern discovery on differential expression data
    
    Parameters:
    -----------
    df_diff : pd.DataFrame with beta_A, beta_B columns
    treatment_metadata : pd.DataFrame with treatment metadata
    n_clusters : int, number of clusters to create
    """
    
    print("\nCLUSTERING-BASED PATTERN DISCOVERY")
    print("="*50)
    
    # Prepare data
    df_diff = df_diff.copy()
    if 'feature_id' not in df_diff.columns:
        df_diff['feature_id'] = (df_diff['contig'] + '<>' + 
                                str(df_diff['strand']) + '<>' + 
                                df_diff['position'].astype(str))
    
    if 'effect_size' not in df_diff.columns:
        df_diff['effect_size'] = df_diff['beta_A'] - df_diff['beta_B']
    
    features = df_diff['feature_id'].unique()
    print(f"Clustering {len(features)} features...")
    
    # 1. Create feature x comparison matrix
    print("1. Creating feature-comparison matrix...")
    
    # Get all unique treatment comparisons
    comparisons = df_diff[['treatment_1', 'treatment_2']].drop_duplicates()
    comparisons['comparison_id'] = comparisons['treatment_1'] + '_vs_' + comparisons['treatment_2']
    
    # Create matrix: features x comparisons with effect sizes as values
    feature_comparison_matrix = []
    feature_list = []
    
    for feature in features:
        feature_data = df_diff[df_diff['feature_id'] == feature]
        feature_row = []
        
        for _, comp in comparisons.iterrows():
            comp_data = feature_data[
                ((feature_data['treatment_1'] == comp['treatment_1']) & 
                 (feature_data['treatment_2'] == comp['treatment_2'])) |
                ((feature_data['treatment_1'] == comp['treatment_2']) & 
                 (feature_data['treatment_2'] == comp['treatment_1']))
            ]
            
            if not comp_data.empty:
                effect_size = comp_data['effect_size'].iloc[0]
                # If comparison is reversed, flip the sign
                if (comp_data['treatment_1'].iloc == comp['treatment_2'] and 
                    comp_data['treatment_2'].iloc == comp['treatment_1']):
                    effect_size = -effect_size
            else:
                effect_size = 0  # No data for this comparison
            
            feature_row.append(effect_size)
        
        feature_comparison_matrix.append(feature_row)
        feature_list.append(feature)
    
    # Convert to DataFrame
    comparison_ids = comparisons['comparison_id'].tolist()
    feature_matrix = pd.DataFrame(feature_comparison_matrix, 
                                 index=feature_list, 
                                 columns=comparison_ids)
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    # 2. Standardize data for clustering
    print("2. Standardizing data...")
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # 3. Determine optimal number of clusters using silhouette score
    print("3. Determining optimal number of clusters...")
    silhouette_scores = []
    K_range = range(2, min(10, len(features)//3))  # Don't go too high with clusters
    
    for k in K_range:
        if k < len(features):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
            silhouette_avg = silhouette_score(feature_matrix_scaled, cluster_labels)
            silhouette_scores.append((k, silhouette_avg))
    
    # Find optimal k
    if silhouette_scores:
        optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
    else:
        optimal_k = n_clusters
        print(f"Using default number of clusters: {optimal_k}")
    
    # 4. Perform K-means clustering
    print("4. Performing K-means clustering...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
    
    # 5. Perform hierarchical clustering for comparison
    print("5. Performing hierarchical clustering...")
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical.fit_predict(feature_matrix_scaled)
    
    # 6. Analyze cluster characteristics
    print("6. Analyzing cluster characteristics...")
    
    clustering_results = {}
    
    # K-means results
    kmeans_clusters = []
    for cluster_id in range(optimal_k):
        cluster_features = [feature_list[i] for i in range(len(feature_list)) 
                           if cluster_labels[i] == cluster_id]
        
        if cluster_features:
            # Get cluster centroid (average response pattern)
            cluster_data = feature_matrix.loc[cluster_features]
            centroid = cluster_data.mean()
            
            # Find the most characteristic comparisons for this cluster
            top_positive_comps = centroid.nlargest(3).index.tolist()
            top_negative_comps = centroid.nsmallest(3).index.tolist()
            
            # Analyze treatment types involved in characteristic comparisons
            cluster_treatment_pattern = analyze_cluster_treatment_pattern(
                top_positive_comps + top_negative_comps, treatment_metadata
            )
            
            for feature in cluster_features:
                kmeans_clusters.append({
                    'feature_id': feature,
                    'cluster_id': cluster_id,
                    'cluster_size': len(cluster_features),
                    'top_positive_comparisons': ', '.join(top_positive_comps[:2]),
                    'top_negative_comparisons': ', '.join(top_negative_comps[:2]),
                    'treatment_pattern': cluster_treatment_pattern,
                    'description': f'Cluster {cluster_id}: {cluster_treatment_pattern}'
                })
    
    clustering_results['kmeans_clusters'] = pd.DataFrame(kmeans_clusters)
    
    # Hierarchical clustering results
    hierarchical_clusters = []
    for cluster_id in range(optimal_k):
        cluster_features = [feature_list[i] for i in range(len(feature_list)) 
                           if hierarchical_labels[i] == cluster_id]
        
        if cluster_features:
            cluster_data = feature_matrix.loc[cluster_features]
            centroid = cluster_data.mean()
            
            top_positive_comps = centroid.nlargest(3).index.tolist()
            top_negative_comps = centroid.nsmallest(3).index.tolist()
            
            cluster_treatment_pattern = analyze_cluster_treatment_pattern(
                top_positive_comps + top_negative_comps, treatment_metadata
            )
            
            for feature in cluster_features:
                hierarchical_clusters.append({
                    'feature_id': feature,
                    'cluster_id': cluster_id,
                    'cluster_size': len(cluster_features),
                    'top_positive_comparisons': ', '.join(top_positive_comps[:2]),
                    'top_negative_comparisons': ', '.join(top_negative_comps[:2]),
                    'treatment_pattern': cluster_treatment_pattern,
                    'description': f'Hierarchical Cluster {cluster_id}: {cluster_treatment_pattern}'
                })
    
    clustering_results['hierarchical_clusters'] = pd.DataFrame(hierarchical_clusters)
    
    # 7. Cluster summary statistics
    cluster_summary = []
    for cluster_id in range(optimal_k):
        kmeans_size = sum(1 for label in cluster_labels if label == cluster_id)
        hierarchical_size = sum(1 for label in hierarchical_labels if label == cluster_id)
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'kmeans_size': kmeans_size,
            'hierarchical_size': hierarchical_size,
            'kmeans_percentage': (kmeans_size / len(features)) * 100,
            'hierarchical_percentage': (hierarchical_size / len(features)) * 100
        })
    
    clustering_results['cluster_summary'] = pd.DataFrame(cluster_summary)
    
    # 8. Feature matrix for reference
    clustering_results['feature_comparison_matrix'] = feature_matrix
    
    print(f"Clustering completed. Created {optimal_k} clusters.")
    print(f"Silhouette score for K-means: {silhouette_score(feature_matrix_scaled, cluster_labels):.3f}")
    print(f"Silhouette score for Hierarchical: {silhouette_score(feature_matrix_scaled, hierarchical_labels):.3f}")
    
    return clustering_results

def analyze_cluster_treatment_pattern(comparison_list, treatment_metadata):
    """
    Analyze what treatment patterns are characteristic of a cluster
    """
    
    patterns = []
    treatment_map = treatment_metadata.set_index('treatment').to_dict('index')
    
    for comparison in comparison_list[:4]:  # Look at top 4 comparisons
        if '_vs_' in comparison:
            t1, t2 = comparison.split('_vs_', 1)
            
            if t1 in treatment_map and t2 in treatment_map:
                t1_meta = treatment_map[t1]
                t2_meta = treatment_map[t2]
                
                # Determine what changes between treatments
                if t1_meta['control'] != t2_meta['control']:
                    patterns.append('control_vs_experimental')
                elif t1_meta['salinity'] != t2_meta['salinity']:
                    patterns.append('salinity_effect')
                elif t1_meta['step'] != t2_meta['step']:
                    patterns.append('temporal_effect')
                else:
                    patterns.append('within_condition')
    
    # Return the most common pattern
    if patterns:
        pattern_counts = {p: patterns.count(p) for p in set(patterns)}
        return max(pattern_counts, key=pattern_counts.get)
    else:
        return 'unknown_pattern'


def comprehensive_advanced_analysis(df_diff, treatment_metadata, output_file='advanced_analysis_results.xlsx'):
    """
    Comprehensive implementation of temporal, magnitude, and clustering analyses.
    Saves all results to an Excel file with multiple sheets.

    Parameters
    ----------
    df_diff : pd.DataFrame
        Differential expression dataframe with beta_A, beta_B, treatment_1, treatment_2, etc.
    treatment_metadata : pd.DataFrame
        Metadata dataframe with treatment info, control, salinity, and step columns.
    output_file : str
        Filename for output Excel file.
    
    Returns
    -------
    all_results : dict
        Dictionary containing results from all analyses.
    """
    print("COMPREHENSIVE ADVANCED ANALYSIS")
    print("="*60)
    
    # Run all three analyses
    temporal_results = analyze_temporal_patterns(df_diff, treatment_metadata)
    magnitude_results = analyze_magnitude_patterns(df_diff, treatment_metadata)
    clustering_results = analyze_clustering_patterns(df_diff, treatment_metadata)
    
    # Combine all results
    all_results = {}
    all_results.update(temporal_results)
    all_results.update(magnitude_results)
    all_results.update(clustering_results)
    
    # Write to Excel
    print(f"\nWriting all results to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in all_results.items():
            if isinstance(df, pd.DataFrame):
                clean_sheet_name = sheet_name.replace('_', ' ').title()[:31]
                df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
                print(f"  - {clean_sheet_name}: {len(df)} rows")
    
    return all_results
