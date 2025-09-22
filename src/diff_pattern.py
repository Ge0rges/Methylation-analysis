import pandas as pd
from xlsxwriter import Workbook
import polars as pl


def analyze_differential_expression_patterns(df_diff, treatment_metadata, output_file) -> dict[str, pl.DataFrame]:
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

    results['1_constant_throughout'] = pl.from_dict({
        'feature_id': constant_features,
        'contig': [f.split('<>')[0] for f in constant_features],
        'strand': [bool(f.split('<>')[1]) for f in constant_features],
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
                        'strand': (feature.split('<>')[1] == "True"),
                        'position': int(feature.split('<>')[2]),
                        'treatment_type': f"control_{control_val}_salinity_{salinity_val}",
                        'description': f"Constant in control={control_val}, salinity={salinity_val} but changes elsewhere"
                    })

    results['2_constant_in_one_type'] = pl.from_dicts(type_specific_constant)

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
                        'strand': (feature.split('<>')[1] == "True"),
                        'position': int(feature.split('<>')[2]),
                        'treatment_type': f"control_{control_val}_salinity_{salinity_val}",
                        'description': f"Changes in control={control_val}, salinity={salinity_val} but constant elsewhere"
                    })

    results['3_changing_in_one_type'] = pl.from_dicts(type_specific_changing, schema={
        'feature_id': pl.Utf8,
        'contig': pl.Utf8,
        'strand': pl.Boolean,
        'position': pl.Int64,
        'treatment_type': pl.Utf8,
        'description': pl.Utf8
    })

    # 4. Features constant in controls but change in experimentals
    print("4. Finding features constant in controls but changing in experimentals...")
    control_constant_exp_change = []
    for feature in features:
        control_constant = not feature_changes_within_group(feature, control_treatments)
        exp_changes = feature_changes_within_group(feature, experimental_treatments)

        if control_constant and exp_changes:
            control_constant_exp_change.append(feature)

    results['4_constant_controls_change_experimental'] = pl.from_dict({
        'feature_id': control_constant_exp_change,
        'contig': [f.split('<>')[0] for f in control_constant_exp_change],
        'strand': [bool(f.split('<>')[1]) for f in control_constant_exp_change],
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

    results['5_change_controls_constant_experimental'] = pl.from_dict({
        'feature_id': control_change_exp_constant,
        'contig': [f.split('<>')[0] for f in control_change_exp_constant],
        'strand': [bool(f.split('<>')[1]) for f in control_change_exp_constant],
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

    results['6_constant_high_salinity_change_low_salinity'] = pl.from_dict({
        'feature_id': high_sal_constant_low_sal_change,
        'contig': [f.split('<>')[0] for f in high_sal_constant_low_sal_change],
        'strand': [bool(f.split('<>')[1]) for f in high_sal_constant_low_sal_change],
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

    results['7_change_high_salinity_constant_low_salinity'] = pl.from_dict({
        'feature_id': high_sal_change_low_sal_constant,
        'contig': [f.split('<>')[0] for f in high_sal_change_low_sal_constant],
        'strand': [bool(f.split('<>')[1]) for f in high_sal_change_low_sal_constant],
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

    results['8_control_step1_vs_other_control'] = pl.from_dict({
        'feature_id': control_step1_diffs,
        'contig': [f.split('<>')[0] for f in control_step1_diffs],
        'strand': [bool(f.split('<>')[1]) for f in control_step1_diffs],
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

    results['9_lowSal_step1_constant_diff_highSal_step1'] = pl.from_dict({
        'feature_id': low_step1_constant_high_diff,
        'contig': [f.split('<>')[0] for f in low_step1_constant_high_diff],
        'strand': [bool(f.split('<>')[1]) for f in low_step1_constant_high_diff],
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

    results['10_constant_high_sal_step1_and_high_sal_control'] = pl.from_dict({
        'feature_id': both_constant_features,
        'contig': [f.split('<>')[0] for f in both_constant_features],
        'strand': [bool(f.split('<>')[1]) for f in both_constant_features],
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

    results['11_constant_exp_early_late_diff_control'] = pl.from_dict({
        'feature_id': both_exp_constant,
        'contig': [f.split('<>')[0] for f in both_exp_constant],
        'strand': [bool(f.split('<>')[1]) for f in both_exp_constant],
        'position': [int(f.split('<>')[2]) for f in both_exp_constant],
        'description': ['In control=False, constant in steps [1,2], and constant in [14,15], but at one point different to any control'] * len(both_exp_constant)
    })

    # 12. Features that shift between consecutive control=False treatments but never shift in any control comparison
    print("Finding features that shift between consecutive control=False treatments but never in any control comparison...")

    # Get all control=False treatments ordered by step
    exp_treatments = treatment_metadata[treatment_metadata['control'] == False].sort_values('step')
    exp_treatment_list = exp_treatments['treatment'].tolist()

    # Get all control=True treatments  
    control_treatments = treatment_metadata[treatment_metadata['control'] == True]['treatment'].tolist()

    # Track shifts for each consecutive pair
    consecutive_pair_shifts = {}
    consecutive_exp_shift_control_stable = []
    feature_shift_details = []

    for feature in features:
        # Check if feature shifts between ANY consecutive pair of experimental treatments
        shifts_between_consecutive_exp = False
        feature_shifts = []
        
        if len(exp_treatment_list) >= 2:
            feature_data = df_diff[df_diff['feature_id'] == feature]
            
            # Check each consecutive pair of experimental treatments
            for i in range(len(exp_treatment_list) - 1):
                t1 = exp_treatment_list[i]
                t2 = exp_treatment_list[i + 1]
                pair_name = f"{t1}_to_{t2}"
                
                comparison = feature_data[
                    ((feature_data['treatment_1'] == t1) & (feature_data['treatment_2'] == t2)) |
                    ((feature_data['treatment_1'] == t2) & (feature_data['treatment_2'] == t1))
                ]
                
                # If significant difference found between these consecutive steps
                if not comparison.empty and comparison['significant'].any():
                    shifts_between_consecutive_exp = True
                    feature_shifts.append(pair_name)
                    
                    # Track count for this pair
                    if pair_name not in consecutive_pair_shifts:
                        consecutive_pair_shifts[pair_name] = 0
        
        # Check if feature NEVER shifts within control comparisons
        never_shifts_in_control = not feature_changes_within_group(feature, control_treatments)
        
        if shifts_between_consecutive_exp and never_shifts_in_control:
            consecutive_exp_shift_control_stable.append(feature)
            
            # Add to counts for each pair this feature shifted in
            for pair in feature_shifts:
                consecutive_pair_shifts[pair] += 1
            
            # Store detailed information for this feature
            feature_shift_details.append({
                'feature_id': feature,
                'contig': feature.split('<>')[0],
                'strand': (feature.split('<>')[1] == "True"),
                'position': int(feature.split('<>')[2]),
                'shifting_pairs': '; '.join(feature_shifts),
                'num_pairs_shifted': len(feature_shifts)
            })

    # Create main results DataFrame with detailed shift information
    results['12_consecutive_exp_shifts_stable_controls'] = pl.from_dicts(feature_shift_details)

    # Print summary
    print(f"\nFound {len(consecutive_exp_shift_control_stable)} features that shift between consecutive experimental treatments but never in controls")
    print("\nShift counts by consecutive pair:")
    for pair, count in consecutive_pair_shifts.items():
        print(f"  {pair}: {count} features")

    # Write results to Excel using polars
    print(f"\nWriting results to {output_file}...")
    with Workbook(output_file) as workbook: 
        for sheet_name, df in results.items():
            clean_sheet_name = sheet_name.replace('_', ' ').title()[:31]

            if df.is_empty():
                df = pl.from_dict({"no data": []})
                df.write_excel(workbook=workbook, worksheet=clean_sheet_name)
                continue
            
            # Join df with df_diff on contig,strand,position
            df = df.join(pl.from_pandas(df_diff), on=['contig', 'strand', 'position'], how='left')
            df.write_excel(workbook=workbook, worksheet=clean_sheet_name)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF FEATURE ANALYSIS")
    print("="*60)
    for key, df in results.items():
        print(f"{key.replace('_', ' ').title()}: {len(df)} features")
    print(f"\nTotal unique features analyzed: {len(features)}")

    return results
