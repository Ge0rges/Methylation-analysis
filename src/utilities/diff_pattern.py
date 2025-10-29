"""
Functions to identify features with interesting variation patterns across treatment groups.

This module provides tools to find features that are constant or changing in specific
treatment groups based on differential expression data.
"""

import polars as pl
from typing import List, Dict, Optional
from xlsxwriter import Workbook


def check_changes_within_group(
    df_diff: pl.DataFrame,
    group_treatments: List[str],
    feature_cols: List[str] = ["contig", "strand", "position"]
) -> pl.DataFrame:
    """
    Identify features that show significant changes within a group of treatments.
    
    Parameters
    ----------
    df_diff : pl.DataFrame
        DataFrame with columns: contig, strand, position, treatment_1, treatment_2, significant
    group_treatments : List[str]
        List of treatment names to check for within-group changes
    feature_cols : List[str]
        Column names that together define a unique feature
        
    Returns
    -------
    pl.DataFrame
        DataFrame with feature columns and a boolean 'changes_within' column
    """
    if len(group_treatments) < 2:
        # No comparisons possible, return all features as not changing
        unique_features = df_diff.select(feature_cols).unique()
        return unique_features.with_columns(pl.lit(False).alias("changes_within"))
    
    # Filter to comparisons within this group
    within_group = df_diff.filter(
        (pl.col("treatment_1").is_in(group_treatments)) &
        (pl.col("treatment_2").is_in(group_treatments))
    )
    
    # Find features with at least one significant comparison
    changing_features = (
        within_group
        .filter(pl.col("significant"))
        .select(feature_cols)
        .unique()
        .with_columns(pl.lit(True).alias("changes_within"))
    )
    
    # Get all features and mark those not changing
    all_features = df_diff.select(feature_cols).unique()
    result = all_features.join(
        changing_features,
        on=feature_cols,
        how="left"
    ).with_columns(
        pl.col("changes_within").fill_null(False)
    )
    
    return result


def check_changes_between_groups(
    df_diff: pl.DataFrame,
    group1_treatments: List[str],
    group2_treatments: List[str],
    feature_cols: List[str] = ["contig", "strand", "position"]
) -> pl.DataFrame:
    """
    Identify features that show significant changes between two treatment groups.
    
    Parameters
    ----------
    df_diff : pl.DataFrame
        DataFrame with columns: contig, strand, position, treatment_1, treatment_2, significant
    group1_treatments : List[str]
        First group of treatments
    group2_treatments : List[str]
        Second group of treatments
    feature_cols : List[str]
        Column names that together define a unique feature
        
    Returns
    -------
    pl.DataFrame
        DataFrame with feature columns and a boolean 'changes_between' column
    """
    # Filter to comparisons between groups
    between_groups = df_diff.filter(
        ((pl.col("treatment_1").is_in(group1_treatments)) & 
         (pl.col("treatment_2").is_in(group2_treatments))) |
        ((pl.col("treatment_1").is_in(group2_treatments)) & 
         (pl.col("treatment_2").is_in(group1_treatments)))
    )
    
    # Find features with at least one significant comparison
    changing_features = (
        between_groups
        .filter(pl.col("significant"))
        .select(feature_cols)
        .unique()
        .with_columns(pl.lit(True).alias("changes_between"))
    )
    
    # Get all features and mark those not changing
    all_features = df_diff.select(feature_cols).unique()
    result = all_features.join(
        changing_features,
        on=feature_cols,
        how="left"
    ).with_columns(
        pl.col("changes_between").fill_null(False)
    )
    
    return result


def find_constant_within_changing_between(
    df_diff: pl.DataFrame,
    constant_group: List[str],
    comparison_group: List[str],
    feature_cols: List[str] = ["contig", "strand", "position"],
    description: Optional[str] = None
) -> pl.DataFrame:
    """
    Find features constant within a group but different when compared to another group.
    
    Parameters
    ----------
    df_diff : pl.DataFrame
        DataFrame with differential expression results
    constant_group : List[str]
        Treatments where features should be constant internally
    comparison_group : List[str]
        Treatments to compare against
    feature_cols : List[str]
        Column names that define unique features
    description : Optional[str]
        Description to add to results
        
    Returns
    -------
    pl.DataFrame
        Features matching the pattern with description column
    """
    # Check if constant within group
    constant_check = check_changes_within_group(df_diff, constant_group, feature_cols)
    constant_within = constant_check.filter(~pl.col("changes_within")).select(feature_cols)
    
    # Check if different between groups
    between_check = check_changes_between_groups(
        df_diff, constant_group, comparison_group, feature_cols
    )
    different_between = between_check.filter(pl.col("changes_between")).select(feature_cols)
    
    # Find intersection
    result = constant_within.join(different_between, on=feature_cols, how="inner")
    
    if description:
        result = result.with_columns(pl.lit(description).alias("description"))
    
    return result


def find_patterns(
    df_diff: pl.DataFrame,
    pattern_config: Dict,
    feature_cols: List[str] = ["contig", "strand", "position"]
) -> pl.DataFrame:
    """
    Generic function to find features matching complex patterns.
    
    Parameters
    ----------
    df_diff : pl.DataFrame
        DataFrame with columns: contig, strand, position, treatment_1, treatment_2, significant
    pattern_config : Dict
        Configuration dictionary specifying the pattern to find. Keys:
        - 'constant_within': List[List[str]] - list of list of treatments that should be internally constant, inner lists are ANDed together
        - 'changing_within': List[List[str]] - list of list of treatments that should be internally changing, inner lists are ANDed together
        - 'constant_between': List[List[List[str]]] - list of List[group 1, group 2] that should be constant between each other, inner lists are ANDed together, group 1 and group 2 are lists.
        - 'changing_between': List[List[List[str]]] - list of List[group 1, group 2] that should be changing between each other, inner lists are ANDed together, group 1 and group 2 are lists.
        - 'description': str - description of the pattern
    feature_cols : List[str]
        Column names that define unique features
        
    Returns
    -------
    pl.DataFrame
        Features matching all specified conditions
    """
    all_features = df_diff.select(feature_cols).unique()
    matching_features = all_features
    
    # Apply constant_within constraints
    if 'constant_within' in pattern_config:
        for group in pattern_config['constant_within']:
            check = check_changes_within_group(df_diff, group, feature_cols)
            constant = check.filter(~pl.col("changes_within")).select(feature_cols)
            matching_features = matching_features.join(constant, on=feature_cols, how="inner")
    
    # Apply changing_within constraints
    if 'changing_within' in pattern_config:
        for group in pattern_config['changing_within']:
            check = check_changes_within_group(df_diff, group, feature_cols)
            changing = check.filter(pl.col("changes_within")).select(feature_cols)
            matching_features = matching_features.join(changing, on=feature_cols, how="inner")
    
    # Apply constant_between constraints
    if 'constant_between' in pattern_config:
        for group1, group2 in pattern_config['constant_between']:
            check = check_changes_between_groups(df_diff, group1, group2, feature_cols)
            constant = check.filter(~pl.col("changes_between")).select(feature_cols)
            matching_features = matching_features.join(constant, on=feature_cols, how="inner")
    
    # Apply changing_between constraints
    if 'changing_between' in pattern_config:
        for group1, group2 in pattern_config['changing_between']:
            check = check_changes_between_groups(df_diff, group1, group2, feature_cols)
            changing = check.filter(pl.col("changes_between")).select(feature_cols)
            matching_features = matching_features.join(changing, on=feature_cols, how="inner")
    
    # Add description if provided
    if 'description' in pattern_config:
        matching_features = matching_features.with_columns(
            pl.lit(pattern_config['description']).alias("description")
        )
    
    return matching_features


# Example usage patterns
def get_patterns(treatment_metadata: pl.DataFrame) -> Dict[str, Dict]:
    """
    Generate example pattern configurations based on treatment metadata.
    
    Parameters
    ----------
    treatment_metadata : pl.DataFrame
        DataFrame with columns: treatment, control (bool), salinity (bool), step (int)
        
    Returns
    -------
    Dict[str, Dict]
        Dictionary of named pattern configurations
    """
    # Extract treatment groups
    controls = treatment_metadata.filter(pl.col("control")).select("treatment").to_series().to_list()
    experimentals = treatment_metadata.filter(~pl.col("control")).select("treatment").to_series().to_list()
    high_sal = treatment_metadata.filter(pl.col("salinity")).select("treatment").to_series().to_list()
    low_sal = treatment_metadata.filter(~pl.col("salinity")).select("treatment").to_series().to_list()
    
    patterns = {
        "constant_throughout": {
            "constant_within": [controls + experimentals],
            "description": "Never significant in any comparison"
        },
        
        "constant_controls_changing_experimental": {
            "constant_within": [controls],
            "changing_within": [experimentals],
            "description": "Constant in controls, changes in experimental"
        },
        
        "changing_controls_constant_experimental": {
            "changing_within": [controls],
            "constant_within": [experimentals],
            "description": "Changes in controls, constant in experimental"
        },
        
        "constant_low_sal_changing_high_sal": {
            "constant_within": [low_sal],
            "changing_within": [high_sal],
            "description": "Constant in low salinity, changes in high salinity"
        },
        
        "constant_high_sal_changing_low_sal": {
            "constant_within": [high_sal],
            "changing_within": [low_sal],
            "description": "Constant in high salinity, changes in low salinity"
        },
        
        "constant_low_sal_different_vs_high_sal": {
            "constant_within": [low_sal],
            "changing_between": [[low_sal, high_sal]],
            "description": "Constant within low salinity but different from high salinity"
        },

        "memory_pattern": {
            "changing_between": [
                [treatment_metadata.filter(pl.col("step") == 1, ~pl.col("control")).select("treatment").to_series().to_list(),
                 treatment_metadata.filter(pl.col("step") == 15, ~pl.col("control")).select("treatment").to_series().to_list()],
            ],
            "constant_between": [
                [treatment_metadata.filter(pl.col("step") == 14, ~pl.col("control")).select("treatment").to_series().to_list(),
                    treatment_metadata.filter(pl.col("step") == 15, ~pl.col("control")).select("treatment").to_series().to_list()]
            ],
            "description": "Memory pattern across steps"
        }
    }
    
    return patterns


def analyze_differential_expression_patterns(
    df_diff: pl.DataFrame,
    treatment_metadata: pl.DataFrame,
    output_file: Optional[str] = None
) -> pl.DataFrame:
    """
    Analyze differential expression patterns based on treatment metadata.
    
    Parameters
    ----------
    df_diff : pl.DataFrame
        DataFrame with differential expression results
    treatment_metadata : pl.DataFrame
        DataFrame with treatment metadata

    Returns
    -------
    pl.DataFrame
        Features matching various patterns with descriptions
    """
    patterns = get_patterns(treatment_metadata)
    
    all_results = {}
    for name, config in patterns.items():
        result = find_patterns(df_diff, config)
        all_results[name] = result

    # Write excel
    if output_file:
        with Workbook(output_file) as wb:
            for pattern_name, result_df in all_results.items():
                result_df.write_excel(wb, worksheet=pattern_name[:31])  # Excel sheet name limit

    df = pl.concat(all_results.values()).unique()
    
    # Merge together descriptions if multiple patterns matched, concatenated with a "/"
    df = df.group_by(["contig", "strand", "position"]).agg(pl.col("description").str.concat("/"))
    return df