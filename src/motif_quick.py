import polars as pl

def get_nearby_funcs(gene_calls_path, function_calls, positions_bed):

    positions_df = pl.read_csv(str(positions_bed), separator="\t", has_header=False, new_columns=["contig", "start", "end", "name", "score", "strand"]).with_columns((pl.col("strand") == "+").alias("strand"), pl.col("start").alias("position"))

    results = []
    
    gene_calls = pl.read_csv(gene_calls_path, separator="\t")

    # Map direction to +/-
    gene_calls = gene_calls.with_columns(pl.col("direction").replace_strict({"f": True, "r": False}))
    gene_calls = gene_calls.rename({"direction": "strand"})
    
    for row in positions_df.iter_rows(named=True):  # Iterate over rows
        contig = row["contig"]
        position = row["position"]
        strand = row["strand"]

        # Compute distances
        genes = gene_calls.filter(pl.col("contig") == contig, pl.col("strand") == strand).with_columns(
            (pl.col("start") - position).abs().alias("distance_to_start"),
            (pl.col("stop") - position).abs().alias("distance_to_end"),
            (pl.col("start") - position).alias("distance_to_start_raw"),
            (pl.col("stop") - position).alias("distance_to_end_raw"),
        )

        # Get nearest gene for start and end
        nearest_start = genes.sort("distance_to_start").head(1).rename({"gene_callers_id": "gene_callers_id_start"}).select("gene_callers_id_start", "distance_to_start", "distance_to_start_raw")
        nearest_end = genes.sort("distance_to_end").head(1).rename({"gene_callers_id": "gene_callers_id_end"}).select("gene_callers_id_end", "distance_to_end", "distance_to_end_raw")
        
        is_in_gene = gene_calls.filter(pl.col("contig") == contig, pl.col("strand") == strand, pl.col("start") <= position, pl.col("stop") >= position).select(pl.col("gene_callers_id").alias("gene_callers_id_in_gene"))        
        
        # Combine results and include original query information
        nearest_combined = pl.concat([nearest_start, nearest_end, is_in_gene], how="horizontal")

        results.append(nearest_combined)

    # Concatenate all results into a single DataFrame
    results = pl.concat(results, how="vertical")
    
    results = pl.concat([positions_df, results], how="horizontal")
    
    functions = pl.read_csv(function_calls, separator="\t", has_header=True).with_columns(pl.col("gene_callers_id").cast(pl.Int64))    
    
    results = results.join(functions, left_on="gene_callers_id_start", right_on="gene_callers_id", how="left").rename({"function": "function_start", "source": "source_start"})
    results = results.join(functions, left_on="gene_callers_id_end", right_on="gene_callers_id", how="left").rename({"function": "function_end", "source": "source_end"})
    
    # Filter only for rows where source_start and source_end are KOfam
    results = results.filter(pl.col("source_start") == "KOfam", pl.col("source_end") == "KOfam")
    
    # Write to CSV
    results.write_csv("nearby_functions.csv", separator=",")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Get nearby functions for a given set of positions.")
    parser.add_argument("gene_calls", type=str, help="Path to the gene calls file.")
    parser.add_argument("function_calls", type=str, help="Path to the function calls file.")
    parser.add_argument("positions_bed", type=str, help="Path to the BED file with positions.")

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.gene_calls):
        raise FileNotFoundError(f"Gene calls file not found: {args.gene_calls}")
    if not os.path.exists(args.function_calls):
        raise FileNotFoundError(f"Function calls file not found: {args.function_calls}")
    if not os.path.exists(args.positions_bed):
        raise FileNotFoundError(f"Positions BED file not found: {args.positions_bed}")

    get_nearby_funcs(args.gene_calls, args.function_calls, args.positions_bed)