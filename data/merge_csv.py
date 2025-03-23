import os
import pandas as pd
import glob

base_dir = "data"
output_dir = "merged_biomes"
os.makedirs(output_dir, exist_ok=True)

# Loop through each biome folder
for biome in os.listdir(base_dir):
    biome_path = os.path.join(base_dir, biome)
    
    if os.path.isdir(biome_path):
        csv_files = glob.glob(os.path.join(biome_path, "*.csv"))
        
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if 'DateTime' not in df.columns:
                    print(f"Warning: No timestamp column in {file}, skipping...")
                    continue
                
                # Get the filename without extension
                sensor_name = os.path.splitext(os.path.basename(file))[0]
                
                # Rename columns except timestamp
                df = df.rename(columns={col: f"{sensor_name}_{col}" for col in df.columns if col != "DateTime"})
                
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")
        
        # Only proceed if we have dataframes to merge
        if len(dfs) > 0:
            # Merge all dataframes on 'timestamp'
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on='DateTime', how='outer')
            
            # Save merged biome data
            output_path = os.path.join(output_dir, f"{biome}_merged.csv")
            merged_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
        else:
            print(f"No valid CSV files found in {biome_path}")
