import pandas as pd

def remove_duplicate_id_column(df):
    # Find all columns with the header 'id'
    id_columns = [col for col in df.columns if col == 'id']
    print(id_columns)
    
    # If there are multiple 'id' columns, drop the second one
    for col in id_columns[:]:
        df.drop(columns=col, inplace=True)
    
    return df

# Example usage
if __name__ == "__main__":
    # Sample DataFrame
    data = pd.read_csv(r'/home/edwheeler/Documents/code/ultrack_scripts/outputs/crop_1_tracks.csv')

    df = pd.DataFrame(data)
    print("Original DataFrame:")
    print(df.head())

    # Remove the duplicate 'id' column
    df = df[['t', 'z', 'y', 'x']]
    print("\nDataFrame after removing duplicate 'id' column:")
    print(df.head())
    # Save the modified DataFrame to a new CSV file
    df.to_csv(r'/home/edwheeler/Documents/code/ultrack_scripts/outputs/crop_1_tracks_no_duplicate_id.csv', index=False)