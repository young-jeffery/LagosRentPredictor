import pandas as pd

def get_distribution_counts():
    try:
        # Load the newly mapped dataset
        df = pd.read_csv('lagos_rent_strictly_mapped.csv')
    except FileNotFoundError:
        print("Error: 'lagos_rent_strictly_mapped.csv' not found. Run the cleaning script first.")
        return

    # Count the occurrences of each Main District and Sub-Location
    main_counts = df['Main_District'].value_counts()
    sub_counts = df['Official_Sub_Location'].value_counts()

    # Export to a readable text file
    output_file = 'Location_Distributions.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== MAIN DISTRICT DISTRIBUTION ===\n")
        f.write(main_counts.to_string())
        f.write("\n\n=== SUB-LOCATION DISTRIBUTION ===\n")
        f.write(sub_counts.to_string())

    print(f"Success! Open '{output_file}' to see your numbers.")

if __name__ == "__main__":
    get_distribution_counts()