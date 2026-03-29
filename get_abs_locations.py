import pandas as pd

# Read suburbs and localities from the ABS allocation file
dfsubs = pd.read_excel('https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/allocation-files/SAL_2021_AUST.xlsx')

# Read LGAs from the ABS allocation file
dflgas = pd.read_excel('https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/allocation-files/LGA_2025_AUST.xlsx')
lganames = set(dflgas['LGA_NAME_2025'])
suburbs = set(dfsubs['SAL_NAME_2021'])

suburbs = suburbs.union(lganames)

# For each string, remove any parenthetical content and trim whitespace
def clean_name(name):
    if pd.isna(name):
        return None
    # Remove parenthetical content
    cleaned = name.split('(')[0].strip()
    return cleaned

suburbs = set(clean_name(name) for name in suburbs)
print(f"Extracted {len(suburbs)} unique suburb and LGA names.")
with open("au_suburbs.txt", "w") as f:
    for name in sorted(suburbs):
        f.write(name + "\n")