import pandas as pd
from supabase import create_client, Client

# Supabase Credentials 
SUPABASE_URL = "xx"
SUPABASE_KEY = "xx"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Config 
TABLE_NAME = "allianz_dashboard_data"
CHUNK_SIZE = 1000  # Supabase default page size
rows = []
offset = 0

# Page through until Supabase returns an empty batch
while True:
    print(f"Fetching rows {offset} to {offset + CHUNK_SIZE}...")
    response = supabase.table(TABLE_NAME).select("*").range(offset, offset + CHUNK_SIZE - 1).execute()
    chunk = response.data
    if not chunk:
        break
    rows.extend(chunk)
    offset += CHUNK_SIZE

# Save 
df = pd.DataFrame(rows)
df.to_csv("allianz_combined_data.csv", index=False)
print(f"Pulled {len(df)} rows")
print(f"Min date: {df['date'].min()}")
print(f"Max date: {df['date'].max()}")
