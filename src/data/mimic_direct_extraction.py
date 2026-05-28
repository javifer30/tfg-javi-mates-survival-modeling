import pandas as pd
import numpy as np
import os
import gc
from collections import defaultdict

# Configuration
RAW_DATA_PATH = r"C:\Users\Javi\Desktop\A 5º Mat-Info\TFG Mates\Mathematics-Dissertation-Survival-Modeling-Javier\data\raw\mimic-iv-3.1"
OUTPUT_PATH = r"C:\Users\Javi\Desktop\A 5º Mat-Info\TFG Mates\Mathematics-Dissertation-Survival-Modeling-Javier\data\processed\mimic_extraction"

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_table(name, module='hosp', usecols=None):
    path = os.path.join(RAW_DATA_PATH, module, f"{name}.csv.gz")
    print(f"Loading {name} from {path}...")
    return pd.read_csv(path, compression='gzip', usecols=usecols)

def generate_labels():
    print("Generating labels...")
    
    # Load required tables
    patients = load_table('patients', 'hosp')
    admissions = load_table('admissions', 'hosp')
    
    # MIMIC-IV v3.1 uses 'race' instead of 'ethnicity'
    if 'race' in admissions.columns:
        admissions = admissions.rename(columns={'race': 'ethnicity'})
        
    icustays = load_table('icustays', 'icu')
    
    # Merge
    print("Merging tables...")
    merged = icustays.merge(admissions, on=['subject_id', 'hadm_id'], how='inner')
    merged = merged.merge(patients, on='subject_id', how='inner')
    
    # Calculate Age
    # (extract(year from i.intime) - p.anchor_year + p.anchor_age)
    merged['intime'] = pd.to_datetime(merged['intime'])
    merged['outtime'] = pd.to_datetime(merged['outtime'])
    merged['age'] = merged['intime'].dt.year - merged['anchor_year'] + merged['anchor_age']
    
    # Filter
    # los > (5/24) and age > 17
    print("Filtering cohort...")
    mask = (merged['los'] > (5/24)) & (merged['age'] > 17)
    cohort = merged[mask].copy()
    
    # Rename and Select
    cohort = cohort.rename(columns={
        'subject_id': 'uniquepid',
        'hadm_id': 'patienthealthsystemstayid',
        'stay_id': 'patientunitstayid',
        'hospital_expire_flag': 'actualhospitalmortality',
        'los': 'actualiculos'
    })
    
    # Save labels
    labels_df = cohort[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 
                        'actualhospitalmortality', 'actualiculos', 'intime', 'outtime', 'age', 
                        'gender', 'ethnicity', 'first_careunit', 'admission_location', 'insurance']]
    
    output_file = os.path.join(OUTPUT_PATH, 'labels.csv')
    labels_df[['uniquepid', 'patienthealthsystemstayid', 'patientunitstayid', 'actualhospitalmortality', 'actualiculos']].to_csv(output_file, index=False)
    print(f"Saved {output_file} with {len(labels_df)} patients.")
    
    return labels_df

def generate_flat_features(labels_df):
    print("Generating flat features...")
    
    # Needed for Item IDs
    d_items = load_table('d_items', 'icu')
    
    target_labels = [
        'Admission Weight (Kg)', 
        'GCS - Eye Opening',
        'GCS - Motor Response',
        'GCS - Verbal Response',
        'Height (cm)'
    ]
    
    target_items = d_items[d_items['label'].isin(target_labels)][['itemid', 'label']]
    target_item_ids = target_items['itemid'].tolist()
    item_id_map = target_items.set_index('itemid')['label'].to_dict()
    
    # Prepare mapping for GCS/Weight/Height
    # We need to pivot these.
    
    cohort_stays = set(labels_df['patientunitstayid'])
    cohort_intimes = labels_df.set_index('patientunitstayid')['intime']
    
    collected_data = []
    
    chunksize = 1000000
    path = os.path.join(RAW_DATA_PATH, 'icu', "chartevents.csv.gz")
    
    print(f"Processing chartevents for flat features in chunks...")
    for chunk in pd.read_csv(path, compression='gzip', chunksize=chunksize, 
                             usecols=['stay_id', 'itemid', 'valuenum', 'charttime']):
        
        # Filter by Item ID
        chunk = chunk[chunk['itemid'].isin(target_item_ids)]
        if chunk.empty: continue
        
        # Filter by Cohort
        chunk = chunk[chunk['stay_id'].isin(cohort_stays)]
        if chunk.empty: continue
        
        # Filter by Time
        # date_part('hour', ch.charttime) - date_part('hour', i.intime) between -24 and 5
        # Note: SQL 'date_part('hour', ...)' extracts the hour of the day (0-23), NOT the difference in hours.
        # Wait, the SQL says: date_part('hour', ch.charttime) - date_part('hour', i.intime)
        # This seems buggy in the original SQL if they meant time difference. 
        # But 'flat_features.sql' uses: 
        # date_part('hour', ch.charttime) - date_part('hour', i.intime) between -24 and 5
        # If I enter at 23:00 and chart is at 01:00 next day, 1 - 23 = -22. Correct?
        # But if I enter at 01:00 and chart is at 23:00 previous day, 23 - 1 = 22.
        # Actually, usually one uses `extract(epoch from ...)` for difference.
        # HOWEVER, the original SQL uses `date_part('hour', ...)` which is literally 0-23.
        # This implies it might be checking the "hour of day" difference, which is weird.
        # Let's look closer at the SQL provided in the prompt:
        # `and date_part('hour', ch.charttime) - date_part('hour', i.intime) between -24 and 5`
        # This logic seems flawed for day boundaries unless they rely on `day` being same/adjacent.
        # BUT, looking at `timeseries.sql`, they use `epoch` differences.
        # I will assume the INTENTION was "Hours from admission".
        # Let's check typical MIMIC preprocessing. Usually it is `charttime - intime`.
        # I will implement (charttime - intime) in hours between -24 and 5.
        
        chunk['charttime'] = pd.to_datetime(chunk['charttime'])
        chunk['intime'] = chunk['stay_id'].map(cohort_intimes)
        
        diff_hours = (chunk['charttime'] - chunk['intime']).dt.total_seconds() / 3600
        
        mask = (diff_hours >= -24) & (diff_hours <= 5)
        filtered = chunk[mask]
        
        if not filtered.empty:
            collected_data.append(filtered[['stay_id', 'itemid', 'valuenum']])
            
    if collected_data:
        full_data = pd.concat(collected_data)
        
        # Pivot
        # We need to average if multiple values exist
        full_data['label'] = full_data['itemid'].map(item_id_map)
        
        # Map labels to variable names expected by `flat_features.sql` crosstab
        # 'Admission Weight (Kg)' -> weight
        # 'GCS - Eye Opening' -> eyes
        # 'GCS - Motor Response' -> motor
        # 'GCS - Verbal Response' -> verbal
        # 'Height (cm)' -> height
        
        label_to_col = {
            'Admission Weight (Kg)': 'weight',
            'GCS - Eye Opening': 'eyes',
            'GCS - Motor Response': 'motor',
            'GCS - Verbal Response': 'verbal',
            'Height (cm)': 'height'
        }
        full_data['var_name'] = full_data['label'].map(label_to_col)
        
        pivoted = full_data.groupby(['stay_id', 'var_name'])['valuenum'].mean().unstack()
        pivoted.index.name = 'patientunitstayid'
        
    else:
        pivoted = pd.DataFrame(index=pd.Index([], name='patientunitstayid'))

    # Join with static info from labels
    # select distinct i.stay_id as patientunitstayid, p.gender, (extract(year from i.intime) - p.anchor_year + p.anchor_age) as age,
    # adm.ethnicity, i.first_careunit, adm.admission_location, adm.insurance
    
    flat_base = labels_df[['patientunitstayid', 'gender', 'age', 'ethnicity', 'first_careunit', 'admission_location', 'insurance', 'intime']].copy()
    flat_base['hour'] = flat_base['intime'].dt.hour
    flat_base = flat_base.set_index('patientunitstayid')
    
    final_flat = flat_base.join(pivoted, how='left')
    
    # Drop intime as it's not in the final output of SQL (only used for 'hour')
    final_flat = final_flat.drop(columns=['intime'])
    
    output_file = os.path.join(OUTPUT_PATH, 'flat_features.csv')
    final_flat.to_csv(output_file)
    print(f"Saved {output_file}")
    
    return final_flat

def get_common_items(labels_df, table_name, module, item_col='itemid', time_col='charttime', val_col='valuenum', min_coverage=0.25, threshold_obs=0):
    print(f"Finding common items in {table_name}...")
    
    cohort_stays = set(labels_df['patientunitstayid'])
    cohort_intimes = labels_df.set_index('patientunitstayid')['intime']
    cohort_los = labels_df.set_index('patientunitstayid')['actualiculos'] # in days
    
    item_counts = defaultdict(set) # itemid -> set of stay_ids
    item_obs_counts = defaultdict(int) # itemid -> total observations (to filtering avg_obs)
    item_stay_obs_counts = defaultdict(lambda: defaultdict(int)) # itemid -> stay_id -> count
    
    chunksize = 2000000
    path = os.path.join(RAW_DATA_PATH, module, f"{table_name}.csv.gz")
    
    # Pre-calculate valid time windows?
    # Logic: (epoch(charttime) - epoch(intime))/(24*3600) between -1 and los
    # i.e. Time from admission is > -1 day AND < los days
    
    count_processed = 0
    for chunk in pd.read_csv(path, compression='gzip', chunksize=chunksize, usecols=[item_col, val_col, time_col, 'hadm_id' if table_name=='labevents' else 'stay_id']):
        # Filter NULL values
        chunk = chunk[chunk[val_col].notna()]
        
        # Link to Stay ID
        if table_name == 'labevents':
            # labevents has hadm_id. Need to map to stay_id?
            # labels_df has hadm_id and patientunitstayid.
            # Warning: One hadm_id can have multiple stay_ids.
            # The SQL joins labevents on hadm_id = la.hadm_id.
            # And `select la.stay_id ...`
            # If multiple stays, the lab event applies to ALL stays?
            # SQL logic: 
            # inner join ld_labels as la on la.hadm_id = l.hadm_id
            # This replicates the lab row for EACH stay of that admission.
            
            # Map hadm_id to list of stay_ids
            # But we only care if it falls in the time window of THAT stay.
            
            # This is complex to do efficiently.
            # Let's filter by hadm_ids in cohort first.
            chunk = chunk[chunk['hadm_id'].isin(labels_df['patienthealthsystemstayid'])]
            
            # Expand to stays?
            # Or just store by hadm_id and resolve later?
            # The time check depends on stay's intime.
            
            # Let's do a merge approach for the chunk
            # Merge chunk with labels on hadm_id
            # Columns in labels needed: hadm_id, stay_id, intime, los
            
            # Use a simplified labels df for merging
            temp_labels = labels_df[['patienthealthsystemstayid', 'patientunitstayid', 'intime', 'actualiculos']]
            chunk = chunk.merge(temp_labels, left_on='hadm_id', right_on='patienthealthsystemstayid', how='inner')
            
        else:
            # chartevents has stay_id
            chunk = chunk.rename(columns={'stay_id': 'patientunitstayid'})
            chunk = chunk[chunk['patientunitstayid'].isin(cohort_stays)]
            
            # Merge for intime/los
            temp_labels = labels_df[['patientunitstayid', 'intime', 'actualiculos']]
            chunk = chunk.merge(temp_labels, on='patientunitstayid', how='inner')

        if chunk.empty: continue

        # Time filter
        chunk[time_col] = pd.to_datetime(chunk[time_col])
        chunk['offset_days'] = (chunk[time_col] - chunk['intime']).dt.total_seconds() / (24*3600)
        
        mask = (chunk['offset_days'] >= -1) & (chunk['offset_days'] <= chunk['actualiculos'])
        valid = chunk[mask]
        
        # Count
        # We need:
        # 1. Count of distinct stays having this item
        # 2. Avg observations per stay (only for stays having it)
        
        # Group by item, stay -> count
        counts = valid.groupby([item_col, 'patientunitstayid']).size()
        
        # Aggregate into global
        for (item, stay), count in counts.items():
            item_counts[item].add(stay)
            item_stay_obs_counts[item][stay] += count

        count_processed += len(chunk)
        print(f"Processed {count_processed} raw rows...", end='\r')
        
    print("\nCalculating stats...")
    
    total_stays = labels_df['patientunitstayid'].nunique()
    threshold = total_stays * min_coverage
    
    selected_items = []
    
    for item, stays in item_counts.items():
        if len(stays) > threshold:
            # Check avg obs requirement
            # SQL: having avg(count) > 3 (labs) or 5 (chart)
            
            total_obs = sum(item_stay_obs_counts[item].values())
            avg_obs = total_obs / len(stays)
            
            if avg_obs > threshold_obs:
                selected_items.append(item)
                
    print(f"Selected {len(selected_items)} items for {table_name}.")
    return selected_items

def generate_timeseries(labels_df):
    print("Generating timeseries...")
    
    # Labs
    common_labs = get_common_items(labels_df, 'labevents', 'hosp', min_coverage=0.25, threshold_obs=3)
    
    # Extract Labs
    print("Extracting Labs Data...")
    
    # Load D_LABITEMS for names
    d_labitems = load_table('d_labitems', 'hosp')
    lab_map = d_labitems.set_index('itemid')['label'].to_dict()
    
    # We repeat the iteration to extract data for selected items
    # (Inefficient but memory safe)
    
    chunksize = 2000000
    lab_path = os.path.join(RAW_DATA_PATH, 'hosp', "labevents.csv.gz")
    
    collected_labs = []
    
    temp_labels = labels_df[['patienthealthsystemstayid', 'patientunitstayid', 'intime', 'actualiculos']]
    
    for chunk in pd.read_csv(lab_path, compression='gzip', chunksize=chunksize, usecols=['itemid', 'valuenum', 'charttime', 'hadm_id']):
        chunk = chunk[chunk['itemid'].isin(common_labs)]
        chunk = chunk[chunk['valuenum'].notna()]
        
        chunk = chunk.merge(temp_labels, left_on='hadm_id', right_on='patienthealthsystemstayid', how='inner')
        if chunk.empty: continue
        
        chunk['charttime'] = pd.to_datetime(chunk['charttime'])
        chunk['offset_days'] = (chunk['charttime'] - chunk['intime']).dt.total_seconds() / (24*3600)
        
        mask = (chunk['offset_days'] >= -1) & (chunk['offset_days'] <= chunk['actualiculos'])
        valid = chunk[mask]
        
        if not valid.empty:
            # Calculate labresultoffset (minutes)
            valid['labresultoffset'] = np.floor((valid['charttime'] - valid['intime']).dt.total_seconds() / 60)
            valid['labname'] = valid['itemid'].map(lab_map)
            
            collected_labs.append(valid[['patientunitstayid', 'labresultoffset', 'labname', 'valuenum']])
            
    if collected_labs:
        final_labs = pd.concat(collected_labs)
        final_labs = final_labs.rename(columns={'valuenum': 'labresult'})
        final_labs = final_labs.sort_values(['patientunitstayid', 'labresultoffset'])
        final_labs.to_csv(os.path.join(OUTPUT_PATH, 'timeserieslab.csv'), index=False)
        print(f"Saved timeserieslab.csv with {len(final_labs)} rows.")
        del final_labs, collected_labs
        gc.collect()

    # Chart
    common_chart = get_common_items(labels_df, 'chartevents', 'icu', min_coverage=0.25, threshold_obs=5)
    
    print("Extracting Chart Data...")
    
    d_items = load_table('d_items', 'icu')
    chart_map = d_items.set_index('itemid')['label'].to_dict()
    
    chart_path = os.path.join(RAW_DATA_PATH, 'icu', "chartevents.csv.gz")
    collected_chart = []
    
    temp_labels_chart = labels_df[['patientunitstayid', 'intime', 'actualiculos']]
    cohort_stays_set = set(labels_df['patientunitstayid'])
    
    for chunk in pd.read_csv(chart_path, compression='gzip', chunksize=chunksize, usecols=['itemid', 'valuenum', 'charttime', 'stay_id']):
        chunk = chunk[chunk['itemid'].isin(common_chart)]
        chunk = chunk[chunk['valuenum'].notna()]
        
        # Rename stay_id
        chunk = chunk.rename(columns={'stay_id': 'patientunitstayid'})
        chunk = chunk[chunk['patientunitstayid'].isin(cohort_stays_set)]
        
        chunk = chunk.merge(temp_labels_chart, on='patientunitstayid', how='inner')
        if chunk.empty: continue
        
        chunk['charttime'] = pd.to_datetime(chunk['charttime'])
        chunk['offset_days'] = (chunk['charttime'] - chunk['intime']).dt.total_seconds() / (24*3600)
        
        mask = (chunk['offset_days'] >= -1) & (chunk['offset_days'] <= chunk['actualiculos'])
        valid = chunk[mask]
        
        if not valid.empty:
            valid['chartoffset'] = np.floor((valid['charttime'] - valid['intime']).dt.total_seconds() / 60)
            valid['chartvaluelabel'] = valid['itemid'].map(chart_map)
            
            collected_chart.append(valid[['patientunitstayid', 'chartoffset', 'chartvaluelabel', 'valuenum']])
            
    if collected_chart:
        final_chart = pd.concat(collected_chart)
        final_chart = final_chart.rename(columns={'valuenum': 'chartvalue'})
        final_chart = final_chart.sort_values(['patientunitstayid', 'chartoffset'])
        final_chart.to_csv(os.path.join(OUTPUT_PATH, 'timeseries.csv'), index=False)
        print(f"Saved timeseries.csv with {len(final_chart)} rows.")
        del final_chart, collected_chart
        gc.collect()

    # Re-filtering labels and flat to match patients with timeseries
    # SQL: create materialized view ld_timeseries_patients as ... select distinct patientunitstayid from repeats
    
    print("Filtering final cohort based on available timeseries...")
    
    # We need to know which patients actually ended up in timeserieslab or timeseries
    ts_labs = pd.read_csv(os.path.join(OUTPUT_PATH, 'timeserieslab.csv'), usecols=['patientunitstayid'])
    ts_chart = pd.read_csv(os.path.join(OUTPUT_PATH, 'timeseries.csv'), usecols=['patientunitstayid'])
    
    valid_stays = set(ts_labs['patientunitstayid']).union(set(ts_chart['patientunitstayid']))
    
    # Update labels.csv and flat_features.csv
    labels_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'labels.csv'))
    labels_df = labels_df[labels_df['patientunitstayid'].isin(valid_stays)]
    labels_df.to_csv(os.path.join(OUTPUT_PATH, 'labels.csv'), index=False)
    
    flat_df = pd.read_csv(os.path.join(OUTPUT_PATH, 'flat_features.csv'))
    flat_df = flat_df[flat_df['patientunitstayid'].isin(valid_stays)]
    flat_df.to_csv(os.path.join(OUTPUT_PATH, 'flat_features.csv'), index=False)
    
    print(f"Final cohort size: {len(labels_df)}")

if __name__ == "__main__":
    labels = generate_labels()
    generate_flat_features(labels)
    generate_timeseries(labels)
