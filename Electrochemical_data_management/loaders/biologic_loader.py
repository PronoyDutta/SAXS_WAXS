# loaders/biologic_loader.py

import os
import pandas as pd
from galvani import BioLogic as BL

def load_biologic_gcd_mpr(filepaths):
    results = {}

    for path in filepaths:
        filename = os.path.basename(path)
        result = {'data': None, 'meta': None, 'error': None}

        try:
            exp = BL.MPRfile(path)
            df = pd.DataFrame(exp.data)
            df.columns = df.columns.str.strip()

            # Rename relevant columns
            df = df.rename(columns={
                'time/s': 'time_s',
                'Ewe/V': 'voltage_V',
                'control/V/mA': 'current_mA',
                'Ns': 'step',
                'Q charge/discharge/mA.h': 'capacity_mAh',
            })

            # Drop unnecessary columns
            drop_cols = [col for col in ['flag', 'Current range', 'P/W'] if col in df.columns]
            df.drop(columns=drop_cols, inplace=True, errors='ignore')

            # Assign current_sign
            df['current_sign'] = df['current_mA'].apply(lambda x: 0 if abs(x) < 1e-5 else (1 if x > 0 else -1))

            # Initialize cycle and half_cycle tracking
            cycle = 1  # First active cycle will be 1
            half = 0
            prev_sign = 0
            cycle_nums = []
            half_cycles = []

            for sign in df['current_sign']:
                if sign == 0:
                    # Rest state
                    half_cycles.append(0)
                    cycle_nums.append(0)
                else:
                    if prev_sign == 0:
                        half = 1
                    elif prev_sign * sign < 0:
                        if half == 1:
                            half = 2
                        else:
                            half = 1
                            cycle += 1
                    half_cycles.append(half)
                    cycle_nums.append(cycle)
                prev_sign = sign

            df['cycle_number'] = cycle_nums
            df['half_cycle'] = half_cycles

            # Final column selection
            df = df[['cycle_number', 'half_cycle', 'time_s', 'current_mA', 'voltage_V', 'capacity_mAh']]

            # Extract metadata
            meta = {
                'num_half_cycles': df['half_cycle'].nunique(),
                'max_voltage': df['voltage_V'].max(),
                'min_voltage': df['voltage_V'].min()
            }

            result['data'] = df
            result['meta'] = meta

        except Exception as e:
            result['error'] = str(e)

        results[filename] = result

    return results
