# Troubleshooting Guide

## Error: "No data found!"

### Problem
The script can't find your TEM data files.

### Solution Options

#### Option 1: Run from the correct directory (Recommended)

Place the script in the **same directory** as your numbered location folders:

```
your_project_folder/
  ├── CNN_target_locator_improved.py  ← Script here
  ├── 1700/                           ← Data folders here
  │   ├── 0moffset1.tem
  │   ├── 0moffset2.tem
  │   └── ...
  ├── 1900/
  │   ├── 0moffset1.tem
  │   └── ...
  └── ...
```

Then run:
```bash
cd your_project_folder
python CNN_target_locator_improved.py
```

#### Option 2: Set the data directory path

Edit the script to point to your data directory:

```python
# At the top of the script, change:
DATA_DIRECTORY = "."

# To (Windows example):
DATA_DIRECTORY = r"C:\Users\anthony\Documents\Work\data"

# Or (Linux/Mac example):
DATA_DIRECTORY = "/home/user/movingloop_data"
```

#### Option 3: Use relative path

If your data is in a subdirectory:

```python
# Script is in: /home/user/scripts/
# Data is in:   /home/user/data/

# Change to:
DATA_DIRECTORY = "../data"
```

### Expected Directory Structure

Your data **must** follow this structure:

```
data_directory/
  ├── 1700/              ← Numeric folder = conductor location in meters
  │   ├── 0moffset1.tem
  │   ├── 0moffset2.tem
  │   ├── ...
  │   ├── 500moffset1.tem
  │   └── 500m_trailing1.tem
  ├── 1900/
  │   └── ...
  ├── 2100/
  │   └── ...
  └── ...
```

**Important:**
- Folder names must be **numeric** (1700, 1900, 2100, etc.)
- Folders represent true conductor locations in meters
- Each folder contains `.tem` files
- File naming: `{offset}m{offset/trailing}{file_id}.tem`

### Verification Steps

1. **Check your current directory:**
   ```python
   import os
   print("Current directory:", os.getcwd())
   print("Files here:", os.listdir('.'))
   ```

2. **List numeric folders:**
   ```python
   import os
   folders = [d for d in os.listdir('.') if os.path.isdir(d)]
   numeric_folders = [f for f in folders if f.replace('.','').isdigit()]
   print("Numeric folders found:", numeric_folders)
   ```

3. **Check a folder's contents:**
   ```python
   import os
   folder = '1700'  # Replace with your folder name
   if os.path.exists(folder):
       tem_files = [f for f in os.listdir(folder) if f.endswith('.tem')]
       print(f"TEM files in {folder}:", tem_files)
   ```

### Common Issues

#### Issue: Script sees folders but says "No data found"

**Cause:** Folders are not numeric

**Fix:** Ensure folder names are pure numbers:
- ✓ Good: `1700`, `1900`, `2100`
- ✗ Bad: `location_1700`, `site1`, `test_data`

#### Issue: "Error parsing file"

**Cause:** TEM file format issue

**Fix:** Check that TEM files:
- Have proper header line containing "EAST" and "STATION"
- Have data rows with numeric values
- Are properly formatted (not corrupted)

#### Issue: Windows path with backslashes

**Cause:** Python string escape issues

**Fix:** Use raw string (r prefix):
```python
# Bad (backslashes cause issues):
DATA_DIRECTORY = "C:\Users\data"

# Good (raw string):
DATA_DIRECTORY = r"C:\Users\data"

# Or (forward slashes work on Windows too):
DATA_DIRECTORY = "C:/Users/data"
```

### Still Having Issues?

Run this diagnostic script in the same directory as your data:

```python
import os

print("="*60)
print("DATA DIRECTORY DIAGNOSTIC")
print("="*60)

# Current location
cwd = os.getcwd()
print(f"\n1. Current directory: {cwd}")

# All items in current directory
all_items = os.listdir('.')
print(f"\n2. All items here ({len(all_items)} total):")
for item in sorted(all_items)[:10]:  # Show first 10
    is_dir = os.path.isdir(item)
    print(f"   {'[DIR]' if is_dir else '[FILE]'} {item}")
if len(all_items) > 10:
    print(f"   ... and {len(all_items)-10} more")

# Folders only
folders = [d for d in all_items if os.path.isdir(d)]
print(f"\n3. Folders found: {len(folders)}")
print(f"   {folders[:5]}")  # Show first 5

# Numeric folders (expected for data)
numeric_folders = [f for f in folders if f.replace('.','').isdigit()]
print(f"\n4. Numeric folders (data locations): {len(numeric_folders)}")
print(f"   {sorted(numeric_folders)}")

# Check contents of first numeric folder
if numeric_folders:
    first_folder = sorted(numeric_folders)[0]
    tem_files = [f for f in os.listdir(first_folder) if f.endswith('.tem')]
    print(f"\n5. TEM files in '{first_folder}': {len(tem_files)}")
    print(f"   {tem_files[:5]}")  # Show first 5

    if tem_files:
        print("\n✓ Data structure looks correct!")
        print(f"  Run script from: {cwd}")
    else:
        print("\n✗ No .tem files found in numeric folders")
else:
    print("\n✗ No numeric folders found")
    print("  Expected folders like: 1700, 1900, 2100, etc.")

print("="*60)
```

Save this as `check_data.py` and run it to diagnose the issue.
