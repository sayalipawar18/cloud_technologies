# Chest X-Ray ETL Project

We built a simple, cloud-style ETL pipeline for the Kaggle Chest X-Ray Pneumonia dataset. It runs locally on a MacBook using PySpark, but follows the same patterns used in cloud data engineering.

## Folder layout

- chest_xray: dataset with train, test, val splits
- image_pipeline: code and outputs

## Files

- chest_xray_etl.py
  - Extract: list image files and labels from folder structure
  - Transform: grayscale, resize to 128x128, normalize to 0-1, flatten to 1D array
  - Load: write curated dataset to Parquet

- chest_xray_augment_and_stats.py
  - Augment train split with horizontal flips
  - Compute pixel mean and std for each image
  - Save augmented dataset and aggregates to Parquet

- chest_xray_report.py
  - Print sample rows and aggregates
  - Summarize class distribution and augmentation counts

## How to run

1. Run ETL
   - python3 chest_xray_etl.py

2. Run augmentation and stats
   - python3 chest_xray_augment_and_stats.py

3. Run report
   - python3 chest_xray_report.py

## Task division

- Sayali
  - Implemented ETL pipeline
  - Spark session setup and memory tuning
  - Preprocessing logic and curated output

- Teammate
  - Implemented augmentation and stats
  - Built aggregates for reporting
  - Wrote reporting script and validated outputs

## Why this is cloud-style

- Clear Extract, Transform, Load stages
- Schema-driven Spark DataFrames
- Parquet outputs for fast analytics
- Memory tuning and simple, reproducible steps
