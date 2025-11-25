# chest_xray_etl.py
# Purpose: Simple ETL for Chest X-Ray dataset
# Extract: list image files and labels
# Transform: grayscale, resize, normalize, flatten
# Load: write curated dataset to Parquet

import os
import cv2
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField

# dataset root path
DATASET_ROOT = "/Users/sayali/Documents/cloud_technologies/chest_xray"

# output path for curated parquet
OUTPUT_PATH = "file:///Users/sayali/Documents/cloud_technologies/image_pipeline/output/chest_xray_curated"

# use smaller image size to keep memory low
IMG_SIZE = (128, 128)

# start spark with reasonable local settings
spark = SparkSession.builder \
    .appName("ChestXRay-ETL") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# function to walk dataset folders and collect image paths with split and label
def list_images(root_dir):
    rows = []
    for split in ["train", "test", "val"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            folder = os.path.join(root_dir, split, label)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                name_lower = fname.lower()
                if name_lower.endswith(".jpg") or name_lower.endswith(".jpeg") or name_lower.endswith(".png"):
                    full_path = os.path.join(folder, fname)
                    rows.append((full_path, split, label))
    return rows

# define schema for the initial dataframe
schema = StructType([
    StructField("path", StringType(), False),
    StructField("split", StringType(), False),
    StructField("label", StringType(), False)
])

# build the raw index of images
rows = list_images(DATASET_ROOT)
df = spark.createDataFrame(rows, schema)
print(f"Extracted {df.count()} image paths")

# image preprocessing function
def preprocess(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype("float32") / 255.0
        return img.flatten().tolist()
    except Exception:
        return None

# register udf and apply preprocessing
preprocess_udf = F.udf(preprocess, ArrayType(FloatType()))
df = df.withColumn("features", preprocess_udf("path")).dropna(subset=["features"])

# write curated data to parquet
df.write.mode("overwrite").parquet(OUTPUT_PATH)
print("ETL pipeline complete. Curated data saved to parquet.")
