# chest_xray_augment_and_stats.py
# Purpose: Augment train images and compute quality stats
# Input: curated parquet from ETL
# Output: augmented parquet and aggregates parquet

import cv2
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, DoubleType

CURATED_INPUT = "file:///Users/sayali/Documents/cloud_technologies/image_pipeline/output/chest_xray_curated"
AUGMENTED_OUTPUT = "file:///Users/sayali/Documents/cloud_technologies/image_pipeline/output/chest_xray_curated_augmented"
AGGREGATES_OUTPUT = "file:///Users/sayali/Documents/cloud_technologies/image_pipeline/output/aggregates"

IMG_SIZE = (128, 128)

spark = SparkSession.builder \
    .appName("ChestXRay-Augment-Stats") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# read curated dataset
df = spark.read.parquet(CURATED_INPUT)

# simple horizontal flip augmentation for train split only
def flip_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.flip(img, 1)
        img = img.astype("float32") / 255.0
        return img.flatten().tolist()
    except Exception:
        return None

flip_udf = F.udf(flip_image, ArrayType(FloatType()))

# create augmented rows for train only
aug_train = (
    df.filter(F.col("split") == "train")
      .withColumn("features_aug", flip_udf(F.col("path")))
      .dropna(subset=["features_aug"])
      .drop("features")
      .withColumnRenamed("features_aug", "features")
      .withColumn("augmented", F.lit(True))
)

# original rows marked as not augmented
original = df.withColumn("augmented", F.lit(False))

# union original and augmented
combined = original.unionByName(aug_train, allowMissingColumns=True)

# compute simple pixel stats for sanity checks
def pixel_stats(features):
    if features is None or len(features) == 0:
        return (None, None)
    arr = np.array(features, dtype=np.float32)
    return (float(arr.mean()), float(arr.std()))

stats_schema = StructType([
    StructField("mean", DoubleType(), True),
    StructField("std", DoubleType(), True),
])

@F.udf(returnType=stats_schema)
def stats_udf(features):
    m, s = pixel_stats(features)
    return {"mean": m, "std": s}

combined = combined.withColumn("pixel_stats", stats_udf(F.col("features")))
combined = combined.withColumn("pixel_mean", F.col("pixel_stats.mean"))
combined = combined.withColumn("pixel_std", F.col("pixel_stats.std")).drop("pixel_stats")

# write augmented dataset
combined.write.mode("overwrite").parquet(AUGMENTED_OUTPUT)

# basic aggregates for reporting
aggregates = (
    combined.groupBy("split", "label", "augmented")
            .agg(
                F.count("*").alias("image_count"),
                F.mean("pixel_mean").alias("avg_pixel_mean"),
                F.mean("pixel_std").alias("avg_pixel_std")
            )
)

aggregates.write.mode("overwrite").parquet(AGGREGATES_OUTPUT)

print("Augmentation and stats complete. Augmented parquet and aggregates saved.")
