# chest_xray_report.py
# Purpose: Print simple summary from augmented dataset and aggregates

from pyspark.sql import SparkSession

AUGMENTED_INPUT = "file:///Users/sayali/Documents/cloud_technologies/image_pipeline/output/chest_xray_curated_augmented"
AGGREGATES_INPUT = "file:///Users/sayali/Documents/cloud_technologies/image_pipeline/output/aggregates"

spark = SparkSession.builder \
    .appName("ChestXRay-Report") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

combined = spark.read.parquet(AUGMENTED_INPUT)
aggregates = spark.read.parquet(AGGREGATES_INPUT)

print("Sample rows from combined curated dataset")
combined.select("path", "split", "label", "augmented", "pixel_mean", "pixel_std").show(10, truncate=False)

print("Aggregates by split, label, augmented")
aggregates.orderBy("split", "label", "augmented").show(50, truncate=False)

total_images = combined.count()
aug_images = combined.filter("augmented = true").count()
normal_count = combined.filter("label = 'NORMAL'").count()
pneumonia_count = combined.filter("label = 'PNEUMONIA'").count()

print(f"Total rows in curated dataset: {total_images}")
print(f"Rows created by augmentation: {aug_images}")
print(f"Class distribution: NORMAL={normal_count}, PNEUMONIA={pneumonia_count}")
