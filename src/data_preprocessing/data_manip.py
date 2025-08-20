import pandas as pd

def fuse_datasets(datasets):
    fused = None
    reference_name = None

    for i, (name, data) in enumerate(datasets.items()):
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        df = df.add_prefix(name + "_")

        if f"{name}_timestamp" in df.columns:
            df = df.rename(columns={f"{name}_timestamp": "timestamp"})

        if fused is None:
            fused = df
            reference_name = name  # <-- store reference dataset
        else:
            fused = pd.merge_asof(
                fused.sort_values("timestamp"),
                df.sort_values("timestamp"),
                on="timestamp",
                direction="nearest"
            )

    #print(f"Fusion timeline based on dataset: {reference_name}")
    return fused
