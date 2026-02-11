from src.data import DataReader, add_history_features
import pandas as pd

def prepare_inference_data(
    config,
    trained_brand2id,
    test_start,
    test_end,
):
    data_cfg = config.data
    brand_col = data_cfg["brand_col"]
    time_col = data_cfg["time_col"]
    target_col = data_cfg["target_col"]
    brand_id_col = data_cfg["brand_id_col"]

    reader = DataReader(
        data_dir=data_cfg["data_dir"],
        file_pattern=data_cfg["file_pattern"],
    )

    ref_year = test_start.year - 1

    ref_data = reader.load(years=[ref_year])[
        [brand_col, time_col, target_col]
    ]

    test_data = reader.load_csv(
        config.inference["test_data_path"]
    )[
        [brand_col, time_col, target_col]
    ]

    # 🔑 CONCAT FIRST (RAW)
    data = pd.concat([ref_data, test_data], ignore_index=True)

    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col).reset_index(drop=True)

    # 🔑 ADD FEATURES ONCE (continuous time)
    data = add_history_features(
        data,
        time_col=time_col,
        brand_col=brand_col,
        target_col=target_col,
    )

    # Encode brands using training-time mapping
    data[brand_id_col] = data[brand_col].map(trained_brand2id)
    data = data.dropna(subset=[brand_id_col])
    data[brand_id_col] = data[brand_id_col].astype(int)

    return data

