import pandas as pd


def analyze_tripinfo(file_path, sheet_name):
    """
    Analyze SUMO tripinfo and stopinfo data stored in Excel.

    Returns overall statistics across all passengers and vehicles.
    """

    # Load the sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Overall passenger waiting time stats
    waiting_stats = {
        "avg_waitingTime": df["waitingTime"].mean(),
        "q1_waitingTime": df["waitingTime"].quantile(0.25),
        "q3_waitingTime": df["waitingTime"].quantile(0.75),
        "min_waitingTime": df["waitingTime"].min(),
        "max_waitingTime": df["waitingTime"].max(),
    }

    # Overall vehicle stats (mean of travel times and timeLoss)
    vehicle_stats = {
        "avg_timeLoss": df["timeLoss"].mean(),
        "q1_timeLoss": df["timeLoss"].quantile(0.25),
        "q3_timeLoss": df["timeLoss"].quantile(0.75),
        "min_timeLoss": df["timeLoss"].min(),
        "max_timeLoss": df["timeLoss"].max(),
        "avg_travelTime": df["traveltime"].mean(),
    }

    return waiting_stats, vehicle_stats


if __name__ == "__main__":
    # Example usage
    excel_file = "D:\\6Sumo\\Analysis.xlsx"  # <-- change this to your file path
    sheet = "GreedyTripInfo"  # <-- or 'GreedyClosestTripInfo'

    waiting_stats, vehicle_stats = analyze_tripinfo(excel_file, sheet)

    print("=== Overall Passenger Waiting Stats ===")
    for k, v in waiting_stats.items():
        print(f"{k}: {v:.2f}")

    print("\n=== Overall Vehicle Stats ===")
    for k, v in vehicle_stats.items():
        print(f"{k}: {v:.2f}")

