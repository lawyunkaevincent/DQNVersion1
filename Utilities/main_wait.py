from wait_stats import compute_wait_stats

path = r"D:\6Sumo\RLTesting\RLTrainingMap2\tripinfos.xml"
avg, p95, series, ids = compute_wait_stats(path)
print(f"Average wait: {avg:.2f}s")
print(f"98th percentile wait: {p95:.2f}s")
