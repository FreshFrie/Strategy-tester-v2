import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load harvested events (guard if missing)
try:
	events = pd.read_csv("results/harvest_events.csv")
except FileNotFoundError:
	raise FileNotFoundError("results/harvest_events.csv not found — run main.py first to produce events.")

# Convert sweep_time to datetime and extract extras
events["sweep_time"] = pd.to_datetime(events["sweep_time"]) if "sweep_time" in events.columns else pd.NaT
events["dow"] = events["sweep_time"].dt.day_name()
events["hour"] = events["sweep_time"].dt.hour

# Binary outcome: 1 if reverted inside, 0 if continuation
if "reverted_inside_by_end" in events.columns:
	events["reverted"] = events["reverted_inside_by_end"].astype(int)
else:
	events["reverted"] = events.get("reverted", pd.Series(0, index=events.index)).astype(int)

# 1. By sweep side
by_side = events.groupby("side")["reverted"].mean()

# 2. By day of week
by_dow = events.groupby("dow")["reverted"].mean().sort_values()

# 3. By Asia range size (wide vs narrow split at median)
asia_range = (events["asia_high"] - events["asia_low"])
median_range = asia_range.median()
events["asia_range_cat"] = np.where(asia_range > median_range, "wide", "narrow")
by_range = events.groupby("asia_range_cat")["reverted"].mean()

# 4. By sweep timing (early vs late in London KZ)
events["london_phase"] = np.where(events["hour"] < 2, "early", "late")
by_phase = events.groupby("london_phase")["reverted"].mean()

# 5. Interaction: side + range
side_range = events.groupby(["side","asia_range_cat"])["reverted"].mean().unstack()

# Summaries
print("Overall revert rate:", events["reverted"].mean())
print("\nBy side:\n", by_side)
print("\nBy day of week:\n", by_dow)
print("\nBy Asia range size:\n", by_range)
print("\nBy London phase:\n", by_phase)
print("\nBy side + range:\n", side_range)

# ---- Asia-range vs breakout analysis and plots ----
try:
	if len(events) == 0:
		print("No events available for Asia-range analysis.")
	else:
		# Asia range and breakout flag (robust to missing reverted_inside_by_end)
		events["asia_range"] = events["asia_high"] - events["asia_low"]
		if "reverted_inside_by_end" in events.columns:
			events["breakout"] = (~events["reverted_inside_by_end"]).astype(int)
		else:
			# fall back to previously-computed 'reverted' column if present
			events["breakout"] = (1 - events.get("reverted", pd.Series(0, index=events.index))).astype(int)

		# Quantile groups (quartiles)
		# qcut can fail if there are too few unique values; wrap defensively
		try:
			events["range_group"] = pd.qcut(events["asia_range"], q=4, labels=["Q1 (tightest)", "Q2", "Q3", "Q4 (widest)"])
		except Exception:
			# fallback: equal-frequency bins failed (e.g., constant values), create a single-group label
			events["range_group"] = "all"

		group_stats = events.groupby("range_group").agg(
			sweeps=("breakout", "count"),
			breakout_rate=("breakout", "mean"),
			avg_asia_range=("asia_range", "mean")
		).reset_index()

		print("\nAsia-range vs breakout by quartile:\n", group_stats)

		# Create results directory if it doesn't exist
		import os
		os.makedirs("results", exist_ok=True)

		# Scatter: Asia range size vs breakout probability (save to file rather than show)
		figpath1 = "results/asia_range_vs_breakout_scatter.png"
		plt.figure(figsize=(7, 4))
		plt.scatter(events["asia_range"], events["breakout"], alpha=0.6)
		plt.xlabel("Asia Range Size")
		plt.ylabel("Breakout (1) / Fakeout (0)")
		plt.title("Asia Range vs London Breakout Probability")
		plt.tight_layout()
		plt.savefig(figpath1)
		plt.close()

		# Bar chart: Breakout rate by quantile (save to file)
		figpath2 = "results/asia_range_vs_breakout_bar.png"
		plt.figure(figsize=(7, 4))
		# If range_group is a column with a single value, group_stats.index will be scalar-like; handle that
		plt.bar(group_stats["range_group"].astype(str), group_stats["breakout_rate"].astype(float))
		plt.ylabel("Breakout Rate")
		plt.title("Breakout Probability by Asia Range Quartile")
		plt.tight_layout()
		plt.savefig(figpath2)
		plt.close()

		# Save stats CSV
		group_stats.to_csv("results/asia_range_vs_breakout.csv", index=False)
		print(f"Saved: {figpath1}, {figpath2}, results/asia_range_vs_breakout.csv")
except Exception as e:
	print("Asia-range analysis skipped due to error:", str(e))

# ------------------ Logistic regression classifier ------------------
# Build features similar to the provided snippet
if len(events) < 10:
	print("Not enough events to train a classifier (need >=10). Skipping ML block.")
else:
	# Features
	events["side_up"] = (events["side"] == "up_sweep").astype(int)
	events["london_late"] = np.where(events["hour"] >= 2, 1, 0)
	events["asia_range_cat_bin"] = np.where((events["asia_high"] - events["asia_low"]) > median_range, 1, 0)

	X = events[["side_up", "london_late", "asia_range_cat_bin", "max_extension"]].copy()

	# One-hot encode day of week and concat
	dow_dummies = pd.get_dummies(events["dow"], prefix="dow")
	X = pd.concat([X, dow_dummies], axis=1)

	y = events["reverted"].astype(int)

	# Train/test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Fit model
	model = LogisticRegression(max_iter=1000)
	model.fit(X_train, y_train)

	# Evaluate
	print("\nClassification Report:")
	print(classification_report(y_test, model.predict(X_test)))

	# Show coefficients
	coefs = pd.Series(model.coef_[0], index=X.columns)
	print("\nFeature Coefficients (log-odds):")
	print(coefs.sort_values(ascending=False))
