import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

source_team = "uniwue"
target_team = "validated"

# Load the data
with open(f"{source_team}.json", 'r') as uniwue_file:
    uniwue_data = json.load(uniwue_file)

with open(f"{target_team}.json", 'r') as validated_file:
    validated_data = json.load(validated_file)

# Comparison function
def compare_values(uniwue_val, validated_val):
    if uniwue_val > validated_val:
        return uniwue_val, validated_val
    elif uniwue_val < validated_val:
        return uniwue_val, validated_val
    else:
        return uniwue_val, validated_val

# Comparison result
comparison_result = {}

for exp, values in uniwue_data['task'].items():
    comparison_result[exp] = {}
    for shot, metrics in values.items():
        comparison_result[exp][shot] = {}
        for metric, uniwue_val in metrics.items():
            validated_val = validated_data['task'][exp][shot].get(metric, None)
            if validated_val:
                uniwue_val, validated_val = compare_values(float(uniwue_val), float(validated_val))
                comparison_result[exp][shot][metric] = {"uniwue": uniwue_val, "validated": validated_val}

# Prepare data for the plots
all_settings = list(comparison_result["exp1"].keys())
all_experiments = list(comparison_result.keys())

# Create a subplot with multiple rows
fig = make_subplots(rows=len(all_settings), cols=1, subplot_titles=all_settings, vertical_spacing=0.05, shared_xaxes=True)

for idx, setting in enumerate(all_settings):
    uniwue_vals = [comparison_result[exp][setting]["mAP_metric"]["uniwue"] if "mAP_metric" in comparison_result[exp][setting] else comparison_result[exp][setting]["Acc_metric"]["uniwue"] for exp in all_experiments]
    validated_vals = [comparison_result[exp][setting]["mAP_metric"]["validated"] if "mAP_metric" in comparison_result[exp][setting] else comparison_result[exp][setting]["Acc_metric"]["validated"] for exp in all_experiments]

    # Add traces to the subplot
    fig.add_trace(go.Bar(x=all_experiments, y=uniwue_vals, name="uniwue", marker=dict(color='blue'), showlegend=(idx==0)), row=idx+1, col=1)
    fig.add_trace(go.Bar(x=all_experiments, y=validated_vals, name="validated", marker=dict(color='red'), showlegend=(idx==0)), row=idx+1, col=1)


fig.update_layout(
    title_text="Comparison for two settings",
    barmode='group',
    height=300*len(all_settings)
)

fig.show()

