import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def init_settings():
    tasks_ = ["colon", "chest", "endo"]
    shots_ = [str(i) + "-shot" for i in [1, 5, 10]]
    exps_ = ["exp" + str(i) for i in range(1, 5 + 1)]
    metrics_ = ["Acc_metric", "mAP_metric"]
    return tasks_, shots_, exps_, metrics_, [task + "_" + shot for task in tasks_ for shot in shots_]


def load_team_data(team1, team2):
    with open(f"{team1}.json", 'r') as src_file:
        src_team_data = json.load(src_file)
    with open(f"{team2}.json", 'r') as trg_file:
        trg_team_data = json.load(trg_file)
    return src_team_data, trg_team_data


def get_values_from_data(data, exp, setting, metric_key):
    """Fetch the metric values from the provided data."""
    metric_data = data['task'][exp][setting].get(metric_key)
    if metric_data:
        return float(metric_data)
    return None


def compare_results(team1, team2):
    comparison_result = {}
    team1_name, team1_data = team1.values()
    team2_name, team2_data = team2.values()
    metrics = ['Acc_metric', 'mAP_metric']

    for exp in exps:
        comparison_result[exp] = {}

        for setting in settings:
            comparison_result[exp][setting] = {}

            for metric in metrics:
                team1_val = team1_data['task'][exp][setting].get(metric)
                team2_val = team2_data['task'][exp][setting].get(metric)
                if team1_val is not None and team2_val is not None:
                    comparison_result[exp][setting][metric] = {team1_name: team1_val, team2_name: team2_val}
    return comparison_result


source_team_name = "uniwue"
target_team_name = "mcmong"

tasks, shots, exps, metrics, settings = init_settings()

task2metric = {'colon': metrics[0], 'chest': metrics[1], 'endo': metrics[1]}

source_data, target_data = load_team_data(team1=source_team_name, team2=target_team_name)
result = compare_results(team1={'name': source_team_name, 'data': source_data},
                         team2={'name': target_team_name, 'data': target_data})

# Create a subplot with multiple rows
fig = make_subplots(rows=len(settings), cols=1, subplot_titles=settings, vertical_spacing=0.05, shared_xaxes=True)

for idx, setting in enumerate(settings):
    metric_key = task2metric[next(task for task in task2metric if task in setting)]

    source_values = [get_values_from_data(source_data, exp, setting, metric_key) for exp in exps]
    target_values = [get_values_from_data(target_data, exp, setting, metric_key) for exp in exps]

    # Add traces to the subplot
    fig.add_trace(go.Bar(x=exps,
                         y=source_values,
                         name=source_team_name,
                         marker=dict(color='blue'),
                         legendgroup=idx + 1
                         ), row=idx + 1, col=1)

    fig.add_trace(go.Bar(x=exps,
                         y=target_values,
                         name=target_team_name,
                         marker=dict(color='red'),
                         legendgroup=idx + 1
                         ), row=idx + 1, col=1)

yaxis_titles = {
    f'yaxis{idx if idx > 1 else ""}_title': task2metric[task]
    for idx, setting in enumerate(settings, 1)
    for task in task2metric
    if task in setting
}

fig.update_layout(
    title_text="Comparison for two evaluation submissions",
    barmode='group',
    height=300 * len(settings),
    legend_tracegroupgap=257,
    **yaxis_titles
)

fig.show()
