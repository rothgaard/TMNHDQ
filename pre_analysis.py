import pandas as pd
from collections import Counter

from datetime import datetime


def epic_estimated_vs_actual(team_size, avg_wip, avg_days_story):
    conv_factor_eff_days = float(220/365)*0.8

    data_epic_jira = pd.read_csv("./data/jira_epic_data.csv", index_col=0)
    data_epic_processed = pd.read_csv("./data/processed_data.csv", index_col=0)

    data_cols = ['Epic-Key', 'Estimated', 'Actual', 'Lead Time', 'Points', 'Error (%)', 'Error (days)']
    data_out = pd.DataFrame()

    for index, row in data_epic_processed.iterrows():
        item = row
        jira_data = data_epic_jira.loc[data_epic_jira['doc_key'] == item['doc_key']]
        if jira_data.empty:
            continue
        elif int(jira_data['story_points'].values[0]) == 0:
            continue

        actual_by_sp = int(jira_data['story_points'].values[0])*(team_size/avg_wip)*avg_days_story*conv_factor_eff_days
        estimate_by_md = item['estimate']
        print("\nEpic: %s " % row['doc_key'])
        print("Original Estimate: %s" % estimate_by_md)
        print("Actual: %0.0f" % actual_by_sp)
        error_md = actual_by_sp-estimate_by_md
        print("Error (Man/Days): %0.0f" %  error_md)
        
        error_pct = 100*(actual_by_sp/estimate_by_md - 1.0)
        
        print("Error (%%): %0.0f" %  error_pct)

        data_out = data_out.append({'Epic-Key': row['doc_key'], \
            'Points': int(jira_data['story_points'].values[0]) , \
            'Estimated': estimate_by_md, \
            'Actual': actual_by_sp, \
            'Lead Time': jira_data['lead_time'].values[0], \
            'Error (%)': error_pct, \
            'Error (days)': error_md }, ignore_index=True)
    
    data_out = data_out.reindex(columns=data_cols)
    histogram_from_dataframe(data_out, data_cols[2:], 'epic_hist.png')
    correlation_heatmap_from_dataframe(data_out.drop(columns=['Epic-Key']), 'epic_corr_heat.png')

def histogram_from_dataframe(df, features, out_file_name):
    import matplotlib.pyplot as plt
    plt.figure()
    df.hist(column=features)
    plt.savefig(out_file_name)

def correlation_heatmap_from_dataframe(df, out_file_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    matrix = np.triu(df.corr())
    plt.figure(figsize=(16, 9))
    corr_mtx = df.corr()
    sns.heatmap(corr_mtx,annot = True, mask=matrix, cmap="RdYlBu",
            xticklabels=corr_mtx.columns,
            yticklabels=corr_mtx.columns)
    
    plt.yticks(rotation=0)
    plt.savefig(out_file_name)

def main():
    data_stories = pd.read_csv("./data/jira_story_data.csv", index_col=0)
    print("Stats for Cycle Time") 
    print(data_stories['Cycle Time'].describe())

    start = '2016-05-10'
    end = '2019-12-05'

    fmt = '%Y-%m-%d'

    delta_days = datetime.strptime(end, fmt) - datetime.strptime(start, fmt)
    delta_days = abs(float(delta_days.days))
    total_stories = float(data_stories.shape[0])

    team_size = 14
    atp = total_stories/delta_days
    act = data_stories["Cycle Time"].mean()
    total_sp = data_stories["Points"].sum()

    awip = act*atp
    tm_story = team_size/awip
    #tm_sp = team_size/asp
    sp_day = total_sp/delta_days
    avg_days_sp = data_stories["Cycle Time"].mean()/data_stories["Points"].mean()
    print("Total timespan in days: %d" % delta_days)
    print("Average Team Size: %d" % team_size)
    print("Average Throughput (Stories/Day): %0.2f" % atp)
    print("Average Throughput (Story Points/Day): %0.2f" % sp_day)
    print("Average Cycle Time (Days): %0.2f" % act)
    print("Average Work-In-Progress (Stories): %0.2f" % awip)
    print("Average Allocation of Team Members / Story : %0.2f" % tm_story)
    print("Average  Days / Story Point: %0.2f" % avg_days_sp)
    #print("Average Allocation of Team Members / Story Point : %0.2f" % tm_sp)

    epic_estimated_vs_actual(team_size, awip, avg_days_sp)



if __name__ == "__main__":
    main()
