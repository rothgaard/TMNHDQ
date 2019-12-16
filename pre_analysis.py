import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from datetime import datetime

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

    team_size = 10
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



if __name__ == "__main__":
    main()
