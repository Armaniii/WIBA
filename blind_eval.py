import pandas as pd
import argparse
""""
This script conducts a blind evaluation, hand labeling for the following datasets:
- UKP
- Arg spoken
- Arg qual for stance

For UKP, 500 rows are selected with an even distribution of target being 0 and target being 1, and an even distribution of selections from each unique topic. The user is then asked to label each text as an argument or not an argument.\
The results are saved in a new csv file.

For Arg spoken, 500 rows are selected with an even distribution of label 0 and 1. The user is then asked to label each text as an argument or not an argument.\
The results are saved in a new csv file.

For Arg qual for stance, 500 rows are selected with an even distribution of rank and an even distribution of pro/con stances. The user is then asked to label each text as a pro argument, con argument, or not an argument.\
The results are saved in a new csv file.

"""

ukp = '/data/ukp.csv'
debate = '/data/debate.csv'
ibm_arg = "/data/ibm_arg.csv"

do_ukp = False
do_debate = False
do_ibm_arg = False

parser = argparse.ArgumentParser()
parser.add_argument("--do_ukp", type=bool, default=False)
parser.add_argument("--do_debate", type=bool, default=False)
parser.add_argument("--do_ibm_arg", type=bool, default=False)
args = parser.parse_args()


if args.do_ukp:
    ukp = pd.read_csv(ukp,names=['text','target'])
    # select 500 total rows, with a 50% split between target being 0 and target being 1, and also an even distribution of selections from each unique topic
    # print out how many unique topics there are
    #in text everything after 'Target: and before 'Text' is the topic
    ukp["topic"] = ukp["text"].str.split("Target: ").str[1].str.split(" Text: ").str[0]
    ukp["text"] = ukp["text"].str.split("Text: ").str[1]

    from sklearn.model_selection import train_test_split

    # Assuming your dataset is in a pandas DataFrame called 'data'
    # with columns 'topic' and 'label'

    # Group the data by topic and label
    grouped = ukp.groupby(['topic', 'target'])

    # Create an empty list to store the selected rows
    selected_rows = []

    # Iterate over each group (topic and label combination)
    for group_name, group in grouped:
        topic, label = group_name
        
        # Calculate the desired number of rows for this group
        group_size = len(group)
        desired_group_size = 350 // (8 * 2)  # 500 rows divided by 8 topics and 2 labels
        
        # If the group size is larger than the desired size, select a random subset
        if group_size > desired_group_size:
            selected_group = group.sample(n=desired_group_size, random_state=42)
        else:
            selected_group = group
        
        # Append the selected rows to the list
        selected_rows.append(selected_group)

    # Concatenate the selected rows into a single DataFrame
    selected_data = pd.concat(selected_rows, ignore_index=True)

    # Optionally, you can shuffle the selected rows
    selected_data = selected_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # print the distribution of target for each topic
    
    human_eval = []
    i = 0
    for index, row in selected_data.iterrows():
        print(i, " of " , len(selected_data))
        print(row['text'])
        eval = input("Does the text qualify as an argument? (y/n): ")
        if eval == 'y':
            human_eval.append(1)
        if eval == 'n':
            human_eval.append(0)
        else:
            while eval != 'y' and eval != 'n':
                print("Invalid input. Please enter 'y' or 'n'")
                eval = input("Does the text qualify as an argument? (y/n): ")
                if eval == 'y':
                    human_eval.append(1)
                if eval == 'n':
                    human_eval.append(0)
        print("-------------------")
        i += 1
    selected_data['human_eval'] = human_eval
    
    selected_data.to_csv('/data/ukp_human_labeled.csv', index=False)


if args.do_debate:
    # randomly select 500 rows with equal distribution of label 0 and 1
    debate = pd.read_csv(debate)
    # none of arg spoken is in arg val

    debate = debate.sample(frac=1).reset_index(drop=True)
    debate_0 = debate[debate['label'] == 0].head(250)
    debate_1 = debate[debate['label'] == 1].head(250)
    debate_new = pd.concat([debate_0, debate_1])

    human_eval = []
    i = 0
    for index, row in debate_new.iterrows():
        print(i, " of " , len(debate_new))
        print(row['sentence'])
        eval = input("Does the text qualify as an argument? (y/n): ")
        if eval == 'y':
            human_eval.append(1)
        if eval == 'n':
            human_eval.append(0)
        else:
            while eval != 'y' and eval != 'n':
                print("Invalid input. Please enter 'y' or 'n'")
                eval = input("Does the text qualify as an argument? (y/n): ")
                if eval == 'y':
                    human_eval.append(1)
                if eval == 'n':
                    human_eval.append(0)
        print("-------------------")
        i+=1
    debate_new['human_eval'] = human_eval
    debate_new.to_csv('/data/debate_human_labeled.csv', index=False)


if args.ibm_arg:
    def bin_rank(rank, num_bins=10):
        return pd.qcut(rank, q=num_bins, labels=False, duplicates='drop')
    ibm_arg = pd.read_csv(ibm_arg)
    ibm_arg['rank_bin'] = bin_rank(ibm_arg['rank'])
    balanced_subset = pd.DataFrame()
    for rank_bin in ibm_arg['rank_bin'].unique():
        for stance in ['pro', 'con']:
            subset = ibm_arg[(ibm_arg['rank_bin'] == rank_bin) & (ibm_arg['stance'] == stance)]
            balanced_subset = pd.concat([balanced_subset, subset.sample(n=min(len(subset), 25), random_state=42)], ignore_index=True)    
    # Give the column rank with continuous values from 0 to 1, select an even distribution of this continuous rank and and an even number of pro/con stances in the stance column
            
    pro_subset = balanced_subset[balanced_subset['stance'] == 'pro'].sample(n=250, random_state=42)
    con_subset = balanced_subset[balanced_subset['stance'] == 'con'].sample(n=250, random_state=42)
    ibm_arg_new = pd.concat([pro_subset, con_subset], ignore_index=True)

    print(ibm_arg_new.head(50))

    human_eval = []
    i = 0
    for index, row in ibm_arg_new.iterrows():
        print(i, " of " , len(ibm_arg_new))
        print(row['topic'])
        print(row['argument'])
        eval = input("If the text is no argument, enter 0, if its pro argument enter 1 if its a con argument enter 2:")
        eval = str(eval)

        if eval == '0':
            human_eval.append(0)
        elif eval == '1':
            human_eval.append(1)
        elif eval == '2':
            human_eval.append(2)
        else:
            human_eval.append(4)
        print("-------------------")
        i += 1
    ibm_arg_new['human_eval'] = human_eval
    ibm_arg_new.to_csv('ibm_arg_human_labeled.csv', index=False)
