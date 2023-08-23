import os
import random

import pandas as pd

K_shot_lst = [1, 5, 10]
dataset_type_lst = ['endo', 'colon', 'chest']
exp_num = 7

annotations = {
    'endo': 'data/MedFMC_train/endo/endo_train.csv',
    'colon': 'data/MedFMC_train/colon/colon_train.csv',
    'chest': 'data/MedFMC_train/chest/chest_train.csv'
}

destinations = {
    'endo': 'data_anns/MedFMC/endo_new',
    'colon': 'data_anns/MedFMC/colon_new',
    'chest': 'data_anns/MedFMC/chest_new'
}


def gen_support_set_chest(df: pd.DataFrame, k_shot: int, strategy: str = "random"):
    """
    generate support set for chest dataset with k_shot images per class while making sure there are no duplicates
    df: dataframe of chest dataset
    k_shot: number of images per class
    strategy: 'random', 'max_label' or 'min_label':
    if 'random, randomly sample k_shot images per class;
    if 'max_label', sample k_shot images which have the most labels;
    if 'min_label', sample k_shot images where the image optimally has only one label, otherwise as little as possible
    return: support set of chest dataset
    """
    support_set = []
    labels = df.columns[2:]
    for label in labels:
        sample = []
        # filter out images where current label is 1
        df_cur = df[df[label] == 1]
        # get all img_ids of the filtered dataframe
        img_ids = df_cur['img_id'].unique().tolist()

        if strategy == 'random':
            # randomly sample k_shot images from the filtered dataframe
            sample = df_cur.sample(n=k_shot)
        elif strategy == 'min_label':
            # TODO
        elif strategy == 'max_label':
            # TODO
        else:
            raise ValueError(f'Invalid strategy {strategy}.')

        # add the sampled images to the support set
        support_set += sample.tolist()

        # remove the sampled images from the dataframe to avoid duplicates
        df = df[~df['filename'].isin(sample['filename'])]

    return support_set

def gen_support_set_endo(df: pd.DataFrame, k_shot: int, strategy: str ='random'):
    """
    generate support set for endo dataset with k_shot images per class while making sure there are no duplicates
    df: dataframe of endo dataset
    k_shot: number of images per class
    strategy: 'random' or 'max':
    if 'random, randomly sample k_shot images per class;
    if 'max', sample k_shot patients which have the most images;
    return: support set of endo dataset
    """
    support_set = []
    labels = df.columns[2:]
    for label in labels:
        # filter out images where current label is 1
        df_cur = df[df[label] == 1]
        # get all study_ids of the filtered dataframe
        study_ids = df_cur['study_id'].unique().tolist()

        if strategy == 'random':
            # randomly sample k_shot study_ids from the filtered dataframe
            sample = random.sample(study_ids, k_shot)

        elif strategy == 'max':
            # order the study_ids by the number of images they have
            study_ids_ordered = df_cur.groupby('study_id').count().sort_values(by='img_id', ascending=False).index.tolist()
            # take the first k_shot study_ids from the ordered list
            sample = study_ids_ordered[:k_shot]
        else:
            raise ValueError(f'Invalid strategy {strategy}.')

        # filter out images where study_id is in the sampled study_ids
        df_cur = df_cur[df_cur['study_id'].isin(sample)]
        # add the img_ids and labels of the sampled study_ids to the support set
        support_set.append(df_cur[['img_id', label]].values.tolist())
        # remove the sampled study_ids from the dataframe to avoid duplicates
        df = df[~df['study_id'].isin(sample)]

    return support_set


def gen_support_set_colon(df: pd.DataFrame, k_shot: int, strategy: str ='random'):
    """
    generate support set for colon dataset with k_shot images per class while making sure there are no duplicates
    df: dataframe of colon dataset
    k_shot: number of images per class
    strategy: 'random' or 'max':
    if 'random, randomly sample k_shot images per class;
    if 'max', sample k_shot patients which have the most images;
    return: support set of colon dataset
    """
    support_set = []
    labels = [0, 1]

    for label in labels:
        # filter out images with current label
        df_cur = df[df[label] == 1]
        # get all study_ids of the filtered dataframe
        study_ids = df_cur['study_id'].unique().tolist()

        if strategy == 'random':
            # randomly sample k_shot study_ids from the filtered dataframe
            sample = random.sample(study_ids, k_shot)

        elif strategy == 'max':
            # order the study_ids by the number of images they have
            study_ids_ordered = df_cur.groupby('study_id').count().sort_values(by='img_id',
                                                                               ascending=False).index.tolist()
            # take the first k_shot study_ids from the ordered list
            sample = study_ids_ordered[:k_shot]
        else:
            raise ValueError(f'Invalid strategy {strategy}.')

        # filter out images where study_id is in the sampled study_ids
        df_cur = df_cur[df_cur['study_id'].isin(sample)]
        # add the img_ids and labels of the sampled study_ids to the support set
        support_set.append(df_cur[['img_id', label]].values.tolist())
        # remove the sampled study_ids from the dataframe to avoid duplicates
        df = df[~df['study_id'].isin(sample)]

    return support_set


    return support_set

# randomly sample k_shot images 5 times for each dataset
for exp_num in range(1, 6):
    for dataset_type in dataset_type_lst:
        for K_shot in K_shot_lst:
            # read csv file as dataframe
            df = pd.read_csv(annotations[dataset_type])

            # generate support set
            if dataset_type == 'endo':
                support_set = gen_support_set_endo(df, K_shot, strategy='random')
            elif dataset_type == 'colon':
                support_set = gen_support_set_colon(df, K_shot, strategy='random')
            elif dataset_type == 'chest':
                support_set = gen_support_set_chest(df, K_shot, strategy='random')
            else:
                raise ValueError(f'Invalid dataset type {dataset_type}.')

            # validation set is simply the rest of the dataset
            val_set = df[~df['filename'].isin(support_set)]

            # write support set to txt file, where labels for chest are separated by commas, but for the others by space
            if dataset_type == 'chest':
                with open(os.path.join(destinations[dataset_type], f'{dataset_type}_{K_shot}-shot_train_exp{exp_num}.txt'), 'w') as f:
                    for i, i_class in enumerate(support_set):
                        f.write(','.join(i_class) + ' ' + str(i) + '\n')
                with open(os.path.join(destinations[dataset_type], f'{dataset_type}_{K_shot}-shot_val_exp{exp_num}.txt'), 'w') as f:
                    for i, i_class in enumerate(val_set):
                        f.write(','.join(i_class) + ' ' + str(i) + '\n')
            else:
                with open(os.path.join(destinations[dataset_type], f'{dataset_type}_{K_shot}-shot_train_exp{exp_num}.txt'), 'w') as f:
                    for i, i_class in enumerate(support_set):
                        for j_id in support_set[i]:
                            f.write(j_id + ' ' + str(i) + '\n')
                with open(os.path.join(destinations[dataset_type], f'{dataset_type}_{K_shot}-shot_val_exp{exp_num}.txt'), 'w') as f:
                    for i, i_class in enumerate(val_set):
                        for j_id in val_set[i]:
                            f.write(j_id + ' ' + str(i) + '\n')



