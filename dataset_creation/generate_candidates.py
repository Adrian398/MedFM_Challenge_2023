import argparse
import os
import random
import pandas as pd

data_prefix = '/scratch/medfm/medfm-challenge/'


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
    df_full = df.copy(deep=True)
    labels = df.columns[2:].tolist()

    for label in labels:
        # filter out images where current label is 1
        df_cur = df[df[label] == 1]
        # get all img_ids of the filtered dataframe
        img_ids = df_cur['img_id'].unique().tolist()

        # safety_check: if there are less img_ids than k_shot, sample the missing id_s from the full dataframe
        if len(img_ids) < k_shot:
            df_temp = df_full[df_full[label] == 1]
            img_ids = df_temp['img_id'].unique().tolist()

        if strategy == 'random':
            # randomly sample k_shot images from the filtered dataframe
            sample = random.sample(img_ids, k_shot)
        elif strategy == 'min_label':
            # order the img_ids by the number of labels they have
            img_ids_ordered = df_cur.groupby('img_id').sum().sort_values(by=label, ascending=True).index.tolist()
            # take the first k_shot img_ids from the ordered list
            sample = img_ids_ordered[:k_shot]
        elif strategy == 'max_label':
            # order the img_ids by the number of labels they have in descending order
            img_ids_ordered = df_cur.groupby('img_id').sum().sort_values(by=label, ascending=False).index.tolist()
            # take the first k_shot img_ids from the ordered list
            sample = img_ids_ordered[:k_shot]
        else:
            raise ValueError(f'Invalid strategy {strategy}.')

        # filter out images where img_id is in the sampled img_ids
        df_cur = df_cur[df_cur['img_id'].isin(sample)]
        # add the img_ids and labels of the sampled img_ids to the support set
        support_set += (df_cur[['img_id'] + labels].values.tolist())
        # remove the sampled images from the dataframe to avoid duplicates
        df = df[~df['img_id'].isin(sample)]

    return support_set


def gen_support_set_endo(df: pd.DataFrame, k_shot: int, strategy: str = 'random'):
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
    df_full = df.copy(deep=True)
    labels = df.columns[3:].tolist()

    for label in labels:
        # filter out images where current label is 1
        df_cur = df[df[label] == 1]

        # get all study_ids of the filtered dataframe
        study_ids = df_cur['study_id'].unique().tolist()

        # safety_check: if there are less study_ids than k_shot, sample the missing ids from the full dataframe
        if len(study_ids) < k_shot:
            df_temp = df_full[df_full[label] == 1]
            study_ids = df_temp['study_id'].unique().tolist()

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
        support_set += (df_cur[['img_id'] + labels].values.tolist())
        # remove the sampled study_ids from the dataframe to avoid duplicates
        df = df[~df['study_id'].isin(sample)]

    return support_set


def gen_support_set_colon(df: pd.DataFrame, k_shot: int, strategy: str = 'random'):
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
    df.insert(2, 'patient_id', df['img_id'].apply(lambda x: x[:-9]))
    df_full = df.copy(deep=True)
    labels = [0, 1]

    for label in labels:
        # filter out images where current label is present
        df_cur = df[df['tumor'] == label]

        # get all study_ids of the filtered dataframe
        patient_ids = df_cur['patient_id'].unique().tolist()

        # safety_check: if there are less patient_ids than k_shot, sample the missing ids from the full dataframe
        if len(patient_ids) < k_shot:
            df_temp = df_full[df_full['tumor'] == label]
            patient_ids = df_temp['patient_id'].unique().tolist()

        if strategy == 'random':
            # randomly sample k_shot study_ids from the filtered dataframe
            sample = random.sample(patient_ids, k_shot)

        elif strategy == 'max':
            # order the study_ids by the number of images they have
            study_ids_ordered = df_cur.groupby('patient_id').count().sort_values(by='img_id',
                                                                                 ascending=False).index.tolist()
            # take the first k_shot study_ids from the ordered list
            sample = study_ids_ordered[:k_shot]
        else:
            raise ValueError(f'Invalid strategy {strategy}.')

        # filter out images where study_id is in the sampled study_ids
        df_cur = df_cur[df_cur['patient_id'].isin(sample)]
        # add the img_ids and labels of the sampled study_ids to the support set
        support_set += (df_cur[['img_id', 'tumor']].values.tolist())
        # remove the sampled study_ids from the dataframe to avoid duplicates
        df = df[~df['patient_id'].isin(sample)]

    return support_set


def generate_val_set(full_df, support_set, dataset_type):
    # extracts all remaining images from full dataframe for val set and returns a list built like support set
    support_ids = [x[0] for x in support_set]
    val_df = full_df[~full_df['img_id'].isin(support_ids)]
    # return the remaining image_ids and labels as list
    if dataset_type == 'colon':
        labels = ['tumor']
    elif dataset_type == 'chest':
        labels = val_df.columns[2:].tolist()
    elif dataset_type == 'endo':
        labels = val_df.columns[3:].tolist()
    else:
        raise ValueError(f'Invalid dataset type {dataset_type}.')
    return val_df[['img_id'] + labels].values.tolist()


def get_annotations(dataset_type: str):
    annotations = {
        'endo': os.path.join(data_prefix, 'data', 'MedFMC_train', 'endo', 'endo_train.csv'),
        'colon': os.path.join(data_prefix, 'data', 'MedFMC_train', 'colon', 'colon_train.csv'),
        'chest': os.path.join(data_prefix, 'data', 'MedFMC_train', 'chest', 'chest_train.csv')
    }

    return pd.read_csv(annotations[dataset_type])


def write_dataset_to_txt_files(dataset, dataset_type, k_shot, exp_num, mode='train'):
    destinations = {
        'endo': os.path.join('dataset_creation', 'candidate_data', 'endo'),
        'colon': os.path.join('dataset_creation', 'candidate_data', 'colon'),
        'chest': os.path.join('dataset_creation', 'candidate_data', 'chest')
    }

    # Make sure that the destination directory exists
    for key, path in destinations.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

    # simply writes the dataset given as a list of lists to a txt file
    # labels are written with integer precision
    with open(os.path.join(destinations[dataset_type], f'{dataset_type}_{k_shot}-shot_{mode}_exp{exp_num}.txt'),
              'w') as f:
        for data in dataset:
            if dataset_type == 'chest':
                label_sep = ','
            else:
                label_sep = ' '
            f.write(data[0] + ' ' + label_sep.join(str(int(x)) for x in data[1:]) + '\n')


def generate_experiments_random(k_shot_list=None, dataset_type_list=None, num_experiments: int = 5):
    if k_shot_list is None:
        k_shot_list = [1, 5, 10]
    if dataset_type_list is None:
        dataset_type_list = ['colon', 'endo', 'chest']

    # randomly sample k_shot images 5 times for each dataset
    for exp_num in range(1, num_experiments + 1):
        for dataset_type in dataset_type_list:
            for k_shot in k_shot_list:
                # read csv file as dataframe
                df = get_annotations(dataset_type)

                # generate support set
                if dataset_type == 'endo':
                    support_set = gen_support_set_endo(df, k_shot, strategy='random')
                elif dataset_type == 'colon':
                    support_set = gen_support_set_colon(df, k_shot, strategy='random')
                elif dataset_type == 'chest':
                    support_set = gen_support_set_chest(df, k_shot, strategy='random')
                else:
                    raise ValueError(f'Invalid dataset type {dataset_type}.')

                # validation set is simply the rest of the dataset
                val_set = generate_val_set(df, support_set, dataset_type)

                # write support and val set to txt file
                write_dataset_to_txt_files(support_set, dataset_type, k_shot, exp_num, mode='train')
                write_dataset_to_txt_files(val_set, dataset_type, k_shot, exp_num, mode='val')


def delete_all_files_from_directory(directory_path):
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Delete each file
    for file in files:
        file_path = os.path.join(directory_path, file)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def clear_candidate_data():
    delete_all_files_from_directory(os.path.join('dataset_creation', 'candidate_data', 'endo'))
    delete_all_files_from_directory(os.path.join('dataset_creation', 'candidate_data', 'chest'))
    delete_all_files_from_directory(os.path.join('dataset_creation', 'candidate_data', 'colon'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to clear candidate data and generate experiments.")
    parser.add_argument('--local', action='store_true', help="Flag to indicate if the script is run locally.")
    parser.add_argument('--num_exp', type=int, default=10, help="The amount of experiments to generate.")
    args = parser.parse_args()

    if args.local:
        data_prefix = ''
    else:
        data_prefix = '/scratch/medfm/medfm-challenge/'

    if args.num_exp:
        num_exp = args.num_exp
    else:
        num_exp = 10

    clear_candidate_data()
    generate_experiments_random(num_experiments=num_exp)
