from tqdm import tqdm
import pandas as pd
import os
import random
from RISparser import readris
from torch.utils.data import Dataset
from torch.utils.data import Sampler


def parse_data(incl, excl, max_entries=1e10):
    """
    Parses included and excluded data into a structured DataFrame.

    Parameters:
    incl (list): List of dictionaries containing included data.
    excl (list): List of dictionaries containing excluded data.
    max_entries (int, optional): Maximum number of entries to process from each list.

    Returns:
    DataFrame: A pandas DataFrame with titles, abstracts, and labels.
    """

    def process_entries(entries, label, limit):
        """
        Processes entries and returns a list of dictionaries with title, abstract, and label.

        Parameters:
        entries (list): List of dictionaries containing data.
        label (int): Label to assign to these entries (1 for included, 0 for excluded).
        limit (int): Maximum number of entries to process.

        Returns:
        list: Processed entries as a list of dictionaries.
        """
        processed = []
        for count, entry in enumerate(tqdm(entries)):
            if count >= limit:
                break
            title = entry.get("primary_title", "none")
            abstract = entry.get("abstract", "none")
            processed.append({"title": title, "abstract": abstract, "label": label})
        return processed

    # Process included and excluded entries
    included_data = process_entries(incl, label=1, limit=max_entries)
    excluded_data = process_entries(excl, label=0, limit=max_entries)

    # Combine and convert to DataFrame
    combined_data = included_data + excluded_data
    data_df = pd.DataFrame(combined_data, columns=["title", "abstract", "label"])

    return data_df


def read_ris_file(data_dir, file_name):
    """
    Reads a RIS file from a specified directory and file name.

    Parameters:
    data_dir (str): The data directory where the RIS file is located.
    file_name (str): The name of the RIS file to be read.

    Returns:
    object: The content read from the RIS file.
    """
    try:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            return readris(file)
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")
        return None


def pu_label_process_trans(trdf, vadf, tsdf, sample_tr_num_p, random_state):
    """
    Process datasets for PU learning, including sampling and label assignments.

    Parameters:
    trdf (DataFrame): Training DataFrame.
    vadf (DataFrame): Validation DataFrame.
    tsdf (DataFrame): Test DataFrame.
    sample_tr_num_p (int): Number of samples to take from trdf and vadf.
    random_state (int): Random state for reproducibility.

    Returns:
    tuple: Tuple containing processed training, validation, and test DataFrames.
    """
    sampled_trdf = trdf[trdf.label == 1].sample(
        n=sample_tr_num_p, random_state=random_state
    )
    sampled_trdf["pulabel"] = 1
    sampled_trdf["tr"] = 1
    sampled_trdf["ca"] = 0
    sampled_trdf["ts"] = 0

    sampled_vadf = vadf[vadf.label == 1].sample(
        n=sample_tr_num_p, random_state=random_state
    )
    sampled_vadf["pulabel"] = 0
    sampled_vadf["tr"] = 1
    sampled_vadf["ca"] = 1
    sampled_vadf["ts"] = 0

    tsdf["pulabel"] = 0
    tsdf["tr"] = 1
    tsdf["ca"] = 0
    tsdf["ts"] = 1

    train_df = pd.concat([sampled_trdf, sampled_vadf, tsdf]).reset_index(drop=True)

    print("Sampled Training Data:")
    print(sampled_trdf["label"].value_counts())
    print(sampled_trdf["pulabel"].value_counts())
    print()
    print("Sampled Calibration Data:")
    print(sampled_vadf["label"].value_counts())
    print(sampled_vadf["pulabel"].value_counts())
    print()
    print("Test Data:")
    print(tsdf["label"].value_counts())
    print(tsdf["pulabel"].value_counts())
    print()
    return train_df, sampled_vadf, tsdf


class BiDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


class BertDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.label = labels

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}, self.label[
            idx
        ]

    def __len__(self):
        return len(self.encodings.input_ids)


class ProportionalSampler(Sampler):
    def __init__(self, dataset, batch_size, num_cycles):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_cycles = num_cycles

        self.all_positive_indices = [
            i for i, label in enumerate(self.dataset.label) if label == 1
        ]
        self.all_negative_indices = [
            i for i, label in enumerate(self.dataset.label) if label == 0
        ]

        self.total_instances = len(self.all_positive_indices) + len(
            self.all_negative_indices
        )
        self.smaller_class, self.larger_class = (
            (set(self.all_positive_indices), set(self.all_negative_indices))
            if len(self.all_positive_indices) < len(self.all_negative_indices)
            else (set(self.all_negative_indices), set(self.all_positive_indices))
        )

    def __iter__(self):
        cycle_counter = self.num_cycles
        total_batches = len(self.dataset) // self.batch_size
        used_smaller_class_indices = set()

        for _ in range(total_batches):
            if cycle_counter > 0:
                num_smaller_per_batch = max(
                    1,
                    round(
                        (len(self.smaller_class) / self.total_instances)
                        * self.batch_size
                    ),
                )

                if (
                        len(self.smaller_class) - len(used_smaller_class_indices)
                        < num_smaller_per_batch
                ):
                    used_smaller_class_indices = set()

                available_smaller_class_indices = list(
                    self.smaller_class - used_smaller_class_indices
                )
                smaller_class_indices = random.sample(
                    available_smaller_class_indices, num_smaller_per_batch
                )
                used_smaller_class_indices.update(smaller_class_indices)

                num_larger_per_batch = self.batch_size - num_smaller_per_batch
                larger_class_indices = random.sample(
                    list(self.larger_class), num_larger_per_batch
                )

                batch_indices = smaller_class_indices + larger_class_indices
                random.shuffle(batch_indices)

                if len(used_smaller_class_indices) == len(self.smaller_class):
                    cycle_counter -= 1
                    used_smaller_class_indices = set()

            else:
                batch_indices = random.sample(
                    list(self.smaller_class) + list(self.larger_class), self.batch_size
                )

            for index in batch_indices:
                yield index

    def __len__(self):
        return len(self.dataset) // self.batch_size
