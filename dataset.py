import tqdm
import torch
from torch.utils.data import Dataset


def remap_chosen_rejected_to_ranking(dataset):
    result = []
    for sample in tqdm(dataset):
        result.append({
            "prompt": sample["prompt"],
            "ranked_outputs": [
                sample["chosen"],
                sample["rejected"]
            ]
        })
    return result


def remove_duplicates(xs):
    # keys are inserted in order, so only the first occurrence of each element in each list is preserved
    return list(dict.fromkeys(xs))


def is_too_short(output):
    return len(output.split()) < 5


def create_comparison_dataset(dataset):
    result = []
    for sample in tqdm(dataset):
        ranked_outputs = remove_duplicates([
            f"{sample['prompt']}\n{output}"
            for output in sample["ranked_outputs"]
            if not is_too_short(output)
        ])
        if len(ranked_outputs) >= 2:
            result.append(ranked_outputs)
    return result


class RankingDataset(Dataset):
    def __init__(self, rankings, tokenizer, max_length):
        self.items = []
        for ranking in tqdm(rankings):
            current_items = []
            for output in ranking:
                encodings_dict = tokenizer(
                    "<|startoftext|>" + output + "<|endoftext|>",
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                current_items.append(encodings_dict["input_ids"])
                current_items.append(encodings_dict["attention_mask"])
            self.items.append(current_items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DataCollator:
    def __init__(self, max_ranks_per_batch=2, max_sequence_length=512, padding_token=0):
        self.max_ranks_per_batch = max_ranks_per_batch
        self.max_sequence_length = max_sequence_length
        self.padding_token = padding_token

    def __call__(self, data):
        batch = {}
        input_ids = []
        attention_mask = []
        for i in range(self.max_ranks_per_batch):
            input_ids.extend([
                f[i * 2] if i * 2 < len(f) else torch.tensor([[self.padding_token] * self.max_sequence_length])
                for f in data
            ])
            attention_mask.extend([
                f[i * 2 + 1] if i * 2 < len(f) else torch.tensor([[2] * self.max_sequence_length])
                for f in data
            ])
        batch["input_ids"] = torch.cat(input_ids)
        batch["attention_mask"] = torch.cat(attention_mask)
        return batch
