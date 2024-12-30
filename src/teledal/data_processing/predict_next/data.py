import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PredictNextData(Dataset):
    def __init__(self, data_path: Path, labels_file: Path = None):
        """
        In order to avoid the time consuming preprocessing process
        it is highly recommended to create a temporary directory and
        dump the encoded blocks in `.pt` format and use this dataloader
        to load the embedded log sequences immediately.

        If anomaly labels are available then it can be passed as a `.csv`
        file and in that case the names of the `.pt` files should be the
        identifier of the corresponding log sequence. The csv file should
        have two column one of which contains the identifier and the other
        should be the binary label which should be 1 if the sequence is
        assiciated with anomalous behaviour.

        Parameters:
        -----------
        - data_path: Path to the directory of `.pt` files
        - labels_file: Path to `.csv` file if available
        """
        super().__init__()
        if isinstance(data_path, Path):
            data_path=Path(data_path)
            
        self._data_files: list[Path] = [file for file in data_path.glob("*.pt")]

        self._labels: dict[str, torch.Tensor] = dict()
        if labels_file is not None:
            with open(labels_file) as lf:
                reader = csv.reader(lf)
                for line in reader:
                    assert len(line) == 2, "The format of labels file is not supported"
                    self._labels[line[0]] = torch.Tensor([int(line[1])])

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_file: Path = self._data_files[idx]
        print(f"Current file: {current_file}")
        sequence_identifier = current_file.stem
        embedding = torch.load(current_file, weights_only=True)
        return (
            embedding[:-1, ...],
            embedding[1:, ...],
            (self._labels[sequence_identifier] if self._labels else -1),
        )

    def __len__(self):
        return len(self._data_files)