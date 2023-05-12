import numpy as np
import torch
import torch.utils.data

from model import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data, mode):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event, process_event

        mode: list with 'one' or 'short'
        """

        self.mode = mode
        if 'test' in mode:
            data = load_test_data()
        if 'one' in mode:
            print('Loading only one item...')
            data = data[:1]
        elif 'subset' in mode:
            print('Loading only 30 items...')
            data = data[:30]
        elif 'subset500' in mode:
            print('Loading only 500 items...')
            data = data[:500]
        elif 'subset2000' in mode:
            print('Loading only 2000 items...')
            data = data[:2000]
        elif 'subset5000' in mode:
            print('Loading only 5000 items...')
            data = data[:5000]

        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[round(elem['time_since_last_event'], 4) for elem in inst] for inst in data]
        self.event_type = [[elem['type_event'] for elem in inst] for inst in data]
        if 'process_event' in data[0][0]:
            self.event_process = [[elem['process_event'] for elem in inst] for inst in data]
        else:
            self.event_process = [[1 for _ in inst] for inst in data]
        if 'value' in mode:
            self.event_value = [[elem['value_event'] for elem in inst] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        if 'value' in self.mode:
            return self.time[idx], self.event_type[idx], self.event_process[idx], self.event_value[idx]
        else:
            return self.time[idx], self.event_type[idx], self.event_process[idx]


def pad_continuous(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_discrete(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, event_type, event_process = list(zip(*insts))
    time = pad_continuous(time)
    event_type = pad_discrete(event_type)
    event_process = pad_discrete(event_process)
    return time, event_type, event_process

def collate_fn_value(insts):
    """ Collate function with value, as required by PyTorch. """

    time, event_type, event_process, event_value = list(zip(*insts))
    time = pad_continuous(time)
    event_type = pad_discrete(event_type)
    event_process = pad_discrete(event_process)
    event_value = pad_continuous(event_value)
    return time, event_type, event_process, event_value


def get_dataloader(data, opt, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data, opt.data.mode)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=opt.train.batch_size,
        collate_fn=collate_fn if 'value' not in opt.data.mode else collate_fn_value,
        shuffle=shuffle
    )
    return dl


def load_test_data():
    """ Load toy data. """

    data = [[{'time_since_start': 2.0, 'time_since_last_event': 0.0, 'type_event': 1, 'process_event': 1},
             {'time_since_start': 3.0, 'time_since_last_event': 0.0, 'type_event': 3, 'process_event': 2},
             {'time_since_start': 4.0, 'time_since_last_event': 0.0, 'type_event': 2, 'process_event': 1},
             {'time_since_start': 6.0, 'time_since_last_event': 0.0, 'type_event': 1, 'process_event': 1},
             {'time_since_start': 8.0, 'time_since_last_event': 0.0, 'type_event': 2, 'process_event': 1},
             {'time_since_start': 9.0, 'time_since_last_event': 0.0, 'type_event': 3, 'process_event': 2},
             {'time_since_start': 10.0, 'time_since_last_event': 0.0, 'type_event': 1, 'process_event': 1},
             {'time_since_start': 12.0, 'time_since_last_event': 0.0, 'type_event': 2, 'process_event': 1},
             {'time_since_start': 14.0, 'time_since_last_event': 0.0, 'type_event': 1, 'process_event': 1},
             {'time_since_start': 15.0, 'time_since_last_event': 0.0, 'type_event': 3, 'process_event': 2},
             {'time_since_start': 16.0, 'time_since_last_event': 0.0, 'type_event': 2, 'process_event': 1},]]
    
    return data