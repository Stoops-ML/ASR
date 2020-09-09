import torch
import torch.nn as nn


char_map_str = (
            "'",  # 0
            "<SPACE>",  # 1
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",  # 27
            "_"  # 28, blank
        )  # TODO check that _ 28 doesn't mess up training. It's needed for ctcdecode
        # TODO make char_map_str better


class TextTransform:
    """Maps characters to integers and vice versa"""

    def __init__(self):
        self.char_map = {ch: int(i) for i, ch in enumerate(char_map_str)}
        self.index_map = {int(i): ch for i, ch in enumerate(char_map_str)}
        # for line in char_map_str.strip().split('\n'):
        #     ch, index = line.split()
        #     self.char_map[ch] = int(index)
        #     self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


def data_processing(data, audio_transforms, text_transform):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        spec = audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def greedy_decoder(output, labels, label_lengths, text_transform, blank_label=28, collapse_repeated=True):
    """not used in code"""
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j-1]:
                    continue
                decode.append(index.item())
            decodes.append(text_transform.int_to_text(decode))
    return decodes, targets