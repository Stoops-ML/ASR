import torch
import torch.nn.functional as F
from processing import TextTransform
import torchaudio


def predict(model, audio_file, decoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    waveform, sample_rate = torchaudio.load(audio_file)  # todo normalization=True??
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform).to(device)  # todo is this float or tensor? model accepts only tensors
    spectrogram = spectrogram.unsqueeze(0)  # add batch dimension of 1 todo make sure 0 is correct

    with torch.no_grad():
        output = model(spectrogram)  # (batch, time, n_class)
        output = F.softmax(output, dim=2)

    beam_results, _, _, out_lens = decoder.decode(output)  # BATCHSIZE x N_TIMESTEPS x N_LABELS
    tokens = beam_results[0][0]  # get the top beam for the first item in your batch
    seq_len = out_lens[0][0]  # beams almost always shorter than the num of time steps (additional data is non-sensical)
    text_transform = TextTransform()

    return text_transform.int_to_text(tokens[:seq_len].numpy())
