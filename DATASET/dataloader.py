import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split

from encodec import EncodecModel
from encodec.utils import convert_audio

class MusicDataset(Dataset):
    def __init__(self, dataset_dir, metadata_dir, sr, channels, min_duration, max_duration,
                 sample_duration, aug_shift, device, durations_path, cumsum_path,
                 audio_file_txt_path):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.metadata_dir = metadata_dir  
        self.sr = sr
        self.channels = channels
        self.min_duration = min_duration 
        self.max_duration = max_duration
        self.sample_duration = sample_duration
        self.aug_shift = aug_shift
        self.device = device
        self.model = EncodecModel.encodec_model_48khz().to(device=self.device)
        self.audio_files_dir = f'{dataset_dir}/audios'
        self.durations = None
        self.cumsum = None
        if durations_path is not None:
            self.durations = torch.load(durations_path)
        if cumsum_path is not None:
            self.cumsum = torch.load(cumsum_path)
        if audio_file_txt_path is not None:
            self.audio_file_txt_path = audio_file_txt_path
            with open(self.audio_file_txt_path, 'r') as file:
                self.audio_files = [line.strip() for line in file]
        self.init_dataset()

    def filter(self, audio_files, durations):
        keep = []
        audio_files_filtered = []
        for i in range(len(audio_files)):
            filepath = audio_files[i]
            if durations[i] < self.min_duration:
                continue
            if durations[i] >= self.max_duration:
                continue
            keep.append(i)
            audio_files_filtered.append(filepath)
        durations_filtered = [durations[i] for i in keep]
        duration_tensor = torch.tensor(durations_filtered)
        self.cumsum = torch.cumsum(duration_tensor, dim=0)

    def init_dataset(self):
        audio_files = os.listdir(self.audio_files_dir)
        audio_files = [f'{self.audio_files_dir}/{file}' for file in audio_files if file.endswith('.wav') or file.endswith('.mp3')]
        if self.durations is None and self.cumsum is None:
            durations = [self.get_duration_sec(file) for file in audio_files]
            try:
                self.filter(audio_files=audio_files, durations=durations)
            except AttributeError as e:
                print("AttributeError:", e)
                print("Ensure that the 'durations' attribute is properly assigned.")
            except Exception as e:
                print("An error occurred during filtering:", e)
        else:
            print("durations:", self.durations)
            print("cumsum:", self.cumsum)

    def get_song_chunk(self, index, offset):
        audio_file_path = os.path.join(self.audio_files_dir, self.audio_files[index])  
        wav, sr = torchaudio.load(audio_file_path)
        start_sample = int(offset * sr)
        end_sample = start_sample + int(self.sample_duration * sr)
        chunk = wav[:, start_sample:end_sample]
        return chunk, sr

    def __len__(self):
        return len(self.durations)

    def __getitem__(self, item):
        index, offset = self.get_index_offset(item)
        chunk, sr = self.get_song_chunk(item, offset)
        song_name = os.path.splitext(os.path.basename(self.audio_files[index]))[0]
        if os.path.exists(f'{self.metadata_dir}/{song_name}.json'):
            with open(f'{self.metadata_dir}/{song_name}.json', 'r') as file:
                metadata = json.load(file)

        chunk = convert_audio(chunk, sr, self.model.sample_rate, self.model.channels)
        chunk = chunk.unsqueeze(0).to(device=self.device)
        with torch.no_grad():
            encoded_frames = self.model.encode(chunk)
        chunk = chunk.mean(0, keepdim=True)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = codes.transpose(0, 1)
        emb = self.model.quantizer.decode(codes)
        emb = emb.to(self.device)

        return chunk, metadata, emb

def collate(batch):
    device = batch[0][0].device
    audio, data, emb = zip(*batch)

    for i, tensor in enumerate(audio):
        print(f"Tensor {i + 1} size: {tensor.size()}")

    for i, d in enumerate(data):
        print(f"Metadata {i + 1} size: {len(d)}")

    audio = torch.cat(audio, dim=0)
    emb = torch.cat(emb, dim=0)
    metadata = [d for d in data]

    return (emb, metadata)

def get_dataloader(dataset_folder, metadata_dir, sr, channels, min_duration, max_duration,
                   sample_duration, aug_shift, device, durations_path, cumsum_path, audio_file_txt_path,
                   batch_size: int = 50, shuffle: bool = True):
    dataset = MusicDataset(dataset_folder, metadata_dir, sr, channels, min_duration, max_duration,
                           sample_duration, aug_shift, device, durations_path, cumsum_path, audio_file_txt_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
    return dataloader

def get_dataloaders(dataset_dir, metadata_dir, sr, channels, min_duration, max_duration, sample_duration, 
                    aug_shift, batch_size: int = 50, shuffle: bool = True, split_ratio=0.8, device='cpu',
                    durations_path=None, cumsum_path=None, audio_file_txt_path=None):
    if not isinstance(dataset_dir, tuple):
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dir, valid_dir = dataset_dir
        train_dataset = MusicDataset(train_dir, metadata_dir, sr, channels, min_duration, max_duration, sample_duration,
                                     aug_shift, device, durations_path, cumsum_path, audio_file_txt_path)
        val_dataset = MusicDataset(valid_dir, metadata_dir, sr, channels, min_duration, max_duration, sample_duration,
                                   aug_shift, device, durations_path, cumsum_path, audio_file_txt_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True)
                        

    return train_dataloader, val_dataloader
