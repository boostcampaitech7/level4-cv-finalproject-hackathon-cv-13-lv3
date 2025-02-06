from dataset import SALMONNDataset

from torch.utils.data import DataLoader, DistributedSampler

dataset = SALMONNDataset("/data/dataset", "/data/dataset/stage1_train_clap_rms0_99.json", "openai/whisper-large-v3-turbo", True)
loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=16,
    pin_memory=True,
    collate_fn=dataset.collater,
    drop_last=False,
)

from tqdm import tqdm

# tqdm을 사용하여 진행 상황 표시
for samples in tqdm(loader, desc="Loading Data", unit="batch"):
    # Your processing logic here...
    pass
