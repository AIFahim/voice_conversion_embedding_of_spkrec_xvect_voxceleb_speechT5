import os
import glob
import numpy
import argparse
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchaudio.transforms as T

spk_model = {
    "speechbrain/spkrec-xvect-voxceleb": 512, 
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}

def f2embed(wav_file, classifier, size_embed):
    signal, fs = torchaudio.load(wav_file)

    resample_rate = 16000
    transform = T.Resample(fs, resample_rate, dtype=signal.dtype)
    signal = transform(signal)
    signal = torch.mean(signal, dim=0, keepdim=True)

    # assert fs == 16000, fs
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings

def process(args):
    wavlst = []
    # for split in args.splits.split(","):
    # wav_dir = os.path.join(args.arctic_root, split)
    # print("wav_dir",wav_dir)
    files = glob.glob(os.path.join(args.arctic_root, "*.wav"))

    wavlst_split = args.arctic_root #glob.glob(os.path.join(args.arctic_root, "wav", "*.wav"))
    # print(wavlst_split)
    # print(f"{split} {len(wavlst_split)} utterances.")
    wavlst.extend(files)

    # print()

    print("wavlst ",wavlst)

    # print(wavlst)
    spkemb_root = args.output_root
    if not os.path.exists(spkemb_root):
        print(f"Create speaker embedding directory: {spkemb_root}")
        os.mkdir(spkemb_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=args.speaker_embed, run_opts={"device": device}, savedir=os.path.join('/tmp', args.speaker_embed))
    size_embed = spk_model[args.speaker_embed]
    for utt_i in tqdm(wavlst, total=len(wavlst), desc="Extract"):
        # TODO rename speaker embedding
        # utt_id = "-".join(utt_i.split("/")[-3:]).replace(".wav", "")
        utt_id = utt_i.split(".")[0].split("/")[-1] #args.arctic_root.split(".")[-1]
        # print(utt_id)
        # print("utt_id",utt_id)
        utt_emb = f2embed(utt_i, classifier, size_embed)
        numpy.save(os.path.join(spkemb_root, f"{utt_id}.npy"), utt_emb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic-root", "-i", required=True, type=str, help="LibriTTS root directory.")
    parser.add_argument("--output-root", "-o", required=True, type=str, help="Output directory.")
    parser.add_argument("--speaker-embed", "-s", type=str, required=True, choices=["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb"],
                        help="Pretrained model for extracting speaker emebdding.")
    # parser.add_argument("--splits",  type=str, help="Split of four speakers seperate by comma.",
    #                     default="cmu_us_bdl_arctic,cmu_us_clb_arctic,cmu_us_rms_arctic,cmu_us_slt_arctic")
    args = parser.parse_args()
    print(
        # f"Loading utterances from {args.arctic_root}/{args.splits}, "
        # +
        f"Save speaker embedding 'npy' to {args.output_root}, "
        + f"Using speaker model {args.speaker_embed} with {spk_model[args.speaker_embed]} size.")
    process(args)

if __name__ == "__main__":
    """
    python utils/prep_cmu_arctic_spkemb.py \
        -i /root/data/cmu_arctic/CMUARCTIC \
        -o /root/data/cmu_arctic/CMUARCTIC/spkrec-xvect \
        -s speechbrain/spkrec-xvect-voxceleb
    
    python /home/asif/augmentations_experiments/voice_conversion_Speaker_emb_SpeechT5.py \
         -i /home/asif/augmentations_experiments/augmentations/downupsampling \
         -o /home/asif/augmentations_experiments/speaker_emb \
         -s speechbrain/spkrec-xvect-voxceleb
    """
    main()