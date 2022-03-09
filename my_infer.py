import torch
import argparse
import soundfile as sf
import torch.nn.functional as F
import itertools as it
from fairseq import tasks
from fairseq import checkpoint_utils
from sew_asapp.models.wav2vec2_asr_v2 import Wav2VecCtcV2
from sew_asapp.decoder.ctc_decoder import CTCArgMaxDecoder

def configure_parser():
    parser = argparse.ArgumentParser(description='Wav2vec-2.0 Recognize')
    parser.add_argument('--wav_path', 
                        type=str,
                        help='path of wave file')
    parser.add_argument('--w2v_path', 
                        type=str,
                        help='path of pre-trained wav2vec-2.0 model')
    parser.add_argument('--target_dict_path', 
                        type=str,
                        help='path to directory with target dict (dict.ltr.txt)')
    return parser

def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == 'wordpiece':
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == 'letter':
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol is not None and symbol != 'none':
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence


def load_model_and_task(model_path, dict_dir='/home/ivainn/Alex/Golos_labels'):
    state = checkpoint_utils.load_checkpoint_to_cpu(model_path)
    cfg = state['cfg']
    cfg.task.data = dict_dir
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    model.load_state_dict(state['model'], strict=True)

    return model, task

def get_feature(filepath):
    def postprocess(feats, sample_rate):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats

    wav, sample_rate = sf.read(filepath)
    feats = torch.from_numpy(wav).float()
    feats = postprocess(feats, sample_rate)
    return feats


def main():
    parser = configure_parser()
    args = parser.parse_args()
    sample = dict()
    net_input = dict()

    feature = get_feature(args.wav_path)
    
    model, task = load_model_and_task(args.w2v_path, args.target_dict_path)
    model.eval()

    
    # please do not touch this unless you test both generate.py and infer.py with audio_pretraining task
    generator = CTCArgMaxDecoder(None, task.target_dictionary)
    net_input["source"] = feature.unsqueeze(0)  

    padding_mask = torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0)         
    net_input["padding_mask"] = padding_mask
    sample["net_input"] = net_input

    with torch.no_grad():
        hypo = generator.generate([model], sample, prefix_tokens=None)
    
    hyp_pieces = task.target_dictionary.string(hypo[0][0]["tokens"].int().cpu())
    print(post_process(hyp_pieces, 'letter'))

if __name__=='__main__':
    main()