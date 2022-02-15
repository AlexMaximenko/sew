import argparse
import os
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcriptions = {}
    data = {}

    with open(args.manifest, 'r') as manifest:
        for line in manifest:
            record = json.loads(line)
            data[record['audio_filepath']] = record['text']

    with open(args.tsv, 'r') as tsv, open(
        os.path.join(args.output_dir, args.output_name + '.phn'), 'w'
    ) as phn_out:
        root = next(tsv).strip()
        for line in tsv:
            line = line.strip()
            audio_filepath = line[0:line.find('\t')]
            try:
                text = data[audio_filepath]
                print(text, file=phn_out)
                #break;
            except:
                pass

if __name__ == '__main__':
    main()

