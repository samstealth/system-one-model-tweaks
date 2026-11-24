"""Download the public sources build_choice_data.py reads (about 320 MB) into one folder.

    python -m innovations.fetch_choice_src --out choice_src
"""
import argparse
import os
import urllib.request

B = "https://huggingface.co/datasets"
FILES = {
    "PolyAI_banking77__data_train-00000-of-00001.parquet": "PolyAI/banking77/resolve/main/data/train-00000-of-00001.parquet",
    "PolyAI_banking77__data_test-00000-of-00001.parquet": "PolyAI/banking77/resolve/main/data/test-00000-of-00001.parquet",
    "banking_readme.md": "PolyAI/banking77/resolve/main/README.md",
    "clinc_train.parquet": "clinc/clinc_oos/resolve/main/plus/train-00000-of-00001.parquet",
    "clinc_test.parquet": "clinc/clinc_oos/resolve/main/plus/test-00000-of-00001.parquet",
    "clinc_readme.md": "clinc/clinc_oos/resolve/main/README.md",
    "allenai_ai2_arc__ARC-Challenge_train-00000-of-00001.parquet": "allenai/ai2_arc/resolve/main/ARC-Challenge/train-00000-of-00001.parquet",
    "allenai_ai2_arc__ARC-Challenge_test-00000-of-00001.parquet": "allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet",
    "allenai_ai2_arc__ARC-Easy_train-00000-of-00001.parquet": "allenai/ai2_arc/resolve/main/ARC-Easy/train-00000-of-00001.parquet",
    "tau_commonsense_qa__data_train-00000-of-00001.parquet": "tau/commonsense_qa/resolve/main/data/train-00000-of-00001.parquet",
    "tau_commonsense_qa__data_validation-00000-of-00001.parquet": "tau/commonsense_qa/resolve/main/data/validation-00000-of-00001.parquet",
    "mmlu_aux.parquet": "cais/mmlu/resolve/main/auxiliary_train/train-00000-of-00001.parquet",
    "glaiveai_glaive-function-calling-v2__glaive-function-calling-v2.json": "glaiveai/glaive-function-calling-v2/resolve/main/glaive-function-calling-v2.json",
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for name, path in FILES.items():
        dst = os.path.join(a.out, name)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(f"{B}/{path}", dst)
        print(name, os.path.getsize(dst), flush=True)


if __name__ == "__main__":
    main()
