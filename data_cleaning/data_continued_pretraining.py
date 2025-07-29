import os
import json
from pathlib import Path


def generate_jsonl_from_folder(folder_path, output_jsonl):
    """
    Reads all files in the specified folder and writes their contents to a JSONL file.
    Each line in the JSONL will be a JSON object: {"text": <file_content>}.
    """
    folder = Path(folder_path)
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_jsonl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for file_path in folder.iterdir():
            if file_path.is_file():
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    content = in_f.read()
                    json_line = json.dumps({'text': content}, ensure_ascii=False)
                    out_f.write(json_line + '\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate JSONL from all files in a folder.")
    parser.add_argument('--folder', type=str, default='C:/Users/Andres.DESKTOP-D77KM25/OneDrive - Stanford/Laboral/Trabajo independiente/Lawgorithm/Corte Constitucional/T-1992-to-2023/2023', help='Path to the folder containing text files')
    parser.add_argument('--output', type=str, default='data/data_continued_pretraining/data_continued_pretraining.jsonl', help='Output JSONL file path')
    args = parser.parse_args()
    generate_jsonl_from_folder(args.folder, args.output)






