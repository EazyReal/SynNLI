"""
auhor: ytlin
usage: convert csv to jsonl format
command: python csv2jsonl.py -i <csv-file-name> -o <jsonl-file-name>
caveat: the default seperator (delimiter) is '\t'
"""

import csv
import json
import argparse
import logging

"""
usage:
@MNLI/data/hans
python ../../src/data_git/csv2jsonl.py -i ./heuristics_evaluation_set.txt -o ./dev.jsonl
"""

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # parse arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", "-i", type=str, required=True)
    parser.add_argument("--output_jsonl", "-o", type=str, required=True)
    args = parser.parse_args()
    # conversion
    with open(args.input_csv, 'r') as csvfile:
        with open(args.output_jsonl, 'w') as jsonfile:
            reader = csv.DictReader(csvfile, delimiter='\t') # use default fields
            for row in reader:
                json.dump(row, jsonfile)
                jsonfile.write('\n')
    # msg
    logger.info(msg=f"first three lines of jsonl {str(args.output_jsonl)}")
    with open(args.output_jsonl, 'r') as jsonfile:
        for i in range(3):
            example = jsonfile.readline()
            example = json.loads(example)
            logger.info(msg=f"{example}")
        
    logger.info(msg="csv to jsonl conversion done")