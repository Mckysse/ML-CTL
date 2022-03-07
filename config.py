# -*- coding: utf-8 -*-
"""
@author: bdchen

"""

import argparse

argparser = argparse.ArgumentParser(description=("Run bert for single sentence."))


argparser.add_argument("--total_train_examples", type=int, default=1000000,
                       help="Number of total trainset.")
argparser.add_argument("--warmup_proportion", type=float, default=0.1,
                       help=("Proportion of training to perform linear learning rate warmup for."))
argparser.add_argument("--batch_size", type=int, default=64,
                       help="Number of instances per batch.")
argparser.add_argument("--bucket_size", type=int, default=2048,
                       help=("The size of the bucket."))
argparser.add_argument("--neg_threshold", type=float, default=0.8,
                       help=("The threshold to choose negtive sample."))
argparser.add_argument("--num_epochs", type=int, default=400,
                       help=("Number of epochs to perform in training."))
argparser.add_argument("--max_seq_length", type=int, default=128,
                       help=("The maximum length of a sentence at the word level. Longer sentences will be truncated, and shorter ones will be padded."))
argparser.add_argument("--short_seq_prob", type=float, default=0.1,
                       help=("The probility to get short sequence"))
argparser.add_argument("--max_predictions_per_seq", type=int, default=5,
                       help="max_predictions_per_seq")
argparser.add_argument("--masked_lm_prob", type=float, default=0.15,
                       help="masked_lm_prob")
argparser.add_argument("--log_dir", type=str, default='./logs/',
                       help=("Directory to save logs to."))
argparser.add_argument("--model_dir", type=str, default='./model',
                       help=("Directory to save model checkpoints to."))

argparser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")


argparser.add_argument("--do_lower_case", action='store_true')

argparser.add_argument("--lr", type=float, default=2e-6,
                       help=('the initial learning rate'))
argparser.add_argument("--is_retrain", action="store_true",default=False,
                       help=("Whether to re-train the model from a loaded model"))

argparser.add_argument("--t", type=float, default=0.1,#0.0001,
                       help=('the initial learning rate'))

config = argparser.parse_args()
