# main.py
from gsm8k_eval.args import parse_arguments
from gsm8k_eval.evaluation import run_evaluation

def main():
    # parse arguments
    args = parse_arguments()

    # run evaluation
    run_evaluation(args)

if __name__ == "__main__":
    # kick off
    main()
