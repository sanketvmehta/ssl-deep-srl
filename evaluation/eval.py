import os
import subprocess
import argparse

def eval(f_script_path, f_gold, f_predicted):

    eval_cmd = ['bash', os.path.join(f_script_path, 'eval.sh'), f_gold, f_predicted]

    print('eval_cmd: ' + ' '.join(eval_cmd))

    try:
        eval_out = subprocess.check_output(eval_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        eval_out = None

    return eval_out

def extract_eval_out(eval_out):

    eval_out = eval_out.decode("utf-8")
    eval_out = eval_out.split('\n')

    # print(eval_out[8])
    results = eval_out[8].split(' ')
    results = [x for x in results if x != '']
    f1_score = float(results[-1])
    recall = float(results[-2])
    prec = float(results[-3])

    return f1_score, recall, prec

def main(eval_path, serialization_dir, split, start, end, step, contain):

    print("Split: %s", split)
    print("Epoch\tF1-measure-overall\tPrecision\tRecall")

    for i in range(start, end, step):
        f_gold = os.path.join(serialization_dir, 'gold-' + split + contain + str(i) + '.txt')
        f_predicted = os.path.join(serialization_dir, 'predictions-' + split + contain + str(i) + '.txt')
        eval_out = eval(f_script_path=eval_path, f_gold=f_gold, f_predicted=f_predicted)
        f1_score, recall, prec = extract_eval_out(eval_out=eval_out)
        print("%d\t%0.4f\t%0.4f\t%0.4f" % (i, f1_score, prec, recall))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, default="evaluation", help='The path to eval.sh script.')
    parser.add_argument('--serialization_dir', type=str, default=None,
                        help='The path where final model predictions are saved.')
    parser.add_argument('--split', type=str, default='development', help='Data split to evaluate for: test or development?')
    parser.add_argument('--start', type=int, default=0, help='Start index!')
    parser.add_argument('--end', type=int, default=1, help='End index!')
    parser.add_argument('--step', type=int, default=1, help='Step!')
    parser.add_argument('--contain', type=str, default='_', help="Whether filename contains _ OR -!")

    args = parser.parse_args()
    main(eval_path=args.eval_path, serialization_dir=args.serialization_dir,
         split=args.split, start=args.start, end=args.end, step=args.step, contain=args.contain)