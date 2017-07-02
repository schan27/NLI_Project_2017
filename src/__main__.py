
# system imports
from sys import argv
import sys
import os
from glob import glob

# add NLI_Project_2017 directory to system path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# user imports
from src.util.Misc import expand_paths
from src.util.Argument import Argument, ArgumentType
from src.feature_extraction.Phoneme_extraction import extract_phonemes
from src.feature_extraction.Spelling_correction import correct_spelling

# classifiers
from src.Classifiers.ClassifierFrameWork import ClassifierFrameWork
from src.Classifiers.LDA import LDA
from src.Classifiers.DNNClassifier import DNNC
from src.Classifiers.BNNClassifier import BNN
from src.Classifiers.SVM import SVM
from src.Classifiers.Ensemble import Ensemble

def print_usage():
    print("python src [-c <[classifiers] -t1 <training files> -t2 <test files> ] [ -e  [options]]  ")
    print("------------------------")
    print("-e [options] <filePath> <filePath2> ..... <filePathX>")
    print("[options]")
    print("\t-p extract phonemes from target file(s)")
    print("\t-c correct spelling in target file(s)")
    print("-c [classifier] -t1 <training files .....> -t2 <test files....>")
    print("[classifier]")
    print("\t-lda use Linear Discriminant Analysis")
    print("\t-dnn use Deep Neural Network")
    print("\t-bnn use Ben Neural Network")
    print('\t-svm use SVM')
    print("\t-en use Ensemble")



def parse_argv():
    if len(argv) <= 1:
        print_usage()

    arg_list = []
    arg_ptr = Argument("",ArgumentType.UNKNOWN)

    try:
        for arg in argv:
            # -e extraction
            if arg == "-e":
                arg_ptr = Argument(arg,ArgumentType.EXTRACT)
                arg_list.append(arg_ptr)
            elif arg == "-c":
                # -c classification task
                arg_ptr = Argument(arg,ArgumentType.CLASSIFY)
                arg_list.append(arg_ptr)
            # -e options
            elif arg == "-p" and arg_ptr.get_type() == ArgumentType.EXTRACT:
                arg_ptr.append_sub_args(Argument(arg,ArgumentType.PHONEME))
                
            elif arg == '-c' and arg_ptr.get_type() == ArgumentType.EXTRACT:
                arg_ptr.append_sub_args(Argument(arg, ArgumentType.CORRECT_SPELLING))

            # -c options
            elif arg == "-t1" and arg_ptr.get_type() == ArgumentType.CLASSIFY:
                # following args are training file path names
                arg_ptr.append_sub_args(Argument(arg,ArgumentType.T1_CLASSIFY))
            elif arg == "-t2" and arg_ptr.get_type() == ArgumentType.CLASSIFY:
                # following args are testing file path names
                arg_ptr.append_sub_args(Argument(arg, ArgumentType.T2_CLASSIFY))

            # -c classifiers
            elif arg == "-lda" and arg_ptr.get_type() == ArgumentType.CLASSIFY:
                arg_ptr.append_sub_args(Argument(arg,ArgumentType.LDA_CLASSIFY))

            elif arg == "-dnn" and arg_ptr.get_type() == ArgumentType.CLASSIFY:
                arg_ptr.append_sub_args(Argument(arg,ArgumentType.DNN_CLASSIFY))

            elif arg == "-bnn" and arg_ptr.get_type() == ArgumentType.CLASSIFY:
                arg_ptr.append_sub_args(Argument(arg,ArgumentType.BNN_CLASSIFY))

            elif arg == '-svm' and arg_ptr.get_type() == ArgumentType.CLASSIFY:
                arg_ptr.append_sub_args(Argument(arg, ArgumentType.SVM_CLASSIFY))

            elif arg == '-en' and arg_ptr.get_type() == ArgumentType.CLASSIFY:
                arg_ptr.append_sub_args(Argument(arg, ArgumentType.EN_CLASSIFY))

            else:
                # not a normal tag. check for command input.
                if arg_ptr.get_type() == ArgumentType.EXTRACT:
                    # we have seen extract command! and there is some input
                    # so it must be file path (I hope).
                    arg_ptr.append_sub_args(Argument(arg,ArgumentType.RAW_STRING))

                elif arg_ptr.get_type() == ArgumentType.CLASSIFY:
                    # arg should be a input file.
                    last_T_flag = arg_ptr.find_last_subarg_of_types([ArgumentType.T1_CLASSIFY,ArgumentType.T2_CLASSIFY])
                    if last_T_flag is not None:
                        last_T_flag.append_sub_args(Argument(arg,ArgumentType.RAW_STRING))
                    else:
                        print("ERROR: bad argument format!")
                        print_usage()
                        sys.exit(-1)

    except IndexError:
        print_usage()
        print("ERROR: Unknown argument error.")
        sys.exit(-1)


    return arg_list



# /////////////// main /////////////////////
def main():
    args = parse_argv()

    for arg in args:
        # execute action based on arg type
        # feature extraction section
        if arg.get_type() == ArgumentType.EXTRACT:
            feature_extraction = {ArgumentType.PHONEME, ArgumentType.CORRECT_SPELLING}
            
            options = list(filter(lambda x: x.get_type() in feature_extraction, arg.get_sub_args()))
            targets = list(filter(lambda x: x.get_type() == ArgumentType.RAW_STRING, arg.get_sub_args()))


            # error checks
            # do we have at least one option
            if len(options) < 1:
                print_usage()
                break
            # do we have at least one target
            if len(targets) < 1:
                print_usage()
                break

            # process operation
            for opt in options:
                if opt.get_type() == ArgumentType.PHONEME:
                    for target in targets:
                        extract_phonemes(target.get_string())
                        
                elif opt.get_type() == ArgumentType.CORRECT_SPELLING:
                    for target in targets:
                        correct_spelling(target.get_string())

        elif arg.get_type() == ArgumentType.CLASSIFY:
            training_files = arg.find_last_subarg_of_types([ArgumentType.T1_CLASSIFY])
            testing_files = arg.find_last_subarg_of_types([ArgumentType.T2_CLASSIFY])

            if training_files is None or testing_files is None:
                print("ERROR: you must specify both of -t1 and -t2")
                print_usage()
                sys.exit(-1)

            # expand paths (because PyCharm does not)
            training_files = expand_paths(training_files)
            testing_files = expand_paths(testing_files)

            # load classifier framework
            cfw = ClassifierFrameWork()
            cfw.load_label_file("nli-shared-task-2017/data/labels/dev/labels.dev.csv")
            cfw.load_label_file("nli-shared-task-2017/data/labels/train/labels.train.csv")

            # load feature files
            for train_file in training_files:
                if train_file.get_type() == ArgumentType.RAW_STRING:
                    cfw.load_data_from_file(train_file.get_string())
            for test_file in testing_files:
                if test_file.get_type() == ArgumentType.RAW_STRING:
                    cfw.load_data_from_file(test_file.get_string(),True)

            # run classifiers on given data
            for sarg in arg:
                if sarg.get_type() == ArgumentType.LDA_CLASSIFY:
                    cfw.add_classifier(LDA())
                elif sarg.get_type() == ArgumentType.DNN_CLASSIFY:
                    cfw.add_classifier(DNNC())
                elif sarg.get_type() == ArgumentType.BNN_CLASSIFY:
                    cfw.add_classifier(BNN())
                elif sarg.get_type() == ArgumentType.SVM_CLASSIFY:
                    cfw.add_classifier(SVM())
                elif sarg.get_type() == ArgumentType.EN_CLASSIFY:
                    cfw.add_classifier(Ensemble())

            # do classification task
            cfw.preprocess_data()
            cfw.train()
            cfw.test()
            cfw.check_results()

    print("done!")


main()

