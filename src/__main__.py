
# system imports
from sys import argv
import sys
import os
import re

# add NLI_Project_2017 directory to system path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# user imports
from src.util.Argument import Argument, ArgumentType
from src.feature_extraction.Phoneme_extraction import extract_phonemes
from src.feature_extraction.Spelling_correction import correct_spelling

def print_usage():
    print("python src [ -e  [-p]] ")
    print("------------------------")
    print("-e [options] <filePath> <filePath2> ..... <filePathX>")
    print("[options]")
    print("\t-p extract phonemes from target file(s)")
    print("\t-c correct spelling in target file(s)")


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

            # -e options
            elif arg == "-p" and arg_ptr.get_type() == ArgumentType.EXTRACT:
                arg_ptr.append_sub_args(Argument(arg,ArgumentType.PHONEME))
                
            elif arg == '-c' and arg_ptr.get_type() == ArgumentType.EXTRACT:
                arg_ptr.append_sub_args(Argument(arg, ArgumentType.CORRECT_SPELLING))

            else:
                # not a normal tag. check for command input.
                if arg_ptr.get_type() == ArgumentType.EXTRACT:
                    # we have seen extract command! and there is some input
                    # so it must be file path (I hope).
                    arg_ptr.append_sub_args(Argument(arg,ArgumentType.RAW_STRING))

    except IndexError:
        print_usage()
        print("ERROR: Unknown argument error.")


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

    print("done!")


main()