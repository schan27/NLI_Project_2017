from src.lib.English_to_IPA_master.conversion import convert
from os import path, listdir
import re

# targets single file or a directory
# if the target is a directory all .txt files in the directory
# will have phoneme extraction performed upon them.
# return True on success
# return False on failure
def extract_phonemes(target):
    if path.isfile(target):
        return __extract_phonemes(target)
    elif path.isdir(target):
        for file in listdir(target):
            if re.match(r".*\.txt$", file) is not None:
                if not __extract_phonemes(path.join(target,file)):
                    return False
        return True
    else:
        raise FileNotFoundError

# target is a single file
# returns True on success
# returns False on failure
def __extract_phonemes(target):

    file_in = None
    file_out = None
    try:
        file_in = open(target,"r")

        output_buffer = []
        for line in file_in.readlines():
            output_buffer.append(convert(line))

        file_out = open(target+".phoneme", "w")
        file_out.writelines(output_buffer)

    except IOError as e:
        print("ERROR: when reading file: " + target + " error message: " + str(e) )
        return False
    finally:
        if file_in is not None:
            file_in.close()
        if file_out is not None:
            file_out.close()

    return True