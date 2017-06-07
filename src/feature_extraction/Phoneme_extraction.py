from src.lib.English_to_IPA_master.conversion import convert
from os import path, mkdir


# targets single file
# the file will have phoneme extraction performed upon it.
# return True on success
# return False on failure
def extract_phonemes(target):
    if path.isfile(target):
        return __extract_phonemes(target)

# target is a single file
# returns True on success
# returns False on failure
def __extract_phonemes(target):

    # create output directory if it does not already exists
    if not path.isdir(path.join(path.dirname(target),"phoneme_out/")):
        mkdir(path.join(path.dirname(target),"phoneme_out/"))

    file_in = None
    file_out = None
    try:
        file_in = open(target,"r")

        output_buffer = []
        for line in file_in.readlines():
            output_buffer.append(convert(line))

        file_out = open(path.join(path.dirname(target),"phoneme_out/")+ path.basename(target) +".phoneme", "w")
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