from copy import copy


class ArgumentType:
    UNKNOWN             = 0
    INVALID_ARGUMENT    = 1
    RAW_STRING          = 2
    CLASSIFY            = 3
    EXTRACT             = 4

    # -e options
    PHONEME             = 11
    CORRECT_SPELLING    = 12

    # -c classifiers
    T1_CLASSIFY         = 21
    T2_CLASSIFY         = 22
    LDA_CLASSIFY        = 23
    DNN_CLASSIFY        = 24





class Argument:

    def __init__(self, str_arg="", type_id=ArgumentType.INVALID_ARGUMENT, arr_sub_args=None):
        self.string = str_arg
        self.type_id = type_id
        self.sub_args = arr_sub_args

        # if the user does not pass a list
        if self.sub_args is None:
            self.sub_args = []
        # convert to one
        if type(self.sub_args) is not list:
            self.sub_args = [self.sub_args]

    def get_type(self):
        return self.type_id

    def get_string(self):
        return self.string

    def get_sub_args(self):
        return copy(self.sub_args)

    def append_sub_args(self, new_arg):
        self.sub_args.append(new_arg)

    def iter_sub_args(self):
        for x in self.sub_args:
            yield x

    # types = [type,type,type.....]
    def find_last_subarg_of_types(self, types):
        for arg in self.sub_args[::-1]:
            if arg.get_type() in types:
                return arg
        return None

    def __iter__(self):
        for s_arg in self.sub_args:
            yield s_arg

    def __eq__(self, other):
        if self.string == other.string and self.type_id == other.type_id:
            return True
        return False