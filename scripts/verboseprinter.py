
# There is probably already a standard Python function for this?
class VerbosePrinter:
    @staticmethod
    def print(text, verbose=False):
        if verbose:
            print(text)

