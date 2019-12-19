import os
import shutil
import sys

# detect the current working directory and print it
path = os.path.dirname(os.path.abspath(__file__))
print("The current working directory is %s" % path)


def yes_or_no(question):
    reply = str(input(question + ' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... please enter ")


def create_folder(name_string):
    if name_string is not None:
        path_create = path + '/' + name_string

        if os.path.isdir(path_create):
            print("Directory already exists")
            if yes_or_no('Do you want to rewrite the directory?'):
                shutil.rmtree(path_create)

        try:
            os.mkdir(path_create)
        except OSError:
            print("Creation of the directory %s failed" % path_create)
            sys.exit("Cannot create directory - cannot save model. No point in training. Abort mission.")
        else:
            print("Successfully created the directory %s " % path_create)

    return path_create