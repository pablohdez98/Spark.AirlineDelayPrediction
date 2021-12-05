def questions():

    file_year = ml_algorithm = tunning_type = analize = -1

    # User inputs
    while file_year not in range(1987, 2009, 1):
        try:
            file_year = int(input("Enter year of file (1987-2008): "))
        except:
            print('Not a valid year')

    while ml_algorithm not in range(1, 3, 1):
        try:
            ml_algorithm = int(input("Enter the number of the algorithm desired:"
                                     "\n(1) Linear Regression"
                                     "\n(2) Decision Tree"
                                     "\n-> "))
        except:
            print('Not a valid algorithm')

    if ml_algorithm == 1:
        while tunning_type not in range(1, 4, 1):
            try:
                tunning_type = int(input("Enter the number of how you want to run it:"
                                         "\n(1) Default parameters"
                                         "\n(2) Tunning by splitting the train into train and dev"
                                         "\n(3) Tunning by applying cross-validation with k=5"
                                         "\n-> "))
            except:
                print('Not a valid number')

    while analize not in range(0, 2, 1):
        try:
            analize = int(input("Would you like to perform some analysis? (0: No, 1: Yes): "))
        except:
            print('Not a valid number')

    return file_year, ml_algorithm, tunning_type, analize
