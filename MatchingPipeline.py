import os

from datetime import datetime

from DataPreprocessing import DataPreprocessing
from FinalMatchingAlgorithm import MatchingAlgorithm

if __name__ == "__main__":
    print(f"Welcome to the Buddy Program Matching Pipeline!\n")
    local_student_filename = input("Please enter the name of the .csv file containing the local students - this file should be in the inputs folder: ").strip() + ".csv"
    incoming_student_filename = input("Please enter the name of the .csv file containing the incoming students - this file should be in the inputs folder: ").strip() + ".csv"
    test_run = True if input("Is this a test run? (y/n): ").strip().lower() == 'y' else False

    date_desired = None
    date_format = "%d-%m-%Y"

    if test_run:
        while date_desired is None:
            date_input = input("Please enter the date you would like to test the algorithm and preprocessing from in the format (DD-MM-YYYY): ").strip()
            try:
                date_desired = datetime.strptime(date_input, date_format)
            except ValueError:
                print("That's not a valid date. Please try again.")

    output_folder = 'output/run_final_matching_algorithm_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(output_folder)

    print("\nRunning the data preprocessing pipeline...\n")
    if test_run:
        data_preprocessing = DataPreprocessing(local_student_filename, incoming_student_filename, output_folder, current_date=date_desired)
    else:
        data_preprocessing = DataPreprocessing(local_student_filename, incoming_student_filename, output_folder)
    data_preprocessing.run_preprocessing()
    print("Data preprocessing pipeline finished.\n")

    print("Running the matching algorithm...\n")
    if test_run:
        matching_algorithm = MatchingAlgorithm(output_folder, current_date=date_desired)
    else:
        matching_algorithm = MatchingAlgorithm(output_folder)
    matching_algorithm.run_algorithm()
    print("Matching algorithm finished.\n")
    print(f"The results of this run can be found in the {output_folder} folder.\n========================================")