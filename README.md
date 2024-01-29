# Final Matching Algorithm

This repository contains a Python script for a matching algorithm that calculates the distance between two students, one incoming and one local based on various factors such as interests, gender preference, university, university faculty, and date available.

## Description
The `DataPreprocessing.py` script reads in a dataset of student preferences and performs data preprocessing to clean the data and prepare it for the matching algorithm. The script uses pandas for data manipulation and the `datetime` library to convert the date format of the dataset to a format that can be used by the matching algorithm.

The script, `FinalMatchingAlgorithm.py`, uses pandas for data manipulation and a custom distance function to calculate the distance between two students. The distance calculation takes into account various factors, including the users' interests and additional factors such as gender preference, university, university faculty, and date available. The factors are weighted differently according to their importance in the matching process. This script reads in a dataset of student preferences, performs a matching algorithm to pair students based on their preferences, and outputs the matches to an Excel file.

The `MatchingPipeline.py` script is a pipeline that runs the `DataPreprocessing.py` and `FinalMatchingAlgorithm.py` scripts in sequence. This script also asks for some user inputs to tune the settings of the run.

## Input
Owing to the presence of confidential information being contained in the input folders they cannot be uploaded to this github, but feel free to contact the developers to talk about the format and contents.

## Output
Most of the output also contains confidential information, however one form of output has been included as an example for this repo which is some statistics about the quality of the matches based on how much the interests of students are aligned. These can be found in the `example_output_report.md` file.

## Usage
Before continuing with the following steps ensure that you have a virtual environment set up on your machine. If you do not, you can follow the instructions [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to set one up.

## Running the Algorithm(s)
To run the script, you'll need Python installed on your machine. You'll also need the several other libraries, which you can install with pip once you have successfully set up a virtual environment and entered it:

```bash
pip install -r requirements.txt
```
Once these have all been successfully installed, you can run the script with the following command:
```bash
python3 MatchingPipeline.py
```
In this script it will first ask you to enter the name of the file in which the local student data is kept (without the .csv) and then the file in which the incoming student data is kept (also without .csv). These files are both assumed to be in the inputs folder.

Then it will ask if this is a test run - if yes it will allow you to test a date you want to simulate the run being on, if not it will assume you want to run it in an actual capacity which means your current date and time. Then it will run both parts of the system and send the outputs from this run into one file. This file will be in the outputs folder and will be named with the date and time of the run.
