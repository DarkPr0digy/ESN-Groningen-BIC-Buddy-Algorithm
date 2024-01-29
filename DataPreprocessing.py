import os

import pandas as pd
from datetime import datetime, timedelta


class DataPreprocessing:
    def __init__(self, local_student_filename: str, incoming_student_filename: str, output_folder: str, current_date=None):
        self.output_folder = output_folder
        self.current_date = current_date
        try:
            # Importing the datasets
            self.local_students = pd.read_csv("inputs/" + local_student_filename)
            self.incoming_students = pd.read_csv("inputs/" + incoming_student_filename)
        except FileNotFoundError:
            print(
                "[DATAPREPROCESSING]: Please ensure that the input files are in the input folder and that the filenames have been entered correct.")
            exit()

    def run_preprocessing(self):
        self._clean_data()

        # Rename the first column to "Timestamp"
        first_column_name_local = self.local_students.columns[0]
        first_column_name_incoming = self.incoming_students.columns[0]
        self.local_students.rename(columns={first_column_name_local: "Timestamp"}, inplace=True)
        self.incoming_students.rename(columns={first_column_name_incoming: "Timestamp"}, inplace=True)

        removed_local_students, removed_incoming_students = self._apply_filters()

        self._save_data(removed_local_students, removed_incoming_students)
        print('Done preprocessing data.\n')

    def _clean_data(self):
        self.local_students.columns = self.local_students.columns.str.replace("''", '"').str.strip()
        self.incoming_students.columns = self.incoming_students.columns.str.replace("''", '"').str.strip()

        self.local_students[
            'From approximately which date are you available to physically meet your buddy match?'] = pd.to_datetime(
            self.local_students['From approximately which date are you available to physically meet your buddy match?'], errors='coerce', format='%d/%m/%Y').dt.date
        self.local_students[
            'From approximately which date are you available to answer questions from your buddy match? (through mail, phone, whatsapp etc.)'] = pd.to_datetime(
            self.local_students[
                'From approximately which date are you available to answer questions from your buddy match? (through mail, phone, whatsapp etc.)'],
             errors='coerce', format='%d/%m/%Y').dt.date

        self.incoming_students['When will you arrive in Groningen (approximately)?'] = pd.to_datetime(
            self.incoming_students['When will you arrive in Groningen (approximately)?'], errors='coerce', format='%d-%m-%Y').dt.date

    def _filter_incoming_student(self, row):
        date_arriving = pd.to_datetime(row['When will you arrive in Groningen (approximately)?'], format='%d/%m/%Y')

        if self.current_date is None:
            three_months_from_now = datetime.now() + timedelta(days=90)
        else:
            three_months_from_now = self.current_date + timedelta(days=90)

        if date_arriving < datetime.now():
            row.loc['When will you arrive in Groningen (approximately)?'] = datetime.today()
            date_arriving = datetime.today()

        # If the student is arriving later than three months from now, return 'Arriving too late'
        if not date_arriving <= three_months_from_now:
            return 'Arriving too late'

        try:
            disability_status = row[
                'Do you have any accessibility requirements you would like us to be aware of or need any sort of support?']

            if disability_status == 'Yes (please fill in below)':
                return 'Incoming student disability'
        except KeyError:
            pass

        return None

    def _filter_local_student(self, row):
        date_available = pd.to_datetime(
            row['From approximately which date are you available to physically meet your buddy match?'],
            format='%d/%m/%Y')

        if self.current_date is None:
            three_months_from_now = datetime.now() + timedelta(days=90)
        else:
            three_months_from_now = self.current_date + timedelta(days=90)

        if date_available is pd.NaT:
            return 'Date not entered correctly - reformat and read to input'

        if date_available < datetime.now():
            row.loc[
                'From approximately which date are you available to physically meet your buddy match?'] = datetime.today()
            date_available = datetime.today()

        if not date_available <= three_months_from_now:
            return 'Available too late'

        return None

    def _apply_filters(self):
        print('Filtering local and incoming students')
        self.local_students['reason'] = self.local_students.apply(self._filter_local_student, axis=1)
        self.incoming_students['reason'] = self.incoming_students.apply(self._filter_incoming_student, axis=1)

        removed_local_students = self.local_students[self.local_students['reason'].notna()]
        removed_incoming_students = self.incoming_students[self.incoming_students['reason'].notna()]

        self.local_students = self.local_students[self.local_students['reason'].isna()]
        self.incoming_students = self.incoming_students[self.incoming_students['reason'].isna()]

        # Remove the 'reason' column from the students DataFrame
        self.local_students = self.local_students.drop(columns=['reason'])
        self.incoming_students = self.incoming_students.drop(columns=['reason'])

        return removed_local_students, removed_incoming_students

    def _save_data(self, removed_local_students, removed_incoming_students):
        print('Saving data')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.local_students.to_csv('inputs/local_students_processed.csv', index=False)
        removed_local_students.to_csv(self.output_folder + '/local_students_removed.csv', index=False)

        self.incoming_students.to_csv('inputs/incoming_students_processed.csv', index=False)
        removed_incoming_students.to_csv(self.output_folder + '/incoming_students_removed.csv', index=False)
