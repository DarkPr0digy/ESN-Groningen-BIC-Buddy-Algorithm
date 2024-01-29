import math

import numpy as np
import pandas as pd
from configparser import ConfigParser
import os
from datetime import datetime
from munkres import Munkres, DISALLOWED
from tqdm import tqdm
from statistics import mean


class MatchingAlgorithm:
    def __init__(self, output_dir: str, current_date=None):
        """Constructor for the Matching Algorithm"""
        self.configs = None
        self.faculty_distances = None

        self.local_students_original = None
        self.local_students_copy = None

        self.incoming_students_original = None
        self.incoming_students_copy = None

        self.deny_list = None

        self.hobbies = None

        self.distance_matrix = None

        self.base_local_capacity = None
        self.base_necessity = None

        self.output_dir = output_dir

        if current_date is None:
            self.current_date = datetime.now()
        else:
            self.current_date = current_date

    def run_algorithm(self):
        """Function to run the matching algorithm"""
        # Generate output directory
        start_time = datetime.now()

        self._generate_cleaned_dataframes()

        self._internal_data_preprocessing()

        self._get_base_capacities()

        self._get_normalization_values()

        self._prepare_for_one_to_one_matching()

        self._generate_distance_matrix()

        matches, avg_distance = self._match_students()

        self._generate_report(matches,
                              avg_distance,
                              self.output_dir,
                              start_time)

    def _generate_cleaned_dataframes(self):
        """Function to generate the cleaned DataFrames"""
        # Read in config file
        self.configs = ConfigParser()

        try:
            self.configs.read('configs/config.ini')
        except FileNotFoundError as e:
            print(f"Error reading config file: {e}\nEnsure there is a config.ini file in the configs directory")
            exit()

        # Read in faculty distances
        try:
            self.faculty_distances = pd.read_excel(r'configs/faculty_distances.xlsx', index_col=0)
        except FileNotFoundError as e:
            print(f"Error reading faculty distances file: {e}\nEnsure there is a faculty_distances.xlsx file in the "
                  f"configs directory")
            exit()

        # Read in input files
        try:
            self.local_students_original = pd.read_csv(r'inputs/local_students_processed.csv')
            self.incoming_students_original = pd.read_csv(r'inputs/incoming_students_processed.csv')
            self.deny_list = pd.read_csv(r'inputs/denylist.csv')
        except FileNotFoundError as e:
            print(
                f"Error reading input files: {e}\nEnsure there are local_students_processed.csv, incoming_students_processed.csv "
                f"and denylist.csv files in the inputs directory")
            exit()

        # Strip spaces from column names
        self.local_students_original.columns = self.local_students_original.columns.str.strip()
        self.incoming_students_original.columns = self.incoming_students_original.columns.str.strip()

        # Make copies of the original DataFrames
        self.local_students_copy = self.local_students_original.copy(deep=True)
        self.incoming_students_copy = self.incoming_students_original.copy(deep=True)

        # Remove Irrelevant Columns from both Dataframes
        try:
            local_students_columns_to_drop = pd.read_csv(r'configs/local_students_irrelevant_columns.csv',
                                                         quotechar="'")
            incoming_students_columns_to_drop = pd.read_csv(r'configs/incoming_students_irrelevant_columns.csv',
                                                            quotechar="'")
        except FileNotFoundError as e:
            print(f"Error reading irrelevant columns files: {e}\nEnsure there are local_students_irrelevant_columns.csv"
                  f" and incoming_students_irrelevant_columns.csv files in the configs directory")
            exit()

        # Convert the DataFrame of columns to drop into a list
        local_students_columns_to_drop = local_students_columns_to_drop.iloc[:, 0].tolist()
        incoming_students_columns_to_drop = incoming_students_columns_to_drop.iloc[:, 0].tolist()

        # Drop the specified columns
        self.local_students_copy = self.local_students_copy.drop(columns=local_students_columns_to_drop)
        self.incoming_students_copy = self.incoming_students_copy.drop(columns=incoming_students_columns_to_drop)

    def _internal_data_preprocessing(self):
        """Function to preprocess the data"""
        # Load the column name mappings
        try:
            local_students_column_mappings = pd.read_csv('configs/local_students_column_renames.csv', quotechar="'",
                                                         skipinitialspace=True)
        except FileNotFoundError as e:
            print(f"Error reading local students column mappings file: {e}\nEnsure there is a "
                  f"local_students_column_renames.csv file in the configs directory")
            exit()

        local_students_column_mappings.columns = local_students_column_mappings.columns.str.strip()

        try:
            incoming_students_column_mappings = pd.read_csv('configs/incoming_students_column_renames.csv',
                                                            quotechar="'",
                                                            skipinitialspace=True)
        except FileNotFoundError as e:
            print(f"Error reading incoming students column mappings file: {e}\nEnsure there is a "
                  f"incoming_students_column_renames.csv file in the configs directory")
            exit()

        incoming_students_column_mappings.columns = incoming_students_column_mappings.columns.str.strip()

        # Convert the DataFrames of name mappings into a dictionary
        local_students_column_mappings = dict(
            zip(local_students_column_mappings['Old Name'], local_students_column_mappings['New Name'])
        )
        incoming_students_column_mappings = dict(
            zip(incoming_students_column_mappings['Old Name'], incoming_students_column_mappings['New Name'])
        )

        # Rename the columns in the DataFrames
        self.local_students_copy = self.local_students_copy.rename(columns=local_students_column_mappings)
        self.incoming_students_copy = self.incoming_students_copy.rename(columns=incoming_students_column_mappings)

        self.local_students_original = self.local_students_original.rename(columns=local_students_column_mappings)
        self.incoming_students_original = self.incoming_students_original.rename(
            columns=incoming_students_column_mappings)

        # Convert Categories into Numerical Values for ease of comparison
        try:
            self.hobbies = pd.read_csv('configs/hobbies.csv', quotechar="'").iloc[:, 0].tolist()
        except FileNotFoundError as e:
            print(f"Error reading hobbies file: {e}\nEnsure there is a hobbies.csv file in the configs directory")
            exit()

        hobby_options = ['Not interested', 'Interests me a little', 'Very interested']
        hobby_options_replacement = list(range(len(hobby_options)))

        for hobby in self.hobbies:
            self.local_students_copy[hobby].replace(hobby_options, hobby_options_replacement, inplace=True)
            self.incoming_students_copy[hobby].replace(hobby_options, hobby_options_replacement, inplace=True)

        frequency_options = ['One time only', 'Once a month', 'Twice a month', 'Once a week or more']
        frequency_options_replacement = list(range(len(frequency_options)))

        self.local_students_copy['MeetFrequency'].replace(frequency_options, frequency_options_replacement,
                                                          inplace=True)
        self.incoming_students_copy['MeetFrequency'].replace(frequency_options, frequency_options_replacement,
                                                             inplace=True)

        # ADJUST DATES
        def adjust_dates(date):
            date = pd.to_datetime(date).to_pydatetime()
            return date if date > self.current_date else self.current_date

        adjust_dates = np.vectorize(adjust_dates)

        self.local_students_copy['Availability'] = adjust_dates(
            pd.to_datetime(self.local_students_copy['Availability']))
        self.local_students_copy['AvailabilityText'] = adjust_dates(
            pd.to_datetime(self.local_students_copy['AvailabilityText']))
        self.incoming_students_copy['Arrival'] = adjust_dates(pd.to_datetime(self.incoming_students_copy['Arrival']))

    def _get_base_capacities(self):
        """Function to get the base capacities of the local students and the necessity of incoming students"""
        self.base_local_capacity = int(self.local_students_original['Capacity'].sum())
        self.base_necessity = int(self.incoming_students_original.count(axis=1).count())

    def _get_normalization_values(self):
        """Function to calculate and attain the values to be used for the normalization process"""
        max_local_age = self.local_students_original['Age'].max()
        max_incoming_age = self.incoming_students_original['Age'].max()

        min_local_age = self.local_students_original['Age'].min()
        min_incoming_age = self.incoming_students_original['Age'].min()

        max_age = max(max_local_age, max_incoming_age)
        min_age = min(min_local_age, min_incoming_age)

        self.age_range = max_age - min_age

        local_gender_preference_penalty = int(self.configs.get('parameters', 'local_gender_preference_penalty'))
        incoming_gender_preference_penalty = int(self.configs.get('parameters', 'incoming_gender_preference_penalty'))

        self.gender_range = local_gender_preference_penalty + incoming_gender_preference_penalty

        self.faculty_range = float(self.faculty_distances.max().max())

        self.hobby_range = 0
        for hobby in self.hobbies:
            self.hobby_range += (3 * float(self.configs.get('hobbies', hobby)))

        max_local_meeting_frequency = float(self.local_students_copy['MeetFrequency'].max())
        max_incoming_meeting_frequency = float(self.incoming_students_copy['MeetFrequency'].max())

        min_local_meeting_frequency = float(self.local_students_copy['MeetFrequency'].min())
        min_incoming_meeting_frequency = float(self.incoming_students_copy['MeetFrequency'].min())

        max_meeting_frequency = max(max_local_meeting_frequency, max_incoming_meeting_frequency)
        min_meeting_frequency = min(min_local_meeting_frequency, min_incoming_meeting_frequency)

        self.meeting_frequency_range = max_meeting_frequency - min_meeting_frequency

        max_local_physical_availability_date = self.local_students_copy['Availability'].max()
        min_local_physical_availability_date = self.local_students_copy['Availability'].min()

        max_incoming_arrival_date = self.incoming_students_copy['Arrival'].max()
        min_incoming_arrival_date = self.incoming_students_copy['Arrival'].min()

        max_dates = max(max_local_physical_availability_date, max_incoming_arrival_date)
        min_dates = min(min_local_physical_availability_date, min_incoming_arrival_date)

        self.date_range = (max_dates - min_dates).days

    def _prepare_for_one_to_one_matching(self):
        """Function to prepare data for one-to-one matching"""
        # Add a column indicating the number of students a buddy has been matched to
        if 'Matches' not in self.local_students_copy:
            capacity_index = self.local_students_copy.columns.get_loc('Capacity')
            try:
                self.local_students_original.insert(capacity_index + 2, 'Matches',
                                                    [0] * len(self.local_students_original))
                self.local_students_copy.insert(capacity_index + 2, 'Matches', [0] * len(self.local_students_copy))
            except IndexError:
                print(f"Error Creating 'Matches' Column in Local Students DataFrame")
                exit()

        # Duplicate each row in the buddies dataframe based on the number of students a buddy can be matched to
        self.local_students_copy.insert(0, 'id', range(0, len(self.local_students_copy)))

        # Repeat each local student based on their 'Capacity'
        # Sets the individual 'Capacity' of these rows to 1 and 'Extra' to 'No'.
        self.local_students_copy = self.local_students_copy.loc[
            self.local_students_copy.index.repeat(self.local_students_copy.Capacity)].assign(Capacity=1, Extra='No')

        # If necessary make use of local students that want extra buddies
        if self.base_local_capacity < self.base_necessity:
            # Get all buddies that say they don't mind an extra student
            print(f"Extra Students Needed")
            additional_local_students = self.local_students_copy.loc[
                self.local_students_copy['ExtraBuddy'] == 'Yes'].assign(Capacity=0)

            # Concat Local Students copy with the additional local students
            self.local_students_copy = pd.concat([self.local_students_copy, additional_local_students],
                                                 ignore_index=True,
                                                 axis=0).sort_values('id').reset_index(drop=True)
        else:
            self.local_students_copy = self.local_students_copy.sort_values('id').reset_index(drop=True)

    def _generate_distance_matrix(self):
        """Function to generate the distance matrix"""
        self.distance_matrix = [[self._distance(local_student, incoming_student) for incoming_student in
                                 range(0, len(self.incoming_students_copy))] for local_student in
                                tqdm(range(0, len(self.local_students_copy)), desc="Generating Distance Matrix")]

    def _match_students(self):
        """Function to match incoming and local students"""
        optimization_algorithm = Munkres()

        print("Computing Matches")

        try:
            computed_matches = optimization_algorithm.compute(self.distance_matrix)

            matches = [(self.local_students_copy.at[i, 'id'], j, self.distance_matrix[i][j]) for (i, j) in
                       computed_matches]
            average_distance = round(mean([k for (i, j, k) in matches]), 2)
        except Exception as e:
            print(f"Error in the Matching Algorithm: {type(e).__name__}, {e}")
            exit()

        return matches, average_distance

    def _age_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        """Function to calculate the distance between the ages of two students
        :param local_student: The local student
        :param incoming_student: The incoming student
        :return: The distance between the ages of the two students
        """
        local_age = local_student['Age']
        incoming_age = incoming_student['Age']

        age_difference = abs(int(local_age) - int(incoming_age))

        age_difference -= int(self.configs.get('parameters', 'desired_age_difference'))

        # Scale age difference using factorial
        age_difference_distance = math.factorial(age_difference) if age_difference > 0 else 0

        # age_difference_distance /= (math.factorial(self.age_range) * age_multiplier)
        age_difference_distance = float(age_difference_distance / (math.factorial(self.age_range)))

        return age_difference_distance

    def _age_gender_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        """Function to calculate the distance between the ages and genders of two students
        :param local_student: The local student
        :param incoming_student: The incoming student
        :return: the distance between the ages and genders of two students
        """
        distance = 0
        local_age = local_student['Age']
        local_gender = local_student['Gender']

        incoming_age = incoming_student['Age']
        incoming_gender = incoming_student['Gender']

        if local_gender != incoming_gender:
            if abs(local_age - incoming_age) > int(self.configs.get('parameters', 'desired_age_difference')):
                return 1

        return 0

    def _gender_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        """Function to calculate the distance between the genders of two students
        :param local_student: The local student
        :param incoming_student: The incoming student
        :return: the distance between the genders of two students
        """
        distance = 0
        local_gender_preference_penalty = int(self.configs.get('parameters', 'local_gender_preference_penalty'))
        incoming_gender_preference_penalty = int(self.configs.get('parameters', 'incoming_gender_preference_penalty'))

        if local_student['GenderPreference'] != 'Mix/No preference' and local_student['GenderPreference'] != \
                incoming_student['Gender']:
            distance += local_gender_preference_penalty

        if incoming_student['GenderPreference'] != 'No preference' and incoming_student['GenderPreference'] != \
                local_student['Gender']:
            distance += incoming_gender_preference_penalty

        distance = float(distance / self.gender_range)

        return distance

    def _university_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        """Function to calculate the distance between the universities of two students
        :param local_student: The local student
        :param incoming_student: The incoming student
        :return: the distance between the universities of two students
        """
        local_student_university = local_student['University']
        incoming_student_university = incoming_student['University']

        if local_student_university != incoming_student_university:
            return 1
        return 0

    def _faculty_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        """Function to calculate the distance between the faculties of two students
        :param local_student: The local student
        :param incoming_student: The incoming student
        :return: the distance between the faculties of two students
        """
        distance = float(self.faculty_distances[local_student['Faculty']].loc[incoming_student['Faculty']])
        distance /= self.faculty_range

        return distance

    def _interests_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        """Function to calculate the distance between the interests of two students
        :param local_student: The local student
        :param incoming_student: The incoming student
        :return: the distance between the interests of two students"""
        distance = 0

        for hobby in self.hobbies:
            hobby_factor = float(self.configs.get('hobbies', hobby))
            distance += abs(local_student[hobby] - incoming_student[hobby]) * hobby_factor

        distance /= self.hobby_range
        return distance

    def _availability_physical_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        """Function to calculate the distance between the availabilities of two students
        :param local_student: The local student
        :param incoming_student: The incoming student
        :return: the distance between the availabilities of two students
        """
        days_difference = (local_student['Availability'] - incoming_student['Arrival']).days

        # If the local student is available before the incoming student, the result will be below 0
        if days_difference >= 0:
            # Penalise incoming being here before local
            # days_difference **= 2
            # days_difference = float(days_difference / (self.date_range ** 2))
            days_difference = float(days_difference / (self.date_range))
            return days_difference
        return 0

    def _availability_text_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        local_student_text_date = local_student['AvailabilityText']
        incoming_student_arrival_date = incoming_student['Arrival']

        ideal_difference = float(self.configs.get('parameters', 'desired_date_difference'))

        if (incoming_student_arrival_date - local_student_text_date).days >= ideal_difference:
            return 0
        elif (incoming_student_arrival_date - local_student_text_date).days <= 0:
            return 100
        else:
            days_between = (incoming_student_arrival_date - local_student_text_date).days
            penalty = ideal_difference - days_between
            penalty = float(penalty / ideal_difference)
            return penalty

    def _meeting_frequency_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        """Function to calculate the distance between the meeting frequencies of two students
        :param local_student: The local student
        :param incoming_student: The incoming student
        :return: the distance between the meeting frequencies of two students
        """
        distance = abs(local_student['MeetFrequency'] - incoming_student['MeetFrequency'])
        distance /= self.meeting_frequency_range

        return distance

    def _expectation_distance(self, local_student: pd.Series, incoming_student: pd.Series):
        """Function to calculate the distance between the expectations of two students
        :param local_student: The local student
        :param incoming_student: The incoming student
        :return: the distance between the expectations of two students
        """
        distance = 0

        local_expectations = [int(expectation in local_student['Expectations']) for expectation in
                              ['Just answering some (practical) questions', 'Showing the new student(s) around',
                               'Becoming friends with my buddies']]
        incoming_expectations = [int(expectation in incoming_student['Expectations']) for expectation in
                                 ['Just asking (practical) questions', 'Being shown around the city',
                                  'Becoming friends with my buddy']]

        # Create a list of sums of corresponding elements in local_expectations and incoming_expectations
        comparison = [0 if x == y else 1 for x, y in zip(local_expectations, incoming_expectations)]

        # Count the number of 1s in the list
        count_ones = comparison.count(1)

        # Max is len(local_expectations) / len(incoming_expectations) because the maximum number of 1s is the number of
        # elements in local_expectations / incoming_expectations
        distance = count_ones / len(local_expectations)

        return distance

    def _distance(self, local_student_index: int, incoming_student_index: int):
        """Function to calculate the distance between two students
        :param local_student_index: The index of the local student
        :param incoming_student_index: The index of the incoming student
        :return: The distance between the two students
        """
        distance = 0

        # Obtain the two students
        local_student = self.local_students_copy.iloc[local_student_index]
        incoming_student = self.incoming_students_copy.iloc[incoming_student_index]

        self.deny_list.columns = self.deny_list.columns.str.strip().str.replace("''", '').str.replace("'", '')

        # Make sure these students aren't in the deny list
        if ((self.deny_list['Email Local'] == local_student['Email']) & (
                self.deny_list['Email Incoming'] == incoming_student['Email'])).any():
            return DISALLOWED

        if max((local_student['Availability'] - incoming_student['Arrival']).days, 0) > float(
                self.configs.get('parameters', 'maximum_unavailable_duration')):
            distance += 100

        # Get all factors
        age_factor = float(self.configs.get('normalization', 'age_factor'))
        gender_factor = float(self.configs.get('normalization', 'gender_factor'))
        age_gender_factor = float(self.configs.get('normalization', 'age_gender_factor'))
        university_factor = float(self.configs.get('normalization', 'university_factor'))
        faculty_factor = float(self.configs.get('normalization', 'faculty_factor'))
        interests_factor = float(self.configs.get('normalization', 'interests_factor'))
        availability_text_factor = float(self.configs.get('normalization', 'availability_text_factor'))
        availability_physical_factor = float(self.configs.get('normalization', 'availability_physical_factor'))
        meeting_frequency_factor = float(self.configs.get('normalization', 'meeting_frequency_factor'))
        expectation_factor = float(self.configs.get('normalization', 'expectations_factor'))

        # Multiple Factors by their respective distances
        distance += (age_factor * self._age_distance(local_student, incoming_student))
        distance += (gender_factor * self._gender_distance(local_student, incoming_student))
        distance += (age_gender_factor * self._age_gender_distance(local_student, incoming_student))
        distance += (university_factor * self._university_distance(local_student, incoming_student))
        distance += (faculty_factor * self._faculty_distance(local_student, incoming_student))
        distance += (interests_factor * self._interests_distance(local_student, incoming_student))

        distance += (availability_physical_factor * self._availability_physical_distance(local_student, incoming_student))

        try:
            distance += (availability_text_factor * self._availability_text_distance(local_student, incoming_student))
        except TypeError:
            # Catch the possibility that this is a DISSALOWED match because of availability to text
            return DISALLOWED

        distance += (meeting_frequency_factor * self._meeting_frequency_distance(local_student, incoming_student))
        distance += (expectation_factor * self._expectation_distance(local_student, incoming_student))

        return distance

    def _generate_report(self, matches: [(int, int, float)], avg_distance: float, output_dir: str,
                         start_time: datetime):
        """Function to generate a report of the matches
        :param matches: An array of tuples containing the matches calculated using Hungarian Algorithm
        :param avg_distance: The average distance between the matches
        :param output_dir: The directory to output the report to
        :param start_time: The time the algorithm started
        """
        print(f"Generating Report")

        # University Statistics
        university_equal = [
            int(self.local_students_original.loc[i, 'University'] == self.incoming_students_copy['University'].iloc[j])
            for (i, j, k) in matches]
        print(f"University Equal: {round(mean(university_equal), 2)}")

        # Faculty Statistics
        faculty_equal = [
            int(self.local_students_original.loc[i, 'Faculty'] == self.incoming_students_copy['Faculty'].iloc[j]) for
            (i, j, k) in matches]
        print(f"Faculty Equal: {round(mean(faculty_equal), 2)}")

        faculty_similar = [int(self.faculty_distances[self.local_students_original.loc[i, 'Faculty']].loc[
                                   self.incoming_students_copy['Faculty'].iloc[j]] == 8) for (i, j, k) in matches]
        print(f"Faculty Similar: {round(mean(faculty_similar), 2)}")

        # Gender Statistics
        gender_preference_local = [int(self.local_students_original.loc[i, 'GenderPreference'] == 'Mix/No preference' or
                                       self.local_students_original.loc[i, 'GenderPreference'] ==
                                       self.local_students_copy['Gender'].iloc[j]) for (i, j, k) in matches]
        print(f"Fraction of local gender preferences satisfied: {round(mean(gender_preference_local), 2)}")

        gender_preference_incoming = [int(self.incoming_students_copy['GenderPreference'].iloc[j] == 'No preference' or
                                          self.incoming_students_copy['GenderPreference'].iloc[j] ==
                                          self.local_students_original.loc[i, 'Gender']) for (i, j, k) in matches]
        print(f"Fraction of incoming gender preferences satisfied: {round(mean(gender_preference_incoming), 2)}")

        gender_preference_stat = [int(x + y == 2) for (x, y) in
                                  zip(gender_preference_local, gender_preference_incoming)]
        print(f"Fraction of gender preferences satisfied simultaneously: {round(mean(gender_preference_stat), 2)}")

        # Age Statistics
        age_difference = [abs(self.local_students_original.loc[i, 'Age'] - self.incoming_students_copy['Age'].iloc[j])
                          for (i, j, k) in matches]
        age_diff_exec = [int(x > 4) for x in age_difference]

        print(f"Average age difference (years): {round(mean(age_difference), 2)}")
        print(f"Fraction of age differences above 4 years: {round(mean(age_diff_exec), 2)}")
        print(f"Maximum age difference: {max(age_difference)}")

        # Availability Statistics

        self.local_students_original['Availability'] = pd.to_datetime(self.local_students_original['Availability'])

        unavailable_time = [max((self.local_students_original.loc[i, 'Availability'] -
                                 self.incoming_students_copy['Arrival'].iloc[j]).days, 0) for (i, j, k) in matches]
        unavailable_fraction = [int(x > 0) for x in unavailable_time]
        unavailable_time_among_incoming = [x for x in unavailable_time if x > 0]

        print(f"Fraction of local unavailable upon arrival of incoming: {round(mean(unavailable_fraction), 2)}")
        print(
            f"Average waiting time of incoming students (days): {round(mean(unavailable_time_among_incoming), 2) if not unavailable_time_among_incoming == [] else 0}")
        print(
            f"Max Waiting time (days): {max(unavailable_time_among_incoming) if not unavailable_time_among_incoming == [] else 0}")

        local_incoming_matches = [
            self.local_students_original.iloc[i, [1, 2, 3, 4, 7]].values.tolist() +
            self.incoming_students_original.iloc[j, [1, 2, 3, 4, 7]].values.tolist() + [round(k, 2)] for
            (i, j, k) in matches]
        columns = ["Local First Name", "Local Last Name", "Local Email", "Local Phone Number", "Local Country",
                   "Incoming First Name", "Incoming Last Name", "Incoming Email", "Incoming Phone Number",
                   "Incoming Country", "Match Score"]

        parent_dir = os.getcwd()
        os.chdir(os.path.join(parent_dir, output_dir))

        pd.DataFrame(local_incoming_matches, columns=columns).to_excel('matches.xlsx', index=False)

        if self.incoming_students_copy.shape[0] < self.local_students_copy.shape[0]:
            matched_local_students = [row[0] for row in matches]

            self.local_students_original['Matches'] = self.local_students_original['Matches'] + [
                matched_local_students.count(i) for i in range(0, len(self.local_students_original))]

            # Create a DataFrame with local students who have no incoming matched to them
            no_matches = self.local_students_original[self.local_students_original['Matches'] == 0]

            # Save the DataFrame to an Excel file
            no_matches.to_excel('local_students_no_matches.xlsx', index=False)

            # Create the condition for the second DataFrame
            condition = self.local_students_original['Matches'] >= self.local_students_original.loc[:, 'Capacity'] + (
                    self.local_students_original.loc[:, 'ExtraBuddy'] == 'Yes').astype(int)

            # Create a DataFrame with local students who meet the condition
            over_capacity_locals = self.local_students_original[condition]

            # Save the DataFrame to an Excel file
            over_capacity_locals.to_excel('local_students_over_capacity.xlsx', index=False)

            # Create a DataFrame with local students who don't meet the condition
            under_capacity_locals = self.local_students_original[~condition]

            # Save the DataFrame to an Excel file
            under_capacity_locals.to_excel('local_students_under_capacity.xlsx', index=False)
        elif self.incoming_students_copy.shape[0] > self.local_students_copy.shape[0]:
            matched_incoming = [row[1] for row in matches]
            remaining_incoming_students = list(set(self.local_students_original.index) - set(matched_incoming))
            self.incoming_students_original.iloc[remaining_incoming_students].to_excel(
                'incoming_students_unmatched.xlsx', index=False)

        time_taken = datetime.now() - start_time

        with open('report.md', 'w') as file:
            file.write(f"# Report for Buddy Matching Algorithm\n")
            file.write(f"## Date of running: {start_time.strftime('%d-%m-%Y')}\n")
            file.write(f"## Time taken to run: {time_taken} seconds\n\n")

            file.write(f"Local Students Available: {repr(self.local_students_copy.shape[0])}\n")
            file.write(f"Incoming Students Available: {repr(self.incoming_students_copy.shape[0])}\n")

            file.write(f"Average Distance: {repr(avg_distance)}\n\n")

            file.write(f"### Age Statistics\n")
            file.write(f"Average age difference (years): {round(mean(age_difference), 2)}\n")
            file.write(f"Fraction of age differences above 4 years: {round(mean(age_diff_exec), 2)}\n")
            file.write(f"Maximum age difference: {max(age_difference)}\n")

            file.write(f"### Gender Statistics\n")
            file.write(
                f"Fraction of local student gender preferences satisfied: {round(mean(gender_preference_local), 2)}\n")
            file.write(
                f"Fraction of incoming gender preferences satisfied: {round(mean(gender_preference_incoming), 2)}\n")

            file.write(f"### University Statistics\n")
            file.write(f"University Equal: {round(mean(university_equal), 2)}\n")
            file.write(f"Faculty Equal: {round(mean(faculty_equal), 2)}\n")

            file.write(f"### Availability Statistics\n")
            file.write(
                f"Fraction of local unavailable upon arrival of incoming: {round(mean(unavailable_fraction), 2)}\n")
            file.write(
                f"Average waiting time of incoming students (days): {round(mean(unavailable_time_among_incoming), 2) if not unavailable_time_among_incoming == [] else 0}\n")
            file.write(
                f"Max Waiting time (days): {max(unavailable_time_among_incoming) if not unavailable_time_among_incoming == [] else 0}\n")

        print(f"Report Generated\nTime Taken: {time_taken} seconds")


if __name__ == '__main__':
    algo = MatchingAlgorithm("hat")
    algo.run_algorithm()
