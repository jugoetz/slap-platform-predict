import datetime
import random


def generate_run_id():
    """
    Return an ID consisting of the current datetime and a 6-digit random part.

    Returns:
        str: id in the format "YYYY-MM-DD-HHMMSS_r" where r is a 6 digit random integer (100000-999999).
    """
    curr_datetime = datetime.datetime.now()
    return f"{curr_datetime.date()}-{curr_datetime.hour}{curr_datetime.minute}{curr_datetime.second}_{random.randint(100000, 999999)}"
