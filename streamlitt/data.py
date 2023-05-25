from datetime import datetime


def time_conversion(time: str) -> int:
    """
    Convert time into seconds
    :param time: String representing time. Accepted format (%H:%M:%S)
    :return: Time represented in seconds
    """
    try:
        time_formatted = datetime.strptime(time, "%H:%M:%S").time()
        seconds = (time_formatted.hour * 60 + time_formatted.minute) * 60 + time_formatted.second
        return seconds
    except ValueError as err:
        print("Invalid time format")


def pace_conversion(pace: str) -> int:
    """
    Convert pace into seconds
    :param pace: String representing pace, excepted units minutes/km. Accepted format (%M:%S)
    :return: Running pace represented in seconds
    """
    try:
        pace_formatted = datetime.strptime(pace, "%M:%S").time()
        seconds = pace_formatted.minute * 60 + pace_formatted.second
        return seconds
    except ValueError as err:
        print("Invalid pace format")
