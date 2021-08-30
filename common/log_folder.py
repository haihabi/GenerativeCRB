import os
from datetime import datetime


def generate_log_folder(base_log_folder):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    run_folder = os.path.join(base_log_folder, date_time)
    os.makedirs(run_folder)
    return run_folder
