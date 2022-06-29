from datetime import datetime

def elapsed_seconds(start_time, end_time):
    dt = end_time - start_time
    
    return (dt.days*24*60*60 + dt.seconds) + dt.microseconds/1000000.0
