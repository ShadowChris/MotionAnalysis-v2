import csv


def get_all_event():
    content = []
    csv_file = csv.reader(open('data_event/VRevent.csv', 'r'))
    for line in csv_file:
        content.append(line)
    return content


# time unit: second (float)
def get_event_by_time(all_events, time):
    # an event would be like [time, event]
    last_event = [0, 0]
    for event in all_events:
        if time > float(event[0]):
            last_event = event
            continue

        # trace back to the last event
        return last_event[1]
    return 0
