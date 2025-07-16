import csv
with open('data\dataVelTruth.csv', 'r') as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter=',')
    data = [row for row in csv_reader]

