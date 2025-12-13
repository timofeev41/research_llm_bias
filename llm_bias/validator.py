import csv

file = "llm_bias/qwen_summarize.csv"

cnt = 0

with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        if row[2] == "Qwen":
            if row[4] == "1":
                cnt += 1
            if row[5] == "1":
                cnt += 1
            if row[6] == "1":
                cnt += 1
            if row[7] == "1":
                cnt += 1
            if row[8] == "1":
                cnt += 1
            if row[9] == "1":
                cnt += 1
            if row[10] == "1":
                cnt += 1
            if row[11] == "1":
                cnt += 1
            print(row)
print(cnt)