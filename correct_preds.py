path_predict = "submissions/elated-aardvark-644-debug.csv"
values = []
with open(path_predict, 'r') as file:
    i = 0
    for line in file:
        if i == 0:
            i+=1
        else:
            line = line.split(',')
            line[-1] = float(line[-1])
            
            if line[-1]>0.9999 :
                pred  = 1 
            elif line[-1]<0.0001 :
                pred  = 0
            else:
                pred  = line[-1]
            tup = line[0],pred
            values.append(tup)
with open(path_predict.replace(".csv","modified.csv"),"w") as file:
    file.write("id,predicted\n")
    for value in values:
        file.write(f"{value[0]},{value[1]}\n")