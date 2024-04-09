import truecase

fi = open("allqueries.txt", "r")
fo = open("allQueries.txt", "w")
for q in fi:
    fo.write(truecase.get_true_case(q))
    fo.write("\n")
fo.close()
fi.close()
