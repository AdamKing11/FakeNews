from fn_score import report_score

# without stemming/lemmas
actual = []
predicted = []

#####################################
#####################################
# make sure that this is the right file!
with open('ak_fn_tokens.csv', 'r') as rf:
	for line in rf:
		line = line.rstrip().rsplit('\t')
		predicted.append(line[0])
		actual.append(line[1])

report_score(actual, predicted)
