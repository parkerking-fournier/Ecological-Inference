import csv
import numpy as np

# Write data to the files. 
def writeData(file_1, file_2, data, label, threshold):
	with open(file_1, 'a+') as f1:
		with open(file_2, 'a') as f2:
			wr1 = csv.writer(f1, quoting=csv.QUOTE_NONE)
			wr2 = csv.writer(f2, quoting=csv.QUOTE_NONE)
			for i in range( data.shape[0] ):
				wr1.writerow(data[i][threshold:])
				wr2.writerow(label)

def writeParameters(file, data):
	with open(file, 'a+') as f:
		wr = csv.writer(f, quoting=csv.QUOTE_NONE)
		wr.writerow(data)

def loadOneLine(file_path):
	data = ""
	with open(file_path, "rb") as f:
		reader = csv.reader(f)
		data = next(reader)
	data = np.asarray(data)
	data = data.astype(float)
	return data

def loadNLines(file_path, n):
	print "\nReading in data..."
	data = np.ndarray(shape=(n,100001), dtype=float)
	with open(file_path, "rb") as f:
		reader = csv.reader(f)
		for i in range(0, n):
			temp = next(reader)
			temp = np.asarray(temp)
			temp = temp.astype(float)
			data[i] = temp
	print "			Finished!\n"
	return data