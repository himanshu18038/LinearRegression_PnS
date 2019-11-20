"""
Name-Himanshu Raj
Roll No-2018038
Section-A
Group-6
Branch-CSE
"""
from csv import reader,writer,QUOTE_ALL
from random import randint

def mean(x):								#for calculating mean of a set of values
	avg=sum(x)/len(x)
	return avg

def covariance(x,y):						#for calculation of covariance between 2 sets of values
	avgx=mean(x)
	avgy=mean(y)
	cov=0
	for i in range(len(x)):
		cov+=(x[i]-avgx)*(y[i]-avgy)
	cov=cov/len(x)
	return cov

def stdev(x):								#for calculation of standard deviation of a set x
	avgx=mean(x)
	sd=0
	for i in range(len(x)):
		sd+=(x[i]-avgx)**2
	sd=(sd/len(x))**0.5
	return sd

def correlation(x,y):						#for calculation of correlation between data-set x and y
	avgx=mean(x)
	avgy=mean(y)
	corr=covariance(x,y)/(stdev(x)*stdev(y))
	return corr

jour=open('found.txt','r')					#opening the data file in read mode

l=[["Name of Journal","H-Index","Impact Factor"]]
p=[]
q=[]
j=0

for i in reader(jour,quotechar='"', delimiter=';',quoting=QUOTE_ALL):		#reading the input file
	if(j==0):
		j=1
		continue
	l.append([i[0],i[1],i[2]])
	p.append(float(i[1]))
	q.append(float(i[2]))

f1=open('Journals.csv', 'w', newline='')												#Opening a new file
writer(f1).writerows(l)	

print('Correlation Coefficient for whole data = ',correlation(p,q))

x=[]
y=[]
xtest=[]
ytest=[]

lentrain=int(0.8*len(p))					#defining the size of train and test set
lentest=len(p)-lentrain

for i in range(len(p)):						#random distribution of values in train and test sets
	u=randint(0,1)
	if(u==0):
		if(len(x)<lentrain):
			x.append(p[i])
			y.append(q[i])
		else:
			xtest.append(p[i])
			ytest.append(q[i])	
	elif(u==1):
		if(len(xtest)<lentest):	
			xtest.append(p[i])
			ytest.append(q[i])
		else:
			x.append(p[i])
			y.append(q[i])

# x=p[:lentrain]							#for non-random distribution
# y=q[:lentrain]
# xtest=p[lentrain:]
# ytest=q[lentrain:]

avgx=mean(x)		
avgy=mean(y)

#Equation of line is y=ax+b, where a is slope and b is the y-intercept

a=correlation(x,y)*(stdev(y)/stdev(x))											#formula for calculating a and b
b=avgy-a*avgx

print('Regression line equation for train data-set is-')
print('Impact_Factor = ',a,' * H-Index + ',b)			

err=[]
for i in range(len(xtest)):
	err.append((a*xtest[i]+b-ytest[i])**2)										#calculating Mean Square error

mse=sum(err)/len(xtest)						
print('Mean Squared error = ',mse)

l=[["Name of Conference","H-Index","Predicted Imapct Factor"]]					#Setting the top row for headings
f2=open('ConferenceData.csv','r')												#Opened ConferenceData file in read mode for input
j=0
for i in reader(f2, quotechar='"', delimiter=';',quoting=QUOTE_ALL):
	if(j==0):
		j=1
		continue																#For ignoring the heading of .csv file
	impfac=a*float(i[7]) + b 													#calculating Impact Factor of Conference
	l.append([i[2],float(i[7]),impfac])

f3=open('ConferenceFinal.csv', 'w', newline='')												#Opening a new file
writer(f3).writerows(l)															#Writing data-set in Output file

print('Open ConferenceFinal.csv file to get the predicted Impact Factor of Conferences')
