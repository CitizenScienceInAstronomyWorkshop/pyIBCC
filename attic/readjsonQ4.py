# created  2013 Feb 23 by MES
# reads in all  Q4 simulation json files in directory  and outputs to Q4sim_positions.txt
# the start x and end x position of tne injected transit(s) if present 
# not all Q4 simulations jsons will have a planet transit in it so if there is no planet transit 
# present in a given json file nothing is outputted ofr it 


import json
import glob


fjsons=glob.glob("APH4*.json")

output=open("Q4sim_positions.txt", "w")


for fs in fjsons:
	print fs
	f=open(fs,'r')
	r=f.read()
	f.close()
	if r[0]=='l':
		r=r[17:].rstrip('\n;)')
	try:
		data=json.loads(r)
	except:
		print 'error reading json file'
		print r[:100]+'...'+r[-100:]
	print data[0]['tr']

	startx=-1.0
	endx=-1.0

	for i in range(len(data)):
		if ((startx < 0) and (data[i]['tr']==1)): # start of a transit 
			startx=data[i]['x']
			print data[i]
		if ((startx >= 0 ) and (data[i]['tr']==1)):  # keep updating endpoint 
			print data[i]
			endx=data[i]['x']

		if (endx >=0 and data[i]['tr']==0): # found the endpoint 
			# end of transit
			print data[i]
			print fs, startx, endx
			output.write(fs+" "+str(startx)+" "+str(endx)+"\n")
			endx=-1.0
			startx=-1.0


	if(endx>0):
		print fs, startx, endx	
		output.write(fs+" "+str(startx)+" "+str(endx)+"\n")

output.close()
