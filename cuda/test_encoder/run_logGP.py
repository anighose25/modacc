import sys
import subprocess
if __name__ == "__main__":
	sizes = [(2**i)/ 4 for i in range(2,31)]
	print sizes
	logGP_cmd="./logGP"
	output = {}
	for num_streams in [1,2,4,8,16,32]:
		for size in sizes:
			if(num_streams<=size):
				# print "run logGP: ",size*4," bytes ",num_streams," streams "
				logGP_arg=str(size)+" "+str(num_streams)
				filename = "logGPdumps/logGP_output_"+str(size)+"_"+str(num_streams)+'.txt'
				logGP_complete_command=logGP_cmd+" " + logGP_arg + " >> " + filename
				print logGP_complete_command
				# output[(size,num_streams)] = subprocess.Popen([logGP_cmd, logGP_arg], 
                          # stdout=subprocess.PIPE).communicate()[0]
	# print output