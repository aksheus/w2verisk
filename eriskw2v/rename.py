import os
import sys

"""
   USAGE : <source-dir> <dest-dir> <rename-tag>
"""

if __name__ == '__main__':
	for f in os.listdir(sys.argv[1]):
		with open(os.path.join(sys.argv[1],f),errors='ignore') as read:
			with open(os.path.join(sys.argv[2],f+sys.argv[3]+'.txt'),'w',encoding='utf-8',errors='ignore') as out:
				for line in read:
					out.write(line)
